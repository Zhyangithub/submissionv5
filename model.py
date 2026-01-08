from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import zoom, label as scipy_label
from scipy import ndimage

# ===============================
# Paths
# ===============================
RESOURCE_PATH = Path("resources")
WEIGHTS_PATH = RESOURCE_PATH / "best_model.pth"

try:
    from trackrad_unet_v2 import UNetV2
except ImportError:
    # 兼容本地测试
    pass

def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = frame.astype(np.float32)
    non_zero = frame[frame > 0]
    if len(non_zero) > 100:
        p_low, p_high = np.percentile(non_zero, [1, 99])
        frame = np.clip(frame, p_low, p_high)
        frame = (frame - p_low) / (p_high - p_low + 1e-8)
    else:
        frame = frame / (frame.max() + 1e-8)
    return frame

def post_process(pred: np.ndarray, prev_mask: np.ndarray) -> np.ndarray:
    if pred.sum() == 0:
        return prev_mask.copy()
    labeled, n = scipy_label(pred)
    if n > 0:
        sizes = ndimage.sum(pred, labeled, range(1, n + 1))
        largest = np.argmax(sizes) + 1
        pred = (labeled == largest).astype(np.float32)
    # 时序平滑
    pred = 0.7 * pred + 0.3 * prev_mask
    return (pred > 0.5).astype(np.uint8)

def run_algorithm(
    frames: np.ndarray,
    target: np.ndarray,
    frame_rate: float,
    magnetic_field_strength: float,
    scanned_region: str
) -> np.ndarray:
    
    # -----------------------------------------------------------
    # 【核心修正】相信 SimpleITK，输入就是 (T, H, W)
    # 不要 transpose(2, 1, 0)！
    # -----------------------------------------------------------
    
    # 确保 target 维度匹配
    # 如果 target 是 (1, H, W) 或 (T, H, W)，取第一帧作为初始 mask
    if target.ndim == 3:
        target = target[0] 
    
    target = (target > 0).astype(np.uint8)

    T, H, W = frames.shape
    image_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = UNetV2(in_channels=4, out_channels=1, base_ch=64)
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found at {WEIGHTS_PATH}")
        
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 预处理：缩放
    frames_norm = np.stack([normalize_frame(f) for f in frames])
    scale_h = image_size / H
    scale_w = image_size / W

    frames_resized = np.stack([
        zoom(frames_norm[t], (scale_h, scale_w), order=1)
        for t in range(T)
    ])
    first_mask_resized = zoom(target.astype(np.float32), (scale_h, scale_w), order=0)

    # 推理
    preds_resized = np.zeros((T, image_size, image_size), dtype=np.uint8)
    preds_resized[0] = first_mask_resized.astype(np.uint8)

    prev_frame = frames_resized[0]
    prev_mask = preds_resized[0]

    for t in range(1, T):
        inp = np.stack([
            frames_resized[t],
            prev_frame,
            first_mask_resized,
            prev_mask
        ], axis=0)

        inp = torch.from_numpy(inp[None]).float().to(device)

        with torch.no_grad():
            out = model(inp, return_deep=False)
            prob = torch.sigmoid(out)[0, 0].cpu().numpy()

        pred = (prob > 0.5).astype(np.uint8)
        pred = post_process(pred, prev_mask)

        preds_resized[t] = pred
        prev_frame = frames_resized[t]
        prev_mask = pred

    # 恢复尺寸
    preds = np.stack([
        zoom(preds_resized[t], (H / image_size, W / image_size), order=0)
        for t in range(T)
    ])
    
    # -----------------------------------------------------------
    # 输出保持 (T, H, W)，SimpleITK 会自动处理写入
    # -----------------------------------------------------------
    return preds.astype(np.uint8)
