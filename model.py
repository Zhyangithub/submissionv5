from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from scipy import ndimage
from scipy.ndimage import label as scipy_label

# ===============================
# Paths
# ===============================
RESOURCE_PATH = Path("resources")
WEIGHTS_PATH = RESOURCE_PATH / "best_model.pth"

# ===============================
# Model definition
# ===============================
# 依赖 Dockerfile 中的 COPY trackrad_unet_v2.py ...
try:
    from trackrad_unet_v2 import UNetV2
except ImportError:
    raise ImportError("CRITICAL: trackrad_unet_v2.py not found. Check your Dockerfile COPY instructions.")

# ===============================
# Utility functions
# ===============================
def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """与训练代码完全一致的百分位归一化"""
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
    """与训练代码一致的后处理"""
    if pred.sum() == 0:
        return prev_mask.copy()

    labeled, n = scipy_label(pred)
    if n > 0:
        sizes = ndimage.sum(pred, labeled, range(1, n + 1))
        largest = np.argmax(sizes) + 1
        pred = (labeled == largest).astype(np.float32)

    pred = 0.7 * pred + 0.3 * prev_mask
    return (pred > 0.5).astype(np.uint8)


# ===============================
# Main entrypoint (REQUIRED)
# ===============================
def run_algorithm(
    frames: np.ndarray,
    target: np.ndarray,
    frame_rate: float,
    magnetic_field_strength: float,
    scanned_region: str
) -> np.ndarray:
    """
    Args:
        frames: (W, H, T) from Official Inference
        target: (W, H, 1) from Official Inference
    """
    # [CRITICAL FIX 1] 维度对齐
    # 官方输入是 (W, H, T)，我们需要 (T, H, W) 进行处理
    frames = frames.transpose(2, 1, 0)
    target = target.transpose(2, 1, 0)

    # 处理 target 维度，确保它是 (H, W)
    if target.ndim == 3:
        target = target[0] 
    
    target = (target > 0).astype(np.uint8)

    # 现在 shape 是正确的 (Time, Height, Width)
    T, H, W = frames.shape
    image_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # Load model
    # -------------------------------
    # base_ch=64 必须与训练配置一致
    model = UNetV2(in_channels=4, out_channels=1, base_ch=64)
    
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found at {WEIGHTS_PATH}")
        
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    # -------------------------------
    # Preprocess frames (Vectorized)
    # -------------------------------
    # 逐帧归一化
    frames_norm = np.stack([normalize_frame(f) for f in frames])

    scale_h = image_size / H
    scale_w = image_size / W

    # Resize 到 256x256 (order=1 match training)
    frames_resized = np.stack([
        zoom(frames_norm[t], (scale_h, scale_w), order=1)
        for t in range(T)
    ])

    first_mask_resized = zoom(target.astype(np.float32), (scale_h, scale_w), order=0)

    # -------------------------------
    # Sequential inference
    # -------------------------------
    preds_resized = np.zeros((T, image_size, image_size), dtype=np.uint8)
    preds_resized[0] = first_mask_resized.astype(np.uint8)

    prev_frame = frames_resized[0]
    prev_mask = preds_resized[0]

    for t in range(1, T):
        # 构造输入: [Current, Prev_Frame, First_Mask, Prev_Mask]
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
        
        # 后处理
        pred = post_process(pred, prev_mask)

        preds_resized[t] = pred
        prev_frame = frames_resized[t]
        prev_mask = pred

    # -------------------------------
    # Resize back
    # -------------------------------
    preds = np.stack([
        zoom(preds_resized[t], (H / image_size, W / image_size), order=0)
        for t in range(T)
    ])
    
    # [CRITICAL FIX 2] 维度还原
    # (T, H, W) -> (W, H, T) 以符合官方输出要求
    preds = preds.transpose(2, 1, 0)

    return preds.astype(np.uint8)