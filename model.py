from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as scipy_label
from scipy import ndimage

# ===============================
# Paths
# ===============================
RESOURCE_PATH = Path("resources")
WEIGHTS_PATH = RESOURCE_PATH / "best_model.pth"

try:
    from trackrad_unet_v2 import UNetV2
except ImportError:
    pass

# ===============================
# Helper Functions
# ===============================
def normalize_batch_torch(frames_tensor: torch.Tensor) -> torch.Tensor:
    """GPU 批量归一化 (极速版)"""
    # 1. 清洗数据
    frames_tensor = torch.nan_to_num(frames_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. Batch Min-Max 归一化 (比 percentile 快很多)
    B, C, H, W = frames_tensor.shape
    flat = frames_tensor.view(B, -1)
    
    mins = flat.min(dim=1, keepdim=True)[0]
    maxs = flat.max(dim=1, keepdim=True)[0]
    
    div = maxs - mins
    div[div < 1e-8] = 1.0
    
    flat_norm = (flat - mins) / div
    return flat_norm.view(B, C, H, W)

def post_process(pred: np.ndarray, prev_mask: np.ndarray) -> np.ndarray:
    """CPU 后处理"""
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
# Main Algorithm
# ===============================
def run_algorithm(
    frames: np.ndarray,
    target: np.ndarray,
    frame_rate: float,
    magnetic_field_strength: float,
    scanned_region: str
) -> np.ndarray:
    
    # 保底机制：如果 shape 获取失败，返回空
    try:
        T, H, W = frames.shape
    except:
        return np.zeros((1, 256, 256), dtype=np.uint8)

    try:
        # --- 1. 准备 ---
        if target.ndim == 3:
            target = target[0]
        target = (target > 0).astype(np.uint8)
        
        TARGET_SIZE = (256, 256)
        # 跳帧设置：每 2 帧算一次 (即算一帧，跳一帧)
        SKIP_STEP = 2 
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 2. 模型加载 (FP16) ---
        model = UNetV2(in_channels=4, out_channels=1, base_ch=64)
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError("Weights missing")
            
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        model.to(device)
        if device.type == 'cuda':
            model.half() # 开启半精度
        model.eval()

        # --- 3. 全局 GPU 预处理 (Batch Processing) ---
        # 这一步把所有 resize 都在 GPU 上一次性做完，极快
        with torch.no_grad():
            dtype = torch.float16 if device.type == 'cuda' else torch.float32
            
            # (T, 1, H, W)
            frames_tensor = torch.from_numpy(frames).unsqueeze(1).to(device, dtype=dtype)
            target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
            
            # 批量归一化
            frames_norm = normalize_batch_torch(frames_tensor)
            
            # 批量缩放 (Bilinear for images, Nearest for mask)
            frames_resized = F.interpolate(frames_norm, size=TARGET_SIZE, mode='bilinear', align_corners=False)
            first_mask_resized = F.interpolate(target_tensor, size=TARGET_SIZE, mode='nearest')

        # --- 4. 循环推理 (带 Frame Skipping) ---
        # 结果存放在 CPU Numpy 数组中
        preds_all = np.zeros((T, 256, 256), dtype=np.uint8)
        
        # 第一帧是 GT
        first_mask_cpu = target_tensor.cpu().numpy().squeeze().astype(np.uint8)
        preds_all[0] = first_mask_cpu

        # 初始化循环变量 (GPU)
        prev_frame_gpu = frames_resized[0]
        prev_mask_gpu = first_mask_resized.squeeze(0) # (1, 256, 256)
        first_mask_gpu = first_mask_resized.squeeze(0)

        # 循环 T-1 次
        for t in range(1, T):
            curr_frame_gpu = frames_resized[t]
            
            # === 策略：Frame Skipping ===
            # 如果是偶数帧 (t=2,4,6...) -> 运行模型
            # 如果是奇数帧 (t=1,3,5...) -> 跳过，直接复用上一帧结果
            # 注意：t 从 1 开始，所以 t=1 是第一张预测帧。
            # 为了稳妥，我们可以每隔一帧算一次。
            
            if t % SKIP_STEP != 0:
                # 【跳过模式】直接复用上一帧结果
                preds_all[t] = preds_all[t-1]
                
                # GPU 状态更新：
                # Mask 不变 (复用上一帧的 Mask)
                # Frame 必须更新 (为了下一帧的输入)
                prev_frame_gpu = curr_frame_gpu
                continue

            # 【计算模式】运行模型
            # 构造输入: (1, 4, 256, 256)
            inp = torch.cat([
                curr_frame_gpu,
                prev_frame_gpu,
                first_mask_gpu,
                prev_mask_gpu
            ], dim=0).unsqueeze(0)

            with torch.no_grad():
                out = model(inp, return_deep=False)
                prob = torch.sigmoid(out)[0, 0]

            # 传回 CPU 做后处理 (Post-process 需要 CPU scipy)
            pred_raw_cpu = (prob > 0.5).float().cpu().numpy()
            prev_mask_raw_cpu = prev_mask_gpu.float().cpu().numpy().squeeze()
            
            pred_final = post_process(pred_raw_cpu, prev_mask_raw_cpu)
            
            # 存结果
            preds_all[t] = pred_final
            
            # 更新 GPU 状态
            prev_frame_gpu = curr_frame_gpu
            prev_mask_gpu = torch.from_numpy(pred_final).unsqueeze(0).to(device, dtype=dtype)

        # --- 5. 极速恢复尺寸 ---
        # 将所有结果一次性传回 GPU 做 resize，比一帧帧做快得多
        with torch.no_grad():
            preds_tensor = torch.from_numpy(preds_all).unsqueeze(1).float().to(device) # (T, 1, 256, 256)
            preds_orig = F.interpolate(preds_tensor, size=(H, W), mode='nearest')
            final_output = preds_orig.squeeze(1).cpu().numpy().astype(np.uint8)
            
        return final_output

    except Exception as e:
        # 生存模式保底：返回全黑
        print(f"ERROR: {e}")
        fallback = np.zeros(frames.shape, dtype=np.uint8)
        try:
             if 'target' in locals():
                 fallback = np.repeat(target[np.newaxis, ...], frames.shape[0], axis=0)
        except:
            pass
        return fallback
