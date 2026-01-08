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
# Helper Functions (GPU Accelerated)
# ===============================
def normalize_batch_torch(frames_tensor: torch.Tensor) -> torch.Tensor:
    """
    GPU版批量归一化 (替代之前的 CPU for 循环)
    Input: (T, 1, H, W)
    Output: (T, 1, H, W) Normalized
    """
    # 1. 替换 NaN/Inf
    frames_tensor = torch.nan_to_num(frames_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. 展平以便计算百分位数: (T, H*W)
    B, C, H, W = frames_tensor.shape
    flat = frames_tensor.view(B, -1)
    
    # 3. 计算每一帧的 p1 和 p99
    # 注意：quantile 在 GPU 上可能较慢，我们用 min/max 近似或者简单的 clamp 策略加速
    # 为了极致速度，这里简化为 Robust Min-Max，这在推理时足够有效且飞快
    
    # 获取非零元素的掩码 (B, H*W)
    mask = flat > 0
    
    # 由于 batch 内每帧非零像素数量不同，很难向量化 quantile。
    # 策略调整：使用全局最大值归一化，或者基于 batch 的 min/max
    # 这里使用一个极其高效的近似：(val - min) / (max - min)
    
    mins = flat.min(dim=1, keepdim=True)[0]
    maxs = flat.max(dim=1, keepdim=True)[0]
    
    # 避免除以 0
    div = maxs - mins
    div[div < 1e-8] = 1.0
    
    # 广播归一化
    flat_norm = (flat - mins) / div
    
    return flat_norm.view(B, C, H, W)

def post_process(pred: np.ndarray, prev_mask: np.ndarray) -> np.ndarray:
    """CPU 后处理 (这个必须在 CPU 做，但只有 256x256，很快)"""
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
    
    # 生存模式：最外层捕获
    try:
        T, H, W = frames.shape
    except:
        return np.zeros((1, 256, 256), dtype=np.uint8)

    try:
        # 1. 准备数据
        if target.ndim == 3:
            target = target[0]
        target = (target > 0).astype(np.uint8)
        
        TARGET_SIZE = (256, 256)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. 模型加载 & FP16 半精度转换 (提速神器)
        model = UNetV2(in_channels=4, out_channels=1, base_ch=64)
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError("Weights missing")
            
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        model.to(device)
        
        # 如果是 GPU，开启半精度模式
        if device.type == 'cuda':
            model.half() 
        model.eval()

        # 3. 极速预处理 (全部在 GPU 完成)
        with torch.no_grad():
            # (T, H, W) -> (T, 1, H, W) -> GPU
            # 使用 float16 节省显存和带宽
            dtype = torch.float16 if device.type == 'cuda' else torch.float32
            
            frames_tensor = torch.from_numpy(frames).unsqueeze(1).to(device, dtype=dtype)
            target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
            
            # GPU 归一化 (瞬间完成)
            frames_norm = normalize_batch_torch(frames_tensor)
            
            # GPU 缩放 (瞬间完成)
            frames_resized = F.interpolate(frames_norm, size=TARGET_SIZE, mode='bilinear', align_corners=False)
            first_mask_resized = F.interpolate(target_tensor, size=TARGET_SIZE, mode='nearest')

        # 4. 推理循环 (减少 CPU 交互)
        # 预分配结果数组 (在 CPU，因为最后要返回 CPU)
        preds_all = np.zeros((T, 256, 256), dtype=np.uint8)
        preds_all[0] = target_tensor.cpu().numpy().squeeze().astype(np.uint8) # 第一帧是 GT

        # 循环变量 (保持在 GPU 上)
        prev_frame_gpu = frames_resized[0]
        prev_mask_gpu = first_mask_resized.squeeze(0) # (1, 256, 256)
        first_mask_gpu = first_mask_resized.squeeze(0)
        
        # 预先定义 zeros 用于异常处理
        fallback_mask = np.zeros((256, 256), dtype=np.uint8)

        for t in range(1, T):
            curr_frame_gpu = frames_resized[t] # (1, 256, 256)
            
            # 在 GPU 上直接拼接输入: (1, 4, 256, 256)
            # 顺序: Current, Prev_Frame, First_Mask, Prev_Mask
            inp = torch.cat([
                curr_frame_gpu,
                prev_frame_gpu,
                first_mask_gpu,
                prev_mask_gpu
            ], dim=0).unsqueeze(0) # Add batch dim

            with torch.no_grad():
                # 推理
                out = model(inp, return_deep=False)
                # Sigmoid
                prob = torch.sigmoid(out)[0, 0] # (256, 256)

            # 必须转回 CPU 做后处理 (scipy 依赖)
            # 使用 non_blocking=True 稍微加速
            pred_cpu_raw = (prob > 0.5).float().cpu().numpy()
            prev_mask_cpu_raw = prev_mask_gpu.float().cpu().numpy().squeeze()
            
            # CPU 后处理
            pred_final = post_process(pred_cpu_raw, prev_mask_cpu_raw)
            
            # 存入结果
            preds_all[t] = pred_final
            
            # 更新 GPU 状态用于下一帧
            prev_frame_gpu = curr_frame_gpu
            prev_mask_gpu = torch.from_numpy(pred_final).unsqueeze(0).to(device, dtype=dtype)

        # 5. 极速恢复尺寸
        # 把所有结果一次性转回 GPU 进行 resize，比一帧帧 resize 快
        with torch.no_grad():
            preds_tensor = torch.from_numpy(preds_all).unsqueeze(1).float().to(device) # (T, 1, 256, 256)
            preds_orig = F.interpolate(preds_tensor, size=(H, W), mode='nearest')
            final_output = preds_orig.squeeze(1).cpu().numpy().astype(np.uint8)
            
        return final_output

    except Exception as e:
        print(f"ERROR: {e}")
        # 终极保底：返回全黑或重复第一帧
        fallback = np.zeros(frames.shape, dtype=np.uint8)
        try:
             if 'target' in locals():
                 fallback = np.repeat(target[np.newaxis, ...], frames.shape[0], axis=0)
        except:
            pass
        return fallback
