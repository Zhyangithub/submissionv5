from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
# 移除 scipy 依赖，循环内不再使用 CPU
# from scipy.ndimage import label as scipy_label
# from scipy import ndimage

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
# Main Algorithm (Pure GPU Version)
# ===============================
def run_algorithm(
    frames: np.ndarray,
    target: np.ndarray,
    frame_rate: float,
    magnetic_field_strength: float,
    scanned_region: str
) -> np.ndarray:
    
    # 1. 基础保底
    try:
        T, H, W = frames.shape
    except:
        return np.zeros((1, 256, 256), dtype=np.uint8)

    try:
        # 2. 准备配置
        if target.ndim == 3:
            target = target[0]
        target = (target > 0).astype(np.uint8)
        
        TARGET_SIZE = (256, 256)
        
        # ⚡️ 激进优化：加大跳帧 (每3帧算一次)
        SKIP_STEP = 3 
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 3. 模型加载 (FP16)
        model = UNetV2(in_channels=4, out_channels=1, base_ch=64)
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError("Weights missing")
            
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        model.to(device)
        
        # ⚡️ 开启 CUDNN 加速 (针对固定尺寸输入非常有效)
        if device.type == 'cuda':
            model.half()
            torch.backends.cudnn.benchmark = True 
        model.eval()

        # 4. 全局 GPU 预处理 (一次性完成，绝不循环)
        with torch.no_grad():
            dtype = torch.float16 if device.type == 'cuda' else torch.float32
            
            # 搬运到 GPU
            frames_tensor = torch.from_numpy(frames).unsqueeze(1).to(device, dtype=dtype)
            target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
            
            # --- GPU 批量归一化 ---
            # 清洗
            frames_tensor = torch.nan_to_num(frames_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Min-Max 归一化
            B, C, h_raw, w_raw = frames_tensor.shape
            flat = frames_tensor.view(B, -1)
            mins = flat.min(dim=1, keepdim=True)[0]
            maxs = flat.max(dim=1, keepdim=True)[0]
            div = maxs - mins
            div[div < 1e-8] = 1.0
            frames_norm = ((flat - mins) / div).view(B, C, h_raw, w_raw)
            
            # --- GPU 批量缩放 ---
            frames_resized = F.interpolate(frames_norm, size=TARGET_SIZE, mode='bilinear', align_corners=False)
            first_mask_resized = F.interpolate(target_tensor, size=TARGET_SIZE, mode='nearest')

        # 5. 纯 GPU 推理循环 (无 CPU 交互)
        # 我们用一个 List 在 GPU 上暂存结果，最后再合并
        # 这样避免了每帧 `.cpu()` 的巨大开销
        
        # 初始状态
        prev_frame_gpu = frames_resized[0]
        prev_mask_gpu = first_mask_resized.squeeze(0) # (1, 256, 256)
        first_mask_gpu = first_mask_resized.squeeze(0)
        
        # 结果容器 (GPU Tensor)
        # 先放入第一帧 GT
        preds_gpu_list = [prev_mask_gpu.unsqueeze(0)] # List of (1, 1, 256, 256)

        for t in range(1, T):
            curr_frame_gpu = frames_resized[t]
            
            # === Frame Skipping ===
            if t % SKIP_STEP != 0:
                # 跳过：直接复用上一帧的 mask (还在 GPU 上)
                # 这里的 prev_mask_gpu 就是上一帧的结果
                preds_gpu_list.append(prev_mask_gpu.unsqueeze(0))
                
                # 更新 frame 用于下一帧输入
                prev_frame_gpu = curr_frame_gpu
                continue

            # === 计算模式 ===
            # 构造输入 (1, 4, 256, 256)
            inp = torch.cat([
                curr_frame_gpu,
                prev_frame_gpu,
                first_mask_gpu,
                prev_mask_gpu
            ], dim=0).unsqueeze(0)

            with torch.no_grad():
                out = model(inp, return_deep=False)
                prob = torch.sigmoid(out) # (1, 1, 256, 256)

            # --- GPU 内后处理 (替代 CPU Scipy) ---
            # 1. 阈值化
            current_pred = (prob > 0.5).float()
            
            # 2. 简单的时序平滑 (纯矩阵运算)
            # pred = 0.7 * curr + 0.3 * prev
            # 注意: prev_mask_gpu 是 (1, 256, 256)，需要 unsqueeze 匹配
            smooth_pred = 0.7 * current_pred + 0.3 * prev_mask_gpu.unsqueeze(0)
            final_pred_gpu = (smooth_pred > 0.5).float().squeeze(0) # (1, 256, 256)

            # 存入列表
            preds_gpu_list.append(final_pred_gpu.unsqueeze(0))
            
            # 更新状态 (全在 GPU，无需传输)
            prev_frame_gpu = curr_frame_gpu
            prev_mask_gpu = final_pred_gpu

        # 6. 终极合并与输出
        # 此时 preds_gpu_list 包含 T 个 (1, 1, 256, 256) 的 Tensor，全在显存里
        with torch.no_grad():
            # 拼接: (T, 1, 256, 256)
            all_preds_tensor = torch.cat(preds_gpu_list, dim=0).to(dtype)
            
            # 一次性 Resize 回原尺寸 (T, 1, H, W)
            # 这一步也是 GPU 批量操作，飞快
            preds_orig = F.interpolate(all_preds_tensor, size=(H, W), mode='nearest')
            
            # --- 唯一的 CPU 传输时刻 ---
            # 整个视频处理完后，只传输这一次
            final_output = preds_orig.squeeze(1).cpu().numpy().astype(np.uint8)
            
        return final_output

    except Exception as e:
        # 生存模式：出错返回全黑或静态复制，绝不报错退出
        print(f"ERROR: {e}")
        try:
             # 尝试简单的静态复制保底
             fallback = np.repeat(target[np.newaxis, ...], T, axis=0)
             return fallback
        except:
             return np.zeros(frames.shape, dtype=np.uint8)
