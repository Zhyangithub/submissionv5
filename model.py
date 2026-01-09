from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# ===============================
# Paths
# ===============================
RESOURCE_PATH = Path("resources")
WEIGHTS_PATH = RESOURCE_PATH / "best_model.pth"

# 【修改 1】导入 v3
try:
    from trackrad_unet_v3 import UNetV2
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
        
        # ⚡️ 极速跳帧策略
        # 既然模型只有 7MB，速度极快，Skip=2 甚至 Skip=1 (不跳) 可能都行
        # 为了稳妥拿到分，我们先保持 Skip=3 或者 Skip=2
        SKIP_STEP = 2 
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 3. 模型加载
        # 【修改 2】必须与训练时一致：base_ch=32
        model = UNetV2(in_channels=4, out_channels=1, base_ch=32) 
        
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError("Weights missing")
            
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        model.to(device)
        
        if device.type == 'cuda':
            model.half()
            # 7MB 模型非常适合编译优化，但为了防止兼容性问题，先注释掉
            # model = torch.compile(model) 
        model.eval()

        # 4. 全局 GPU 预处理
        with torch.no_grad():
            dtype = torch.float16 if device.type == 'cuda' else torch.float32
            
            frames_tensor = torch.from_numpy(frames).unsqueeze(1).to(device, dtype=dtype)
            target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
            
            # 归一化
            frames_tensor = torch.nan_to_num(frames_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            B, C, h_raw, w_raw = frames_tensor.shape
            flat = frames_tensor.view(B, -1)
            mins = flat.min(dim=1, keepdim=True)[0]
            maxs = flat.max(dim=1, keepdim=True)[0]
            div = maxs - mins
            div[div < 1e-8] = 1.0
            frames_norm = ((flat - mins) / div).view(B, C, h_raw, w_raw)
            
            # 缩放
            frames_resized = F.interpolate(frames_norm, size=TARGET_SIZE, mode='bilinear', align_corners=False)
            first_mask_resized = F.interpolate(target_tensor, size=TARGET_SIZE, mode='nearest')

        # 5. 推理循环
        prev_frame_gpu = frames_resized[0]
        prev_mask_gpu = first_mask_resized.squeeze(0)
        first_mask_gpu = first_mask_resized.squeeze(0)
        
        preds_gpu_list = [prev_mask_gpu.unsqueeze(0)]

        for t in range(1, T):
            curr_frame_gpu = frames_resized[t]
            
            if t % SKIP_STEP != 0:
                preds_gpu_list.append(prev_mask_gpu.unsqueeze(0))
                prev_frame_gpu = curr_frame_gpu
                continue

            # 4通道输入
            inp = torch.cat([
                curr_frame_gpu,
                prev_frame_gpu,
                first_mask_gpu,
                prev_mask_gpu
            ], dim=0).unsqueeze(0)

            with torch.no_grad():
                out = model(inp, return_deep=False)
                prob = torch.sigmoid(out)

            current_pred = (prob > 0.5).float()
            smooth_pred = 0.7 * current_pred + 0.3 * prev_mask_gpu.unsqueeze(0)
            final_pred_gpu = (smooth_pred > 0.5).float().squeeze(0)

            preds_gpu_list.append(final_pred_gpu.unsqueeze(0))
            
            prev_frame_gpu = curr_frame_gpu
            prev_mask_gpu = final_pred_gpu

        # 6. 输出
        with torch.no_grad():
            all_preds_tensor = torch.cat(preds_gpu_list, dim=0).to(dtype)
            preds_orig = F.interpolate(all_preds_tensor, size=(H, W), mode='nearest')
            final_output = preds_orig.squeeze(1).cpu().numpy().astype(np.uint8)
            
        return final_output

    except Exception as e:
        # 生存模式：返回静态复制
        print(f"ERROR: {e}")
        try:
             return np.repeat(target[np.newaxis, ...], T, axis=0)
        except:
             return np.zeros(frames.shape, dtype=np.uint8)
