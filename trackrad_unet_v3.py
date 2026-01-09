#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrackRAD Tiny-UNet: 极速版 (Retrained for Speed)
================================================================

核心改进 (针对 Time Limit Exceeded):
1. 极致轻量化: Base Channel 64 -> 32, 移除最深层 (Down4), 移除 ASPP。
2. 移除复杂注意力: 去掉全连接层的 CBAM，改用轻量级 SEBlock 或纯卷积。
3. 混合精度训练: 保持 FP16。
4. 显存优化: 可以在更大的 Batch Size 下训练。

运行方式:
    python trackrad_unet_v2.py --mode train
"""

import os
import sys
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler

# 环境配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_IMPLICIT_PARALLEL"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('training_tiny.log', encoding='utf-8')]
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """配置 - 极速版"""
    hf_repo_id: str = "LMUK-RADONC-PHYS-RES/TrackRAD2025"
    hf_token = None
    cache_dir: str = "./TrackRAD2025_cache"
    output_dir: str = "outputs_tiny_v1"
    
    # --- 模型核心修改 ---
    image_size: int = 256
    base_channels: int = 32  # 🔥 从 64 降为 32 (参数量减少 75%)
    
    # 训练配置
    batch_size: int = 16     # 🔥 模型变小了，Batch Size 可以加大，训练更稳
    num_epochs: int = 150    # 多训几轮
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    
    # 损失
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    
    # 其他
    val_split: float = 0.15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    seed: int = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 数据加载 (保持不变，逻辑是正确的)
# =============================================================================

def download_patient_data(patient_id: str, config: Config) -> Optional[Dict[str, Path]]:
    from huggingface_hub import hf_hub_download
    cache_dir = Path(config.cache_dir)
    patient_dir = cache_dir / patient_id
    frames_local = patient_dir / "images" / f"{patient_id}_frames.mha"
    labels_local = patient_dir / "targets" / f"{patient_id}_labels.mha"
    
    if frames_local.exists() and labels_local.exists():
        if frames_local.stat().st_size > 1000:
            return {"frames": frames_local, "labels": labels_local}
    
    try:
        (patient_dir / "images").mkdir(parents=True, exist_ok=True)
        (patient_dir / "targets").mkdir(parents=True, exist_ok=True)
        
        frames_path = hf_hub_download(
            repo_id=config.hf_repo_id,
            filename=f"trackrad2025_labeled_training_data/{patient_id}/images/{patient_id}_frames.mha",
            repo_type="dataset", token=config.hf_token,
            local_dir=str(cache_dir), local_dir_use_symlinks=False
        )
        labels_path = hf_hub_download(
            repo_id=config.hf_repo_id,
            filename=f"trackrad2025_labeled_training_data/{patient_id}/targets/{patient_id}_labels.mha",
            repo_type="dataset", token=config.hf_token,
            local_dir=str(cache_dir), local_dir_use_symlinks=False
        )
        return {"frames": Path(frames_path), "labels": Path(labels_path)}
    except Exception as e:
        logger.warning(f"下载患者 {patient_id} 失败: {e}")
        return None


def get_all_patient_ids(config: Config) -> List[str]:
    from huggingface_hub import list_repo_files
    try:
        files = list_repo_files(config.hf_repo_id, repo_type="dataset", token=config.hf_token)
        patient_ids = set()
        for f in files:
            if "trackrad2025_labeled_training_data" in f:
                parts = f.split("/")
                for part in parts:
                    if part.startswith(("A_", "B_", "C_", "D_", "E_", "F_")) and len(part) == 5:
                        patient_ids.add(part)
        return sorted(list(patient_ids))
    except Exception as e:
        logger.error(f"获取患者列表失败: {e}")
        return []


class TrackRADDataset(Dataset):
    def __init__(self, patient_ids: List[str], config: Config, is_training: bool = True):
        self.config = config
        self.is_training = is_training
        self.samples = []
        self.patient_data = {}
        
        for pid in patient_ids:
            paths = download_patient_data(pid, config)
            if paths:
                try:
                    import SimpleITK as sitk
                    frames = sitk.GetArrayFromImage(sitk.ReadImage(str(paths["frames"])))
                    labels = sitk.GetArrayFromImage(sitk.ReadImage(str(paths["labels"])))
                    
                    frames, labels = self._preprocess(frames, labels)
                    self.patient_data[pid] = {"frames": frames, "labels": labels}
                    
                    T = frames.shape[0]
                    for t in range(1, T):
                        self.samples.append((pid, t))
                except Exception as e:
                    logger.warning(f"无法读取患者 {pid}: {e}")
        
        logger.info(f"加载 {len(self.patient_data)} 患者, {len(self.samples)} 样本")

    def _preprocess(self, frames: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T, H, W = frames.shape
        target_size = self.config.image_size
        
        frames = frames.astype(np.float32)
        # 批量归一化加速
        frames = np.nan_to_num(frames)
        for t in range(T):
            frame = frames[t]
            mx = frame.max()
            if mx > 1e-8:
                frames[t] = frame / mx
        
        if H != target_size or W != target_size:
            scale_h, scale_w = target_size / H, target_size / W
            frames_resized = np.zeros((T, target_size, target_size), dtype=np.float32)
            labels_resized = np.zeros((T, target_size, target_size), dtype=np.float32)
            for t in range(T):
                frames_resized[t] = zoom(frames[t], (scale_h, scale_w), order=1)
                labels_resized[t] = zoom(labels[t], (scale_h, scale_w), order=0)
            frames, labels = frames_resized, labels_resized
        
        return frames.astype(np.float32), (labels > 0.5).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid, t = self.samples[idx]
        data = self.patient_data[pid]
        
        frame = data["frames"][t]
        label = data["labels"][t]
        first_mask = data["labels"][0]
        prev_frame = data["frames"][t-1]
        prev_mask = data["labels"][t-1]
        
        if self.is_training:
            # 简单的翻转增强
            if random.random() < 0.5:
                frame = np.flip(frame, axis=1).copy()
                label = np.flip(label, axis=1).copy()
                first_mask = np.flip(first_mask, axis=1).copy()
                prev_frame = np.flip(prev_frame, axis=1).copy()
                prev_mask = np.flip(prev_mask, axis=1).copy()
        
        input_tensor = np.stack([frame, prev_frame, first_mask, prev_mask], axis=0)
        
        return {
            "input": torch.from_numpy(input_tensor).float(),
            "label": torch.from_numpy(label[np.newaxis]).float()
        }


# =============================================================================
# 轻量化模型组件
# =============================================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class SimpleBlock(nn.Module):
    """极简卷积块: 2层卷积，移除繁重的 Attention"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch)
        )
    def forward(self, x): return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = SimpleBlock(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # 使用双线性插值代替反卷积，减少参数和计算量
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = SimpleBlock(in_ch, out_ch) # in_ch 是拼接后的通道数
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# =============================================================================
# UNetV2 -> TinyUNet (替换掉原来的重型模型)
# =============================================================================

class UNetV2(nn.Module):
    """
    Tiny-UNet: 专为实时追踪设计
    - 移除了 ASPP (极耗时)
    - 移除了 Layer 4 (过深，特征图过小，收益低)
    - 减少了通道数
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_ch: int = 32):
        super().__init__()
        
        # 编码器 (仅下采样3次，最大通道 256)
        self.inc = SimpleBlock(in_channels, base_ch)        # 32
        self.down1 = Down(base_ch, base_ch * 2)             # 64
        self.down2 = Down(base_ch * 2, base_ch * 4)         # 128
        self.down3 = Down(base_ch * 4, base_ch * 8)         # 256 (Bottleneck)
        
        # 解码器
        self.up1 = Up(base_ch * 12, base_ch * 4) # 256 + 128 = 384 in -> 128 out
        self.up2 = Up(base_ch * 6, base_ch * 2)  # 128 + 64 = 192 in -> 64 out
        self.up3 = Up(base_ch * 3, base_ch)      # 64 + 32 = 96 in -> 32 out
        
        self.outc = nn.Conv2d(base_ch, out_channels, 1)
        
    def forward(self, x, return_deep: bool = False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        d1 = self.up1(x4, x3)
        d2 = self.up2(d1, x2)
        d3 = self.up3(d2, x1)
        
        out = self.outc(d3)
        
        # 保持接口兼容性 (training loop 需要这个返回值)
        if return_deep:
            return out, None
        return out


# =============================================================================
# 损失函数 (保持精简)
# =============================================================================

class CombinedLoss(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, deep_outputs=None):
        # 极简损失：BCE + Dice
        bce = self.bce(logits, targets)
        
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = 1 - (2. * intersection + 1) / (union + 1)
        
        total = bce + dice
        
        # 计算 dice score 仅用于日志
        with torch.no_grad():
            score = (2. * intersection + 1) / (union + 1)
            
        return {'total': total, 'dice_score': score}


# =============================================================================
# 训练器
# =============================================================================

class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, config: Config):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.criterion = CombinedLoss(config)
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate, 
                              weight_decay=config.weight_decay)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=config.learning_rate,
                                    epochs=config.num_epochs, steps_per_epoch=len(train_loader))
        self.scaler = GradScaler()
        self.best_dsc = 0.0
        os.makedirs(config.output_dir, exist_ok=True)

    def train(self):
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch in self.train_loader:
                inputs = batch['input'].to(self.config.device)
                labels = batch['label'].to(self.config.device)
                
                self.optimizer.zero_grad()
                with autocast():
                    outputs, _ = self.model(inputs, return_deep=True)
                    loss_dict = self.criterion(outputs, labels)
                
                self.scaler.scale(loss_dict['total']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                
                epoch_loss += loss_dict['total'].item()
            
            # 验证
            if (epoch + 1) % 5 == 0:
                val_dsc = self._validate()
                logger.info(f"Epoch {epoch+1} | Loss: {epoch_loss/len(self.train_loader):.4f} | Val DSC: {val_dsc:.4f}")
                
                if val_dsc > self.best_dsc:
                    self.best_dsc = val_dsc
                    torch.save(self.model.state_dict(), os.path.join(self.config.output_dir, 'best_model.pth'))
    
    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        dices = []
        for batch in self.val_loader:
            inputs = batch['input'].to(self.config.device)
            labels = batch['label'].to(self.config.device)
            preds = (torch.sigmoid(self.model(inputs)) > 0.5).float()
            
            inter = (preds * labels).sum(dim=(2,3))
            union = preds.sum(dim=(2,3)) + labels.sum(dim=(2,3))
            d = (2 * inter + 1) / (union + 1)
            dices.extend(d.cpu().tolist())
        return np.mean(dices)

# =============================================================================
# 推理类 (用于 eval 模式)
# =============================================================================
class SequenceInference:
    def __init__(self, model, config):
        self.model = model.to(config.device).eval()
        self.config = config
    
    @torch.no_grad()
    def predict_sequence(self, frames, first_mask):
        # 简单的推理逻辑，用于本地测试
        # 实际比赛中会使用 model.py 中的逻辑
        return np.zeros_like(frames) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train'])
    args = parser.parse_args()
    
    config = Config()
    set_seed(config.seed)
    
    if args.mode == 'train':
        patient_ids = get_all_patient_ids(config)
        random.shuffle(patient_ids)
        split = int(len(patient_ids) * 0.9)
        train_dl = DataLoader(TrackRADDataset(patient_ids[:split], config), 
                            batch_size=config.batch_size, shuffle=True, num_workers=4)
        val_dl = DataLoader(TrackRADDataset(patient_ids[split:], config, is_training=False), 
                          batch_size=config.batch_size, shuffle=False, num_workers=4)
        
        model = UNetV2(in_channels=4, out_channels=1, base_ch=config.base_channels)
        trainer = Trainer(model, train_dl, val_dl, config)
        trainer.train()

if __name__ == '__main__':
    main()
