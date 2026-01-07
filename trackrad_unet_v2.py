#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrackRAD U-Net V2: 增强版肿瘤追踪方案
================================================================

核心改进:
1. 修复nan问题 - 梯度裁剪、数值稳定性处理
2. 加入注意力机制 - CBAM、空间注意力
3. 加入残差连接 - 帮助梯度流动
4. 加入深度监督 - 多尺度输出
5. 加入时序特征融合 - 利用前一帧信息
6. 使用Focal Loss - 更好处理类别不平衡

运行方式:
    python trackrad_unet_v2.py --mode train

"""

import os
import sys
import random
import logging
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom, binary_dilation, binary_erosion, label as scipy_label
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler

# 环境配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_IMPLICIT_PARALLEL"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('training_unet_v2.log', encoding='utf-8')]
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """配置"""
    hf_repo_id: str = "LMUK-RADONC-PHYS-RES/TrackRAD2025"
    hf_token = None
    cache_dir: str = "./TrackRAD2025_cache"
    output_dir: str = "outputs_unet_v2"
    
    # 模型
    image_size: int = 256
    base_channels: int = 64  # 增加通道数
    
    # 训练 - 更保守的设置防止nan
    batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 3e-4  # 降低学习率
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0  # 梯度裁剪
    
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
# 数据加载
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
    """数据集 - 增强版"""
    
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
        
        logger.info(f"加载 {len(self.patient_data)} 患者, {len(self.samples)} 样本 ({'训练' if is_training else '验证'})")

    def _preprocess(self, frames: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T, H, W = frames.shape
        target_size = self.config.image_size
        
        # 强度归一化 - 更稳定的方式
        frames = frames.astype(np.float32)
        for t in range(T):
            frame = frames[t]
            non_zero = frame[frame > 0]
            if len(non_zero) > 100:
                p_low, p_high = np.percentile(non_zero, [1, 99])
                frame = np.clip(frame, p_low, p_high)
                frame = (frame - p_low) / (p_high - p_low + 1e-8)
            else:
                frame = frame / (frame.max() + 1e-8)
            frames[t] = frame
        
        # 调整尺寸
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
        
        frame = data["frames"][t].copy()
        label = data["labels"][t].copy()
        first_mask = data["labels"][0].copy()
        
        # 前一帧信息
        prev_frame = data["frames"][t-1].copy()
        prev_mask = data["labels"][t-1].copy()
        
        # 数据增强
        if self.is_training:
            if random.random() < 0.5:
                frame = np.flip(frame, axis=1).copy()
                label = np.flip(label, axis=1).copy()
                first_mask = np.flip(first_mask, axis=1).copy()
                prev_frame = np.flip(prev_frame, axis=1).copy()
                prev_mask = np.flip(prev_mask, axis=1).copy()
            if random.random() < 0.5:
                frame = np.flip(frame, axis=0).copy()
                label = np.flip(label, axis=0).copy()
                first_mask = np.flip(first_mask, axis=0).copy()
                prev_frame = np.flip(prev_frame, axis=0).copy()
                prev_mask = np.flip(prev_mask, axis=0).copy()
            # 强度扰动
            if random.random() < 0.3:
                frame = frame * random.uniform(0.9, 1.1)
                frame = np.clip(frame, 0, 1)
                prev_frame = prev_frame * random.uniform(0.9, 1.1)
                prev_frame = np.clip(prev_frame, 0, 1)
        
        # 输入: [当前帧, 前一帧, 第一帧mask, 前一帧mask] -> 4通道
        input_tensor = np.stack([frame, prev_frame, first_mask, prev_mask], axis=0)
        
        return {
            "input": torch.from_numpy(input_tensor).float(),
            "label": torch.from_numpy(label[np.newaxis]).float(),
            "patient_id": pid,
            "frame_idx": t
        }


# =============================================================================
# 注意力模块
# =============================================================================

class ChannelAttention(nn.Module):
    """通道注意力"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False)
        )
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    """CBAM注意力模块"""
    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation块"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


# =============================================================================
# 模型模块
# =============================================================================

class ConvBNReLU(nn.Module):
    """卷积+BN+ReLU"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)


class DoubleConvWithAttention(nn.Module):
    """双卷积块 + 注意力"""
    def __init__(self, in_ch: int, out_ch: int, use_attention: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )
        self.residual = ResidualBlock(out_ch)
        self.attention = CBAM(out_ch) if use_attention else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.residual(x)
        x = self.attention(x)
        return x


class Down(nn.Module):
    """下采样模块"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvWithAttention(in_ch, out_ch)
    
    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """上采样模块 - 带跳跃连接注意力"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConvWithAttention(in_ch, out_ch)
        # 跳跃连接注意力
        self.skip_attention = nn.Sequential(
            nn.Conv2d(in_ch // 2, in_ch // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 尺寸对齐
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # 跳跃连接注意力
        x2 = x2 * self.skip_attention(x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ASPP(nn.Module):
    """空洞空间金字塔池化 - 多尺度特征"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # 1x1卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # 空洞卷积 rate=6
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # 空洞卷积 rate=12
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # 空洞卷积 rate=18
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # 全局池化
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        # 融合
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=False)
        
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.fuse(out)


# =============================================================================
# 主模型
# =============================================================================

class UNetV2(nn.Module):
    """增强版U-Net"""
    
    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_ch: int = 64):
        super().__init__()
        
        # 编码器
        self.inc = DoubleConvWithAttention(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)
        
        # 瓶颈层 - ASPP
        self.aspp = ASPP(base_ch * 16, base_ch * 16)
        
        # 解码器
        self.up1 = Up(base_ch * 16, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch)
        
        # 深度监督输出头
        self.deep_out4 = nn.Conv2d(base_ch * 8, out_channels, 1)
        self.deep_out3 = nn.Conv2d(base_ch * 4, out_channels, 1)
        self.deep_out2 = nn.Conv2d(base_ch * 2, out_channels, 1)
        
        # 最终输出
        self.outc = nn.Sequential(
            ConvBNReLU(base_ch, base_ch // 2),
            nn.Conv2d(base_ch // 2, out_channels, 1)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, return_deep: bool = True):
        # 编码
        x1 = self.inc(x)      # base_ch
        x2 = self.down1(x1)   # base_ch * 2
        x3 = self.down2(x2)   # base_ch * 4
        x4 = self.down3(x3)   # base_ch * 8
        x5 = self.down4(x4)   # base_ch * 16
        
        # 瓶颈
        x5 = self.aspp(x5)
        
        # 解码
        d4 = self.up1(x5, x4)  # base_ch * 8
        d3 = self.up2(d4, x3)  # base_ch * 4
        d2 = self.up3(d3, x2)  # base_ch * 2
        d1 = self.up4(d2, x1)  # base_ch
        
        # 主输出
        out = self.outc(d1)
        
        if return_deep and self.training:
            # 深度监督
            deep4 = F.interpolate(self.deep_out4(d4), size=out.shape[2:], mode='bilinear', align_corners=False)
            deep3 = F.interpolate(self.deep_out3(d3), size=out.shape[2:], mode='bilinear', align_corners=False)
            deep2 = F.interpolate(self.deep_out2(d2), size=out.shape[2:], mode='bilinear', align_corners=False)
            return out, [deep4, deep3, deep2]
        
        return out


# =============================================================================
# 损失函数
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)  # 数值稳定
        
        # Focal weight
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weight
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss - 数值稳定版"""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
        
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class BoundaryLoss(nn.Module):
    """边界损失"""
    def __init__(self):
        super().__init__()
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # 预测边界
        # 确保sobel算子与输入的数据类型一致
        sobel_x_typed = self.sobel_x.to(probs.device, dtype=probs.dtype)
        sobel_y_typed = self.sobel_y.to(probs.device, dtype=probs.dtype)

        pred_edge_x = F.conv2d(probs, sobel_x_typed, padding=1)
        pred_edge_y = F.conv2d(probs, sobel_y_typed, padding=1)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)
        
        # 目标边界
        tgt_edge_x = F.conv2d(targets.to(probs.dtype), sobel_x_typed, padding=1)
        tgt_edge_y = F.conv2d(targets.to(probs.dtype), sobel_y_typed, padding=1)
        tgt_edge = torch.sqrt(tgt_edge_x ** 2 + tgt_edge_y ** 2 + 1e-6)
        
        return F.mse_loss(pred_edge, tgt_edge)


class CombinedLoss(nn.Module):
    """组合损失"""
    def __init__(self, config: Config):
        super().__init__()
        self.focal = FocalLoss(config.focal_alpha, config.focal_gamma)
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                deep_outputs: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        
        # 主损失
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        boundary_loss = self.boundary(logits, targets)
        
        # 深度监督损失
        deep_loss = torch.tensor(0.0, device=logits.device)
        if deep_outputs is not None:
            for i, deep_out in enumerate(deep_outputs):
                weight = 0.5 ** (i + 1)  # 递减权重
                deep_loss += weight * (self.focal(deep_out, targets) + self.dice(deep_out, targets))
        
        # 计算Dice分数用于监控
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            probs_flat = probs.view(-1)
            targets_flat = targets.view(-1)
            intersection = (probs_flat * targets_flat).sum()
            union = probs_flat.sum() + targets_flat.sum()
            dice_score = (2 * intersection + 1) / (union + 1)
        
        total = focal_loss + dice_loss + 0.5 * boundary_loss + 0.4 * deep_loss
        
        # 检查nan
        if torch.isnan(total):
            logger.warning("检测到nan损失，使用备用损失")
            total = F.binary_cross_entropy_with_logits(logits, targets)
        
        return {
            'total': total,
            'focal': focal_loss,
            'dice': dice_loss,
            'boundary': boundary_loss,
            'deep': deep_loss,
            'dice_score': dice_score
        }


# =============================================================================
# 评估指标
# =============================================================================

def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2 * intersection / union


def center_distance(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 or target.sum() == 0:
        return float('inf')
    pred_coords = np.array(np.where(pred)).T
    target_coords = np.array(np.where(target)).T
    pred_centroid = pred_coords.mean(axis=0)
    target_centroid = target_coords.mean(axis=0)
    return np.sqrt(((pred_centroid - target_centroid) ** 2).sum())


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
        
        # OneCycleLR - 更稳定的学习率调度
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        self.scaler = GradScaler()
        self.best_dsc = 0.0
        os.makedirs(config.output_dir, exist_ok=True)

    def train(self):
        logger.info(f"开始训练... 设备: {self.config.device}")
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            total_dice = 0
            nan_count = 0
            
            for i, batch in enumerate(self.train_loader):
                inputs = batch['input'].to(self.config.device)
                labels = batch['label'].to(self.config.device)
                
                self.optimizer.zero_grad()
                
                with autocast():
                    outputs, deep_outputs = self.model(inputs, return_deep=True)
                    losses = self.criterion(outputs, labels, deep_outputs)
                
                # 检查nan
                if torch.isnan(losses['total']):
                    nan_count += 1
                    if nan_count > 10:
                        logger.warning(f"连续{nan_count}次nan，跳过此batch")
                    continue
                
                self.scaler.scale(losses['total']).backward()
                
                # 梯度裁剪 - 关键！
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                
                total_loss += losses['total'].item()
                total_dice += losses['dice_score'].item()
                nan_count = 0  # 重置
                
                if i % 100 == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    logger.info(f"Epoch {epoch+1} [{i}/{len(self.train_loader)}] | "
                               f"Loss: {losses['total'].item():.4f} | "
                               f"Dice: {losses['dice_score'].item():.4f} | "
                               f"LR: {lr:.2e}")
            
            avg_loss = total_loss / max(len(self.train_loader) - nan_count, 1)
            avg_dice = total_dice / max(len(self.train_loader) - nan_count, 1)
            
            # 验证
            val_metrics = self._validate()
            
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} | "
                       f"Train Loss: {avg_loss:.4f} | Train Dice: {avg_dice:.4f} | "
                       f"Val DSC: {val_metrics['dsc']:.4f} | Val CD: {val_metrics['cd']:.2f}mm")
            
            # 保存最佳模型
            if val_metrics['dsc'] > self.best_dsc:
                self.best_dsc = val_metrics['dsc']
                torch.save(self.model.state_dict(), 
                          os.path.join(self.config.output_dir, 'best_model.pth'))
                logger.info(f"保存最佳模型, DSC: {self.best_dsc:.4f}")
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), 
                          os.path.join(self.config.output_dir, f'model_epoch_{epoch+1}.pth'))
        
        logger.info(f"训练完成! 最佳DSC: {self.best_dsc:.4f}")

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        all_dsc = []
        all_cd = []
        
        for batch in self.val_loader:
            inputs = batch['input'].to(self.config.device)
            labels = batch['label'].numpy()
            
            outputs = self.model(inputs, return_deep=False)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            
            for b in range(preds.shape[0]):
                pred = preds[b, 0]
                target = labels[b, 0]
                
                dsc = dice_score(pred, target)
                cd = center_distance(pred, target)
                
                all_dsc.append(dsc)
                if cd != float('inf'):
                    all_cd.append(cd)
        
        return {
            'dsc': np.mean(all_dsc) if all_dsc else 0.0,
            'cd': np.mean(all_cd) if all_cd else float('inf')
        }


# =============================================================================
# 序列推理
# =============================================================================

class SequenceInference:
    """序列推理"""
    
    def __init__(self, model: nn.Module, config: Config):
        self.model = model.to(config.device)
        self.model.eval()
        self.config = config
    
    @torch.no_grad()
    def predict_sequence(self, frames: np.ndarray, first_mask: np.ndarray) -> np.ndarray:
        T, H, W = frames.shape
        target_size = self.config.image_size
        
        # 预处理
        frames_norm = frames.astype(np.float32)
        for t in range(T):
            frame = frames_norm[t]
            non_zero = frame[frame > 0]
            if len(non_zero) > 100:
                p_low, p_high = np.percentile(non_zero, [1, 99])
                frame = np.clip(frame, p_low, p_high)
                frame = (frame - p_low) / (p_high - p_low + 1e-8)
            else:
                frame = frame / (frame.max() + 1e-8)
            frames_norm[t] = frame
        
        # 调整尺寸
        scale_h, scale_w = target_size / H, target_size / W
        frames_resized = np.zeros((T, target_size, target_size), dtype=np.float32)
        for t in range(T):
            frames_resized[t] = zoom(frames_norm[t], (scale_h, scale_w), order=1)
        first_mask_resized = zoom(first_mask.astype(float), (scale_h, scale_w), order=0)
        
        # 预测
        predictions = np.zeros((T, target_size, target_size), dtype=np.float32)
        predictions[0] = first_mask_resized
        
        prev_frame = frames_resized[0].copy()
        prev_mask = first_mask_resized.copy()
        
        for t in range(1, T):
            # 4通道输入
            input_tensor = np.stack([
                frames_resized[t], 
                prev_frame,
                first_mask_resized, 
                prev_mask
            ], axis=0)
            input_tensor = torch.from_numpy(input_tensor[np.newaxis]).float().to(self.config.device)
            
            output = self.model(input_tensor, return_deep=False)
            pred = (torch.sigmoid(output) > 0.5).cpu().numpy()[0, 0]
            
            # 后处理
            pred = self._post_process(pred, prev_mask)
            
            predictions[t] = pred
            prev_frame = frames_resized[t].copy()
            prev_mask = pred.copy()
        
        # 恢复原始尺寸
        predictions_orig = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            predictions_orig[t] = zoom(predictions[t], (H / target_size, W / target_size), order=0)
        
        return predictions_orig > 0.5
    
    def _post_process(self, pred: np.ndarray, prev_mask: np.ndarray) -> np.ndarray:
        if pred.sum() == 0:
            return prev_mask.copy()
        
        # 保留最大连通域
        labeled, n = scipy_label(pred)
        if n > 0:
            sizes = ndimage.sum(pred, labeled, range(1, n + 1))
            largest = np.argmax(sizes) + 1
            pred = (labeled == largest).astype(float)
        
        # 时序平滑
        pred = 0.7 * pred + 0.3 * prev_mask
        pred = (pred > 0.5).astype(float)
        
        return pred


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    config = Config()
    set_seed(config.seed)
    
    logger.info("=" * 60)
    logger.info("TrackRAD U-Net V2 - 增强版")
    logger.info("=" * 60)
    logger.info(f"图像尺寸: {config.image_size}")
    logger.info(f"基础通道: {config.base_channels}")
    logger.info(f"学习率: {config.learning_rate}")
    logger.info(f"梯度裁剪: {config.max_grad_norm}")
    logger.info(f"Focal Loss: alpha={config.focal_alpha}, gamma={config.focal_gamma}")
    logger.info("=" * 60)
    
    if args.mode == 'train':
        patient_ids = get_all_patient_ids(config)
        logger.info(f"发现 {len(patient_ids)} 个患者")
        
        random.shuffle(patient_ids)
        val_n = max(1, int(len(patient_ids) * config.val_split))
        train_ids = patient_ids[val_n:]
        val_ids = patient_ids[:val_n]
        
        train_ds = TrackRADDataset(train_ids, config, is_training=True)
        val_ds = TrackRADDataset(val_ids, config, is_training=False)
        
        train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, 
                             num_workers=config.num_workers, pin_memory=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                           num_workers=config.num_workers, pin_memory=True)
        
        model = UNetV2(in_channels=4, out_channels=1, base_ch=config.base_channels)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型参数: {total_params/1e6:.2f}M")
        
        trainer = Trainer(model, train_dl, val_dl, config)
        trainer.train()
    
    elif args.mode == 'eval':
        if args.checkpoint is None:
            logger.error("需要指定checkpoint!")
            return
        
        model = UNetV2(in_channels=4, out_channels=1, base_ch=config.base_channels)
        model.load_state_dict(torch.load(args.checkpoint, map_location=config.device))
        
        patient_ids = get_all_patient_ids(config)
        
        all_dsc = []
        all_cd = []
        
        inference = SequenceInference(model, config)
        
        for pid in patient_ids:
            paths = download_patient_data(pid, config)
            if not paths:
                continue
            
            import SimpleITK as sitk
            frames = sitk.GetArrayFromImage(sitk.ReadImage(str(paths["frames"])))
            labels = sitk.GetArrayFromImage(sitk.ReadImage(str(paths["labels"])))
            
            first_mask = labels[0]
            preds = inference.predict_sequence(frames, first_mask)
            
            patient_dsc = []
            patient_cd = []
            
            for t in range(1, len(frames)):
                dsc = dice_score(preds[t], labels[t])
                cd = center_distance(preds[t], labels[t])
                patient_dsc.append(dsc)
                if cd != float('inf'):
                    patient_cd.append(cd)
                    all_cd.append(cd)
                all_dsc.append(dsc)
            
            logger.info(f"患者 {pid}: DSC={np.mean(patient_dsc):.4f}, "
                       f"CD={np.mean(patient_cd) if patient_cd else float('inf'):.2f}mm")
        
        logger.info("=" * 60)
        logger.info(f"总体 DSC: {np.mean(all_dsc):.4f} ± {np.std(all_dsc):.4f}")
        logger.info(f"总体 CD: {np.mean(all_cd):.2f} ± {np.std(all_cd):.2f} mm")
        logger.info("=" * 60)
        logger.info(f"目标: DSC > 0.891, CD < 1.47mm")


if __name__ == '__main__':
    main()
