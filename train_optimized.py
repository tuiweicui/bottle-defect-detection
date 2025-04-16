#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torchvision.models import wide_resnet50_2

import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from PIL import Image
import cv2
import matplotlib.pyplot as plt

# ==========================
# 设置命令行参数
# ==========================
def parse_args():
    parser = argparse.ArgumentParser(description='瓶子印刷质量检测模型训练')
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='数据集根目录')
    parser.add_argument('--category', type=str, default='bottle',
                        help='要训练的类别')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备 (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='保存结果的目录')
    parser.add_argument('--flow_steps', type=int, default=16,
                        help='FastFlow步数')
    parser.add_argument('--region_aware', action='store_true',
                        help='启用区域感知训练')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='启用混合精度训练')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='梯度累积步数')
    
    # 新增参数
    parser.add_argument('--cosine_lr', action='store_true',
                      help='使用余弦退火学习率调度')
    parser.add_argument('--center_crop', action='store_true',
                      help='只使用中心区域进行训练(解决边缘模糊问题)')
    parser.add_argument('--center_crop_ratio', type=float, default=0.8,
                      help='中心裁剪比例，默认为0.8(保留中心80%区域)')
    parser.add_argument('--focus_factor', type=float, default=2.0,
                      help='文本区域关注因子，控制文本区域的权重')
    parser.add_argument('--viz_freq', type=int, default=10,
                      help='可视化频率(每N个epoch)')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                      help='早停耐心值，默认20个epoch无改善则停止')
    
    return parser.parse_args()

# ==========================
# 数据增强和数据加载
# ==========================
class BottleDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, category, transform=None, is_train=True, center_crop=False, center_crop_ratio=0.8):
        self.root_dir = root_dir
        self.category = category
        self.transform = transform
        self.is_train = is_train
        self.center_crop = center_crop
        self.center_crop_ratio = center_crop_ratio
        
        # 设置图像路径
        split = 'train' if is_train else 'test'
        self.image_dir = os.path.join(root_dir, category, split, 'good')
        
        # 获取图像文件列表
        self.image_paths = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) 
                           if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"{'训练' if is_train else '测试'}集加载了 {len(self.image_paths)} 张图像")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 对边缘模糊的图像进行中心裁剪预处理
        if self.center_crop:
            h, w = image.shape[:2]
            # 计算裁剪区域
            crop_h = int(h * self.center_crop_ratio)
            crop_w = int(w * self.center_crop_ratio)
            # 计算裁剪的起始点
            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
            # 裁剪
            image = image[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # 训练和测试集都是正常样本
        label = 0  # 正常标签
        
        return {
            'image': image,
            'label': label,
            'path': img_path
        }

def get_augmentations(input_size=448):
    """创建增强的数据增强策略"""
    train_transform = A.Compose([
        # 基础处理
        A.Resize(height=input_size, width=input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # 几何变换 - 模拟不同拍摄角度
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5),
            A.Affine(scale=(0.95, 1.05), translate_percent=0.05, rotate=(-5, 5)),
            # 新增：透视变换，模拟不同视角
            A.Perspective(scale=(0.02, 0.05), p=0.3),
        ], p=0.6),  # 增加概率
        
        # 亮度对比度变化 - 模拟不同光照条件
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            A.ColorJitter(brightness=0.1, contrast=0.1),
            # 新增：锐化和对比度自适应调整
            A.Sharpen(alpha=(0.1, 0.3), p=0.5),
            A.CLAHE(clip_limit=(1, 3), p=0.3),
        ], p=0.7),  # 增加概率
        
        # 针对边缘模糊问题的专门处理
        A.OneOf([
            # 中心清晰，边缘模糊增强
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.3),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.MotionBlur(blur_limit=(3, 5), p=0.1),
            # 新增：径向模糊，模拟手机照片边缘模糊特性
            A.Lambda(
                name="RadialBlur",
                image=lambda x, **kwargs: add_radial_blur(x),
                p=0.3
            ),
        ], p=0.3),
        
        # 噪声添加 - 提高模型鲁棒性
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 15.0)),
            A.MultiplicativeNoise(multiplier=(0.95, 1.05)),
            # 新增：泊松噪声，更贴近真实相机噪声
            A.ISONoise(intensity=(0.1, 0.3), p=0.3),
        ], p=0.2),
        
        # 转换为张量
        ToTensorV2(),
    ])
    
    # 测试转换保持简单
    test_transform = A.Compose([
        A.Resize(height=input_size, width=input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, test_transform

def create_dataloaders(args):
    """创建训练和测试数据加载器"""
    train_transform, test_transform = get_augmentations(input_size=448)
    
    train_dataset = BottleDataset(
        root_dir=args.data_dir,
        category=args.category,
        transform=train_transform,
        is_train=True,
        center_crop=args.center_crop,
        center_crop_ratio=args.center_crop_ratio
    )
    
    test_dataset = BottleDataset(
        root_dir=args.data_dir,
        category=args.category,
        transform=test_transform,
        is_train=False,
        center_crop=args.center_crop,
        center_crop_ratio=args.center_crop_ratio
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader

# ==========================
# FastFlow模型定义
# ==========================
class FastFlowBlock(nn.Module):
    def __init__(self, input_dim, hidden_ratio, conv3x3_only, use_dropout=False, dropout_rate=0.2):
        super(FastFlowBlock, self).__init__()
        
        hidden_dim = int(input_dim * hidden_ratio)
        
        # 第一个1x1卷积层或3x3卷积层
        if conv3x3_only:
            self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(input_dim, hidden_dim, 1)
        
        # 第二个3x3卷积层
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
        # 第三个1x1卷积层，用于输出结果
        self.conv3 = nn.Conv2d(hidden_dim, input_dim, 1)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        
        # Dropout层
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        residual = x
        
        # 第一个卷积
        out = self.conv1(x)
        out = self.relu(out)
        
        # 第二个卷积
        out = self.conv2(out)
        out = self.relu(out)
        
        # Dropout层
        if self.use_dropout:
            out = self.dropout(out)
        
        # 第三个卷积
        out = self.conv3(out)
        
        # 添加残差连接
        out = out + residual
        
        return out

class FastFlow(nn.Module):
    def __init__(self, backbone="wide_resnet50_2", flow_steps=8, input_size=448, 
                 hidden_ratio=1.0, conv3x3_only=True, use_dropout=False, dropout_rate=0.2):
        super(FastFlow, self).__init__()
        
        # 加载预训练的骨干网络
        if backbone == "wide_resnet50_2":
            self.backbone = wide_resnet50_2(pretrained=True)
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        # 去掉全连接层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # 特征维度 (wide_resnet50_2的输出通道为2048)
        feature_dims = [256, 512, 1024, 2048]
        
        # 为每个特征层创建FastFlow块
        self.flow_blocks = nn.ModuleList()
        
        for dim in feature_dims:
            flow_block = nn.ModuleList()
            for _ in range(flow_steps):
                flow_block.append(FastFlowBlock(
                    input_dim=dim, 
                    hidden_ratio=hidden_ratio,
                    conv3x3_only=conv3x3_only,
                    use_dropout=use_dropout,
                    dropout_rate=dropout_rate
                ))
            self.flow_blocks.append(flow_block)
        
        # 记录超参数
        self.input_size = input_size
        self.feature_dims = feature_dims
    
    def forward(self, x):
        # 如果输入是字典格式
        if isinstance(x, dict):
            x = x['image']
        
        # 提取特征 (4个尺度)
        features = []
        
        # 第一个块 (layer1)
        x = self.backbone[0](x)  # 卷积层
        x = self.backbone[1](x)  # BatchNorm
        x = self.backbone[2](x)  # ReLU
        x = self.backbone[3](x)  # MaxPool
        x = self.backbone[4](x)  # layer1
        features.append(x)
        
        # 剩余的块
        x = self.backbone[5](x)  # layer2
        features.append(x)
        x = self.backbone[6](x)  # layer3
        features.append(x)
        x = self.backbone[7](x)  # layer4
        features.append(x)
        
        # 应用FastFlow块
        nll_total = 0
        feat_maps = []
        
        for i, feature in enumerate(features):
            # 每个特征图应用对应的FastFlow块
            feat_nll = torch.zeros(feature.shape[0], device=feature.device)
            feat_map = torch.zeros_like(feature)
            
            for flow_step in self.flow_blocks[i]:
                out = flow_step(feature)
                # 计算NLL损失
                diff = out - feature
                feat_nll += 0.5 * (diff ** 2).sum(dim=(1, 2, 3))
                feat_map += diff.abs().sum(dim=1, keepdim=True)
                feature = out
            
            # 归一化特征图尺寸
            B, C, H, W = feat_map.shape
            feat_map = feat_map / (C * H * W)
            feat_maps.append(feat_map)
            
            # 累加NLL损失
            nll_total += feat_nll
        
        # 上采样所有特征图到相同大小进行融合
        anomaly_map = torch.zeros_like(feat_maps[0])
        for feat_map in feat_maps:
            # 上采样到第一个特征图的大小
            if feat_map.shape != anomaly_map.shape:
                feat_map = nn.functional.interpolate(
                    feat_map, 
                    size=anomaly_map.shape[2:],
                    mode='bilinear', 
                    align_corners=False
                )
            anomaly_map += feat_map
        
        # 归一化异常图
        anomaly_map = anomaly_map / len(feat_maps)
        
        # 计算异常分数 (图像级)
        anomaly_score = torch.mean(anomaly_map, dim=(1, 2, 3))
        
        return {
            'anomaly_map': anomaly_map,
            'anomaly_score': anomaly_score,
            'nll_loss': nll_total,
            'features': features
        }

# ==========================
# 区域检测和权重函数
# ==========================
def merge_overlapping_boxes(boxes, overlap_threshold=0.3):
    """合并重叠或接近的边界框"""
    if not boxes:
        return []
    
    # 转换为numpy数组以便计算
    boxes_array = np.array(boxes)
    
    # 初始化结果列表
    merged_boxes = []
    
    while len(boxes_array) > 0:
        # 选择当前面积最大的边界框
        areas = (boxes_array[:, 2] - boxes_array[:, 0]) * (boxes_array[:, 3] - boxes_array[:, 1])
        max_idx = np.argmax(areas)
        current_box = boxes_array[max_idx]
        boxes_array = np.delete(boxes_array, max_idx, axis=0)
        
        # 找出与当前边界框重叠的所有边界框
        to_merge_idx = []
        for i, box in enumerate(boxes_array):
            # 计算重叠区域
            x1 = max(current_box[0], box[0])
            y1 = max(current_box[1], box[1])
            x2 = min(current_box[2], box[2])
            y2 = min(current_box[3], box[3])
            
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            overlap_area = w * h
            
            # 计算两个边界框的面积
            area1 = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            area2 = (box[2] - box[0]) * (box[3] - box[1])
            
            # 计算重叠比例
            overlap_ratio = overlap_area / min(area1, area2)
            
            # 如果重叠比例超过阈值，则合并
            if overlap_ratio > overlap_threshold:
                to_merge_idx.append(i)
        
        # 合并边界框
        for idx in sorted(to_merge_idx, reverse=True):
            box = boxes_array[idx]
            # 创建新的合并边界框
            merged_box = [
                min(current_box[0], box[0]),
                min(current_box[1], box[1]),
                max(current_box[2], box[2]),
                max(current_box[3], box[3])
            ]
            current_box = merged_box
            # 从待处理列表中删除已合并的边界框
            boxes_array = np.delete(boxes_array, idx, axis=0)
        
        merged_boxes.append(tuple(current_box))
    
    return merged_boxes

def detect_text_regions(images):
    """检测图像中的文本区域，使用综合方法"""
    if isinstance(images, torch.Tensor):
        if images.ndim == 4:  # 批量图像
            images_np = images.detach().cpu().numpy()
            batch_regions = []
            
            for img in images_np:
                img = np.transpose(img, (1, 2, 0))  # 转换为(H, W, C)
                img = ((img * 0.229 + 0.485) * 255).astype(np.uint8)  # 反归一化
                
                # 1. 边缘检测方法
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
                
                # 2. 梯度方法 - 捕获文本区域通常有强梯度
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                gradient_mask = gradient_magnitude > np.percentile(gradient_magnitude, 80)
                gradient_mask = gradient_mask.astype(np.uint8) * 255
                
                # 3. 二值化方法
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # 合并所有方法的结果
                combined_mask = cv2.bitwise_or(edges, gradient_mask)
                combined_mask = cv2.bitwise_or(combined_mask, binary)
                
                # 形态学操作，连接临近区域
                kernel = np.ones((5,5), np.uint8)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
                
                # 寻找轮廓
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 提取轮廓包围框并合并重叠区域
                regions = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w * h > 200:  # 稍微增大阈值，过滤掉小区域
                        regions.append((x, y, x+w, y+h))
                
                # 合并重叠或接近的边界框
                merged_regions = merge_overlapping_boxes(regions)
                batch_regions.append(merged_regions)
            
            return batch_regions
    
    return []

def create_weight_mask(batch_size, height, width, text_regions, device, focus_factor=2.0):
    """创建基于文本区域的权重掩码，可调整关注因子"""
    weight_mask = torch.ones(batch_size, height, width, device=device)
    
    for b, regions in enumerate(text_regions):
        for x1, y1, x2, y2 in regions:
            # 确保坐标在有效范围内
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            
            if x2 > x1 and y2 > y1:
                weight_mask[b, y1:y2, x1:x2] = focus_factor  # 使用可配置的关注因子
    
    return weight_mask

# ==========================
# 可视化函数
# ==========================
def visualize_epoch_results(model, test_loader, device, epoch, save_dir):
    """可视化每个epoch的检测结果"""
    model.eval()
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        # 随机选择一些图像进行可视化
        samples_to_viz = min(5, len(test_loader.dataset))
        indices = np.random.choice(len(test_loader.dataset), samples_to_viz, replace=False)
        
        for i, idx in enumerate(indices):
            sample = test_loader.dataset[idx]
            image_tensor = sample['image'].unsqueeze(0).to(device)
            
            # 获取检测结果
            outputs = model(image_tensor)
            anomaly_map = outputs['anomaly_map']
            anomaly_score = outputs['anomaly_score'].item()
            
            # 将异常图转回CPU并转为numpy
            anomaly_map = anomaly_map.squeeze().cpu().numpy()
            
            # 获取原始图像
            orig_image = cv2.imread(sample['path'])
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            
            # 创建可视化图
            plt.figure(figsize=(12, 4))
            
            # 原始图像
            plt.subplot(1, 3, 1)
            plt.imshow(orig_image)
            plt.title('原始图像')
            plt.axis('off')
            
            # 上采样异常图到原始大小
            anomaly_map_resized = cv2.resize(anomaly_map, (orig_image.shape[1], orig_image.shape[0]))
            
            # 异常图
            plt.subplot(1, 3, 2)
            plt.imshow(anomaly_map_resized, cmap='jet')
            plt.title(f'异常图 (分数: {anomaly_score:.4f})')
            plt.axis('off')
            
            # 叠加图
            plt.subplot(1, 3, 3)
            plt.imshow(orig_image)
            plt.imshow(anomaly_map_resized, cmap='jet', alpha=0.5)
            plt.title('叠加图')
            plt.axis('off')
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'epoch_{epoch+1}_sample_{i+1}.png'))
            plt.close()

# ==========================
# 训练和评估函数
# ==========================
class EarlyStopping:
    """提前停止训练机制"""
    def __init__(self, patience=20, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def custom_loss_function(outputs, alpha=0.8, beta=0.2):
    """组合多种损失函数"""
    nll_loss = outputs['nll_loss'].mean()  # 原始NLL损失
    
    # 添加特征一致性损失
    consistency_loss = 0
    features = outputs['features']
    if len(features) > 1:
        for i in range(len(features) - 1):
            # 上采样特征以匹配大小
            feat1 = features[i]
            feat2 = nn.functional.interpolate(
                features[i+1], 
                size=feat1.shape[2:],
                mode='bilinear', 
                align_corners=False
            )
            consistency_loss += torch.mean(torch.abs(feat1 - feat2))
        consistency_loss /= (len(features) - 1)
    
    # 组合损失
    total_loss = alpha * nll_loss + beta * consistency_loss
    
    return total_loss

def train_epoch(model, train_loader, optimizer, device, use_mixed_precision=False, 
               accumulation_steps=1, region_aware=False, focus_factor=2.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    # 混合精度训练设置
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    for i, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        
        # 区域感知训练
        if region_aware:
            text_regions = detect_text_regions(images)
            weight_mask = create_weight_mask(
                images.size(0), 
                images.size(2), 
                images.size(3), 
                text_regions, 
                device,
                focus_factor=focus_factor
            )
        
        # 使用混合精度训练
        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = custom_loss_function(outputs)
                
                # 对区域感知进行加权
                if region_aware:
                    # 上采样异常图到原始大小
                    anomaly_map = nn.functional.interpolate(
                        outputs['anomaly_map'],
                        size=(images.size(2), images.size(3)),
                        mode='bilinear',
                        align_corners=False
                    )
                    anomaly_map = anomaly_map.squeeze(1)  # (B, H, W)
                    
                    # 应用权重掩码
                    weighted_map = anomaly_map * weight_mask
                    region_loss = weighted_map.mean()
                    loss = loss + region_loss
                
                # 梯度累积
                loss = loss / accumulation_steps
            
            # 缩放反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积
            if (i + 1) % accumulation_steps == 0:
                # 梯度裁剪避免梯度爆炸
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # 常规训练
            outputs = model(images)
            loss = custom_loss_function(outputs)
            
            # 对区域感知进行加权
            if region_aware:
                # 上采样异常图到原始大小
                anomaly_map = nn.functional.interpolate(
                    outputs['anomaly_map'],
                    size=(images.size(2), images.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
                anomaly_map = anomaly_map.squeeze(1)  # (B, H, W)
                
                # 应用权重掩码
                weighted_map = anomaly_map * weight_mask
                region_loss = weighted_map.mean()
                loss = loss + region_loss
            
            # 梯度累积
            loss = loss / accumulation_steps
            loss.backward()
            
            # 梯度累积
            if (i + 1) % accumulation_steps == 0:
                # 梯度裁剪避免梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    return total_loss / len(train_loader)

def evaluate(model, test_loader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    
    # 用于评估指标
    all_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            
            outputs = model(images)
            loss = custom_loss_function(outputs)
            
            total_loss += loss.item()
            
            # 收集异常分数
            scores = outputs['anomaly_score'].cpu().numpy()
            all_scores.extend(scores)
    
    # 计算平均损失
    avg_loss = total_loss / len(test_loader)
    
    # 分析分数分布
    score_mean = np.mean(all_scores)
    score_std = np.std(all_scores)
    score_min = np.min(all_scores)
    score_max = np.max(all_scores)
    
    eval_results = {
        'loss': avg_loss,
        'score_mean': score_mean,
        'score_std': score_std,
        'score_min': score_min,
        'score_max': score_max
    }
    
    return eval_results

# ==========================
# 检查点保存和加载
# ==========================
def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, save_path):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_loss': best_loss,
    }, save_path)
    print(f"检查点已保存到 {save_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        print(f"检查点 {checkpoint_path} 不存在，从头开始训练")
        return 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"从轮次 {checkpoint['epoch']} 恢复训练，最佳损失: {checkpoint['best_loss']:.6f}")
    return checkpoint['epoch'], checkpoint['best_loss']

# ==========================
# 配置保存函数
# ==========================
def save_model_config(args, save_dir):
    """保存模型配置"""
    config = {
        'input_size': 448,
        'backbone': 'wide_resnet50_2',
        'flow_steps': args.flow_steps,
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'weights': os.path.join(save_dir, 'best_model.pth'),
        'device': args.device
    }
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(save_dir, 'model_config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    print(f"模型配置已保存到 {os.path.join(save_dir, 'model_config.yaml')}")

# ==========================
# 主训练函数
# ==========================
def train(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建保存目录
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(args)
    
    # 创建模型
    model = FastFlow(
        backbone="wide_resnet50_2",
        flow_steps=args.flow_steps,
        input_size=448,
        hidden_ratio=1.0,
        conv3x3_only=True,
        use_dropout=True,
        dropout_rate=0.2
    )
    model = model.to(args.device)
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # 创建学习率调度器
    if args.cosine_lr:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,               # 重启周期
            T_mult=2,             # 周期倍增因子
            eta_min=args.lr/100   # 最小学习率
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=True
        )
    
    # 创建早停机制
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, min_delta=1e-5)
    
    # 从检查点恢复训练 (如果有)
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, args.resume)
        start_epoch += 1  # 从下一个epoch开始
    
    # 保存模型配置
    save_model_config(args, save_dir)
    
    # 训练循环
    print(f"开始训练，设备: {args.device}")
    for epoch in range(start_epoch, args.epochs):
        # 训练一个epoch
        train_loss = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            args.device,
            use_mixed_precision=args.mixed_precision,
            accumulation_steps=args.accumulation_steps,
            region_aware=args.region_aware,
            focus_factor=args.focus_factor
        )
        
        # 评估模型
        eval_results = evaluate(model, test_loader, args.device)
        val_loss = eval_results['loss']
        
        # 更新学习率
        if args.cosine_lr:
            scheduler.step()  # 循环学习率只需要step
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.6f}")
        else:
            scheduler.step(val_loss)  # ReduceLROnPlateau需要验证损失
        
        # 打印进度
        print(f"轮次 [{epoch+1}/{args.epochs}] 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}, "
              f"异常分数: {eval_results['score_mean']:.6f}±{eval_results['score_std']:.6f}")
        
        # 定期可视化结果
        if (epoch + 1) % args.viz_freq == 0 or epoch == 0 or epoch == args.epochs - 1:
            visualize_epoch_results(model, test_loader, args.device, epoch, args.save_dir)
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, 
                optimizer, 
                scheduler, 
                epoch, 
                best_loss, 
                os.path.join(save_dir, 'best_model.pth')
            )
        
        # 保存最近的检查点
        save_checkpoint(
            model, 
            optimizer, 
            scheduler, 
            epoch, 
            best_loss, 
            os.path.join(save_dir, 'last_checkpoint.pth')
        )
        
        # 检查早停
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"触发早停机制，在轮次 {epoch+1} 停止训练")
            break
    
    print(f"训练完成，最佳验证损失: {best_loss:.6f}")
    return os.path.join(save_dir, 'best_model.pth')

# ==========================
# 主函数
# ==========================
if __name__ == "__main__":
    args = parse_args()
    best_model_path = train(args)
    print(f"最佳模型已保存到: {best_model_path}")

# 添加径向模糊函数
def add_radial_blur(image, center_x_ratio=0.5, center_y_ratio=0.5, blur_strength=5):
    """添加径向模糊，中心清晰，边缘模糊"""
    h, w = image.shape[:2]
    center_x = int(w * center_x_ratio)
    center_y = int(h * center_y_ratio)
    
    # 创建距离矩阵
    y, x = np.indices((h, w))
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # 归一化距离
    max_distance = np.sqrt(w**2 + h**2) / 2
    normalized_distance = distance / max_distance
    
    # 创建模糊核大小矩阵 (边缘模糊核更大)
    blur_map = np.clip(normalized_distance * blur_strength, 0, 10).astype(np.int32)
    blur_map = blur_map * 2 + 1  # 确保是奇数
    
    # 应用可变模糊
    result = np.copy(image)
    for y in range(h):
        for x in range(w):
            if blur_map[y, x] > 1:
                kernel_size = blur_map[y, x]
                roi = image[max(0, y-kernel_size//2):min(h, y+kernel_size//2+1),
                           max(0, x-kernel_size//2):min(w, x+kernel_size//2+1)]
                if roi.size > 0:
                    result[y, x] = np.mean(roi, axis=(0, 1))
    
    return result 