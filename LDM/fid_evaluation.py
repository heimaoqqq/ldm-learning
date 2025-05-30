import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, transforms
from scipy import linalg
import sys
import os
from typing import Tuple, Optional, List
from tqdm import tqdm

# 在Kaggle环境中，VAE模块已复制到当前目录
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VAE'))

def denormalize(x):
    """反归一化图像，从[-1,1]转换到[0,1]"""
    return (x * 0.5 + 0.5).clamp(0, 1)

class InceptionV3Features(nn.Module):
    """
    使用InceptionV3提取特征用于FID计算
    """
    def __init__(self, device='cuda'):
        super().__init__()
        
        # 加载预训练的InceptionV3
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        self.inception.eval()
        
        # 冻结参数
        for param in self.inception.parameters():
            param.requires_grad = False
            
        self.device = device
        self.to(device)
        
        # 预处理：InceptionV3期望输入为[0,1]范围，299x299尺寸
        # 注意：不在这里进行resize，因为transforms.Resize期望PIL图像
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
    
    def forward(self, x):
        """
        提取特征
        Args:
            x: [batch_size, 3, H, W] 图像张量，范围[0,1]
        Returns:
            features: [batch_size, 2048] 特征向量
        """
        # 调整图像尺寸到299x299
        if x.size(-1) != 299 or x.size(-2) != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # 应用ImageNet标准化
        x = self.normalize(x)
        
        # 自定义前向传播，跳过辅助分类器
        # N x 3 x 299 x 299
        x = self.inception.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6e(x)
        # N x 768 x 17 x 17
        # 跳过 AuxLogits
        x = self.inception.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.inception.avgpool(x)
        # N x 2048 x 1 x 1
        
        # 扁平化特征
        features = x.view(x.size(0), -1)
        # N x 2048
        
        return features

def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, 
                             mu2: np.ndarray, sigma2: np.ndarray, 
                             eps: float = 1e-6) -> float:
    """
    计算两个多变量高斯分布之间的Fréchet距离
    
    Args:
        mu1, mu2: 均值向量
        sigma1, sigma2: 协方差矩阵
        eps: 数值稳定性参数
    
    Returns:
        FID分数
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "均值向量形状不匹配"
    assert sigma1.shape == sigma2.shape, "协方差矩阵形状不匹配"
    
    diff = mu1 - mu2
    
    # 计算协方差矩阵的平方根
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # 处理数值误差
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; "
               "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # 数值稳定性
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.absolute(covmean.imag))
            raise ValueError(f"虚部过大: {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + np.trace(sigma1) + 
            np.trace(sigma2) - 2 * tr_covmean)

def extract_features_from_dataloader(feature_extractor: InceptionV3Features,
                                   dataloader: DataLoader,
                                   max_samples: Optional[int] = None,
                                   description: str = "提取特征") -> np.ndarray:
    """
    从数据加载器中提取特征
    
    Args:
        feature_extractor: 特征提取器
        dataloader: 数据加载器
        max_samples: 最大样本数
        description: 进度条描述
    
    Returns:
        features: [N, 2048] 特征数组
    """
    features_list = []
    sample_count = 0
    
    feature_extractor.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=description)
        for batch_idx, (images, _) in enumerate(pbar):
            if max_samples and sample_count >= max_samples:
                break
                
            images = images.to(feature_extractor.device)
            
            # 如果输入是[-1,1]范围，需要反归一化到[0,1]
            if images.min() < 0:
                images = denormalize(images)
            
            batch_features = feature_extractor(images)
            features_list.append(batch_features.cpu().numpy())
            
            sample_count += images.size(0)
            pbar.set_postfix({'samples': sample_count})
            
            if max_samples and sample_count >= max_samples:
                break
    
    features = np.concatenate(features_list, axis=0)
    if max_samples:
        features = features[:max_samples]
    
    return features

def extract_features_from_images(feature_extractor: InceptionV3Features,
                                images: torch.Tensor) -> np.ndarray:
    """
    从图像张量中提取特征
    
    Args:
        feature_extractor: 特征提取器
        images: [N, 3, H, W] 图像张量
    
    Returns:
        features: [N, 2048] 特征数组
    """
    feature_extractor.eval()
    with torch.no_grad():
        # 如果输入是[-1,1]范围，需要反归一化到[0,1]
        if images.min() < 0:
            images = denormalize(images)
        
        # 分批处理以节省内存
        batch_size = 32
        features_list = []
        
        for i in range(0, images.size(0), batch_size):
            batch = images[i:i+batch_size].to(feature_extractor.device)
            batch_features = feature_extractor(batch)
            features_list.append(batch_features.cpu().numpy())
        
        features = np.concatenate(features_list, axis=0)
    
    return features

def calculate_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算特征的均值和协方差矩阵
    
    Args:
        features: [N, D] 特征数组
    
    Returns:
        mu: 均值向量
        sigma: 协方差矩阵
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

class FIDEvaluator:
    """
    FID评估器
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.feature_extractor = InceptionV3Features(device)
        self.real_features_cache = None
        self.real_mu = None
        self.real_sigma = None
    
    def compute_real_features(self, real_dataloader: DataLoader, 
                            max_samples: Optional[int] = None,
                            cache: bool = True):
        """
        计算真实数据的特征统计
        
        Args:
            real_dataloader: 真实数据的数据加载器
            max_samples: 最大样本数
            cache: 是否缓存结果
        """
        print("计算真实数据特征...")
        real_features = extract_features_from_dataloader(
            self.feature_extractor, real_dataloader, max_samples, "提取真实数据特征"
        )
        
        self.real_mu, self.real_sigma = calculate_statistics(real_features)
        
        if cache:
            self.real_features_cache = real_features
        
        print(f"真实数据特征计算完成: {real_features.shape[0]} 个样本")
    
    def calculate_fid_score(self, generated_images: torch.Tensor) -> float:
        """
        计算生成图像的FID分数
        
        Args:
            generated_images: [N, 3, H, W] 生成的图像张量
        
        Returns:
            FID分数
        """
        if self.real_mu is None or self.real_sigma is None:
            raise ValueError("请先调用 compute_real_features() 计算真实数据特征")
        
        # 提取生成图像特征
        gen_features = extract_features_from_images(self.feature_extractor, generated_images)
        gen_mu, gen_sigma = calculate_statistics(gen_features)
        
        # 计算FID
        fid_score = calculate_frechet_distance(self.real_mu, self.real_sigma, 
                                             gen_mu, gen_sigma)
        
        return fid_score
    
    def evaluate_model(self, model, num_samples: int = 1000, 
                      batch_size: int = 8, num_classes: int = 31) -> float:
        """
        评估模型的FID分数
        
        Args:
            model: LDM模型
            num_samples: 生成样本数
            batch_size: 批量大小
            num_classes: 类别数
        
        Returns:
            FID分数
        """
        if self.real_mu is None or self.real_sigma is None:
            raise ValueError("请先调用 compute_real_features() 计算真实数据特征")
        
        model.eval()
        generated_images = []
        
        # 生成样本
        with torch.no_grad():
            samples_per_class = num_samples // num_classes
            remaining_samples = num_samples % num_classes
            
            for class_id in range(num_classes):
                # 每个类别生成相应数量的样本
                current_samples = samples_per_class
                if class_id < remaining_samples:
                    current_samples += 1
                
                if current_samples == 0:
                    continue
                
                # 分批生成
                for i in range(0, current_samples, batch_size):
                    current_batch_size = min(batch_size, current_samples - i)
                    class_labels = torch.tensor([class_id] * current_batch_size, 
                                               device=self.device)
                    
                    images = model.sample(
                        batch_size=current_batch_size,
                        class_labels=class_labels,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                    )
                    
                    generated_images.append(images.cpu())
        
        # 合并所有生成的图像
        all_generated = torch.cat(generated_images, dim=0)
        
        # 计算FID
        fid_score = self.calculate_fid_score(all_generated)
        
        return fid_score 
