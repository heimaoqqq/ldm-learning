"""
LDM训练指标计算模块
包含IS分数、噪声预测精度等评估指标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from scipy.stats import entropy
from typing import Tuple, List, Optional
import math

class InceptionScoreCalculator:
    """
    Inception Score计算器
    评估生成图像的质量和多样性
    """
    def __init__(self, device: torch.device, batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        
        # 加载预训练的Inception v3模型
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False)
        self.inception_model.eval()
        self.inception_model.to(device)
        
        # 预处理：Inception期望[0,1]范围的299x299图像
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def calculate_inception_score(self, images: torch.Tensor, splits: int = 10) -> Tuple[float, float]:
        """
        计算Inception Score
        
        Args:
            images: [N, 3, H, W] 生成的图像，范围[0,1]
            splits: 分割数量用于计算统计量
            
        Returns:
            (mean_score, std_score): IS分数的均值和标准差
        """
        N = images.shape[0]
        
        # 预处理图像
        processed_images = []
        for i in range(0, N, self.batch_size):
            batch = images[i:i+self.batch_size]
            # 确保图像在[0,1]范围内
            batch = torch.clamp(batch, 0, 1)
            
            # 调整尺寸并归一化
            batch_processed = []
            for img in batch:
                img_processed = self.preprocess(img)
                batch_processed.append(img_processed)
            
            batch_processed = torch.stack(batch_processed)
            processed_images.append(batch_processed)
        
        processed_images = torch.cat(processed_images, dim=0)
        
        # 获取Inception预测
        predictions = []
        with torch.no_grad():
            for i in range(0, N, self.batch_size):
                batch = processed_images[i:i+self.batch_size].to(self.device)
                pred = self.inception_model(batch)
                if isinstance(pred, tuple):  # Inception v3在训练模式下返回tuple
                    pred = pred[0]
                pred = F.softmax(pred, dim=1)
                predictions.append(pred.cpu())
        
        predictions = torch.cat(predictions, dim=0).numpy()
        
        # 计算IS分数
        scores = []
        for i in range(splits):
            part = predictions[i * (N // splits): (i + 1) * (N // splits), :]
            p_y = np.mean(part, axis=0)
            
            # 计算KL散度
            kl_scores = []
            for j in range(part.shape[0]):
                kl = entropy(part[j, :], p_y)
                kl_scores.append(kl)
            
            scores.append(np.exp(np.mean(kl_scores)))
        
        return float(np.mean(scores)), float(np.std(scores))

class NoiseMetrics:
    """
    噪声预测精度计算器
    """
    @staticmethod
    def calculate_noise_metrics(predicted_noise: torch.Tensor, 
                              target_noise: torch.Tensor) -> dict:
        """
        计算噪声预测的各种精度指标
        
        Args:
            predicted_noise: [B, C, H, W] 预测的噪声
            target_noise: [B, C, H, W] 真实噪声
            
        Returns:
            包含各种指标的字典
        """
        with torch.no_grad():
            # MSE
            mse = F.mse_loss(predicted_noise, target_noise).item()
            
            # MAE  
            mae = F.l1_loss(predicted_noise, target_noise).item()
            
            # PSNR (Peak Signal-to-Noise Ratio)
            psnr = 20 * math.log10(2.0) - 10 * math.log10(mse) if mse > 0 else float('inf')
            
            # 余弦相似度
            pred_flat = predicted_noise.flatten(1)
            target_flat = target_noise.flatten(1)
            cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean().item()
            
            # 相关系数 (皮尔逊相关系数) - 改进版本
            correlation_list = []
            
            for i in range(pred_flat.shape[0]):
                pred_vec = pred_flat[i].cpu().numpy().flatten()
                target_vec = target_flat[i].cpu().numpy().flatten()
                
                # 使用numpy计算皮尔逊相关系数
                try:
                    # 检查标准差是否足够大
                    if np.std(pred_vec) > 1e-8 and np.std(target_vec) > 1e-8:
                        # 计算相关系数矩阵
                        corr_matrix = np.corrcoef(pred_vec, target_vec)
                        correlation_sample = corr_matrix[0, 1]
                        
                        # 检查结果有效性
                        if np.isfinite(correlation_sample):
                            correlation_list.append(float(correlation_sample))
                        else:
                            correlation_list.append(0.0)
                    else:
                        correlation_list.append(0.0)
                except Exception:
                    correlation_list.append(0.0)
            
            # 计算平均值
            correlation = float(np.mean(correlation_list)) if correlation_list else 0.0
            
            return {
                'noise_mse': mse,
                'noise_mae': mae,
                'noise_psnr': psnr,
                'noise_cosine_similarity': cosine_sim,
                'noise_correlation': correlation
            }

class DiffusionMetrics:
    """
    扩散模型综合指标计算器
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.inception_calculator = InceptionScoreCalculator(device)
        
    def calculate_all_metrics(self, 
                            predicted_noise: torch.Tensor,
                            target_noise: torch.Tensor,
                            generated_images: Optional[torch.Tensor] = None) -> dict:
        """
        计算所有可用的指标
        
        Args:
            predicted_noise: 预测噪声
            target_noise: 真实噪声  
            generated_images: 生成的图像（用于IS计算）
            
        Returns:
            所有指标的字典
        """
        metrics = {}
        
        # 噪声预测指标
        noise_metrics = NoiseMetrics.calculate_noise_metrics(predicted_noise, target_noise)
        metrics.update(noise_metrics)
        
        # IS分数（如果提供了生成图像）
        if generated_images is not None:
            try:
                is_mean, is_std = self.inception_calculator.calculate_inception_score(generated_images)
                metrics['inception_score_mean'] = is_mean
                metrics['inception_score_std'] = is_std
            except Exception as e:
                print(f"警告: IS分数计算失败: {e}")
                metrics['inception_score_mean'] = None
                metrics['inception_score_std'] = None
        
        return metrics

def denormalize_for_metrics(x: torch.Tensor) -> torch.Tensor:
    """
    反归一化图像用于指标计算，从[-1,1]转换到[0,1]
    """
    return (x * 0.5 + 0.5).clamp(0, 1) 
