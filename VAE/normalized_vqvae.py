"""
归一化VQ-VAE模块
解决潜在编码数值范围过大的问题
"""

import torch
import torch.nn as nn
import numpy as np

class NormalizedVQVAE(nn.Module):
    """
    VQ-VAE包装器，自动处理潜在编码的归一化
    解决编码值过大的问题，使其适合LDM训练
    """
    
    def __init__(self, vqvae_model, normalization_method='standardize', 
                 target_range=(-1.0, 1.0), eps=1e-5):
        """
        Args:
            vqvae_model: 训练好的AdvVQVAE模型
            normalization_method: 'standardize' (Z-score) 或 'minmax' (Min-Max scaling)
            target_range: 当使用minmax时的目标范围
            eps: 避免除零的小常数
        """
        super().__init__()
        self.vqvae = vqvae_model
        self.normalization_method = normalization_method
        self.target_range = target_range
        self.eps = eps
        
        # 用于存储归一化统计量
        self.register_buffer('is_fitted', torch.tensor(False))
        self.register_buffer('mean', torch.tensor(0.0))
        self.register_buffer('std', torch.tensor(1.0))
        self.register_buffer('min_val', torch.tensor(0.0))
        self.register_buffer('max_val', torch.tensor(1.0))
        
    def fit_normalization_stats(self, dataloader, max_samples=1000):
        """
        在数据集上计算归一化统计量
        
        Args:
            dataloader: 数据加载器
            max_samples: 最大采样数量
        """
        print("🔍 计算潜在编码的归一化统计量...")
        
        self.vqvae.eval()
        all_encodings = []
        sample_count = 0
        
        with torch.no_grad():
            for images, _ in dataloader:
                if sample_count >= max_samples:
                    break
                    
                images = images.to(next(self.vqvae.parameters()).device)
                
                # 获取编码器输出（VQ之前的连续表示）
                z = self.vqvae.encoder(images)
                all_encodings.append(z.cpu())
                sample_count += images.size(0)
        
        # 合并所有编码
        all_encodings = torch.cat(all_encodings, dim=0)
        
        if self.normalization_method == 'standardize':
            # 计算全局均值和标准差
            self.mean = all_encodings.mean()
            self.std = all_encodings.std()
            print(f"📊 标准化统计量: 均值={self.mean:.4f}, 标准差={self.std:.4f}")
            
        elif self.normalization_method == 'minmax':
            # 计算全局最小值和最大值
            self.min_val = all_encodings.min()
            self.max_val = all_encodings.max()
            print(f"📊 Min-Max统计量: 最小值={self.min_val:.4f}, 最大值={self.max_val:.4f}")
        
        self.is_fitted = torch.tensor(True)
        print("✅ 归一化统计量计算完成！")
    
    def normalize_encoding(self, z):
        """归一化编码"""
        if not self.is_fitted:
            raise RuntimeError("必须先调用fit_normalization_stats()来计算统计量！")
        
        if hasattr(self, '_debug') and self._debug:
            print(f"🔍 归一化前: 输入范围[{z.min():.4f}, {z.max():.4f}], 均值={z.mean():.4f}, 标准差={z.std():.4f}")
            print(f"🔍 统计量: mean={self.mean:.4f}, std={self.std:.4f}")
            
        if self.normalization_method == 'standardize':
            # Z-score标准化
            z_normalized = (z - self.mean) / (self.std + self.eps)
            
        elif self.normalization_method == 'minmax':
            # Min-Max缩放
            # 先缩放到[0, 1]
            z_01 = (z - self.min_val) / (self.max_val - self.min_val + self.eps)
            # 再缩放到目标范围
            target_min, target_max = self.target_range
            z_normalized = z_01 * (target_max - target_min) + target_min
        
        if hasattr(self, '_debug') and self._debug:
            print(f"🔍 归一化后: 输出范围[{z_normalized.min():.4f}, {z_normalized.max():.4f}], 均值={z_normalized.mean():.4f}, 标准差={z_normalized.std():.4f}")
            # 只在第一次显示调试信息后关闭，避免刷屏
            self._debug = False
            
        return z_normalized
    
    def denormalize_encoding(self, z_normalized):
        """反归一化编码（用于解码）"""
        if not self.is_fitted:
            raise RuntimeError("必须先调用fit_normalization_stats()来计算统计量！")
            
        if self.normalization_method == 'standardize':
            # Z-score反标准化
            z = z_normalized * (self.std + self.eps) + self.mean
            
        elif self.normalization_method == 'minmax':
            # Min-Max反缩放
            target_min, target_max = self.target_range
            # 先从目标范围缩放回[0, 1]
            z_01 = (z_normalized - target_min) / (target_max - target_min)
            # 再缩放回原始范围
            z = z_01 * (self.max_val - self.min_val + self.eps) + self.min_val
            
        return z
    
    def encode(self, x):
        """编码并归一化"""
        z = self.vqvae.encoder(x)
        z_normalized = self.normalize_encoding(z)
        z_q, _, _, _ = self.vqvae.vq(z_normalized)
        return z_q
    
    def decode(self, z_q):
        """反归一化并解码"""
        z_denormalized = self.denormalize_encoding(z_q)
        return self.vqvae.decoder(z_denormalized)
    
    def encode_without_vq(self, x):
        """只进行编码和归一化，不经过VQ层（用于LDM训练）"""
        if not self.is_fitted:
            raise RuntimeError("必须先调用fit_normalization_stats()来计算统计量！")
            
        z = self.vqvae.encoder(x)
        z_normalized = self.normalize_encoding(z)
        
        # 添加调试信息（只在需要时）
        if hasattr(self, '_debug') and self._debug:
            print(f"🔍 编码调试: 原始范围[{z.min():.4f}, {z.max():.4f}] -> 归一化后[{z_normalized.min():.4f}, {z_normalized.max():.4f}]")
        
        return z_normalized
    
    def decode_from_normalized(self, z_normalized):
        """从归一化的编码直接解码（用于LDM采样）"""
        if not self.is_fitted:
            raise RuntimeError("必须先调用fit_normalization_stats()来计算统计量！")
            
        z_denormalized = self.denormalize_encoding(z_normalized)
        return self.vqvae.decoder(z_denormalized)
    
    def forward(self, x):
        """完整的前向传播"""
        z = self.vqvae.encoder(x)
        z_normalized = self.normalize_encoding(z)
        z_q, commitment_loss, codebook_loss, usage_info = self.vqvae.vq(z_normalized)
        x_recon = self.decode(z_q)
        return x_recon, commitment_loss, codebook_loss, usage_info
    
    def get_stats_summary(self):
        """获取统计量摘要"""
        if not self.is_fitted:
            return "未拟合归一化统计量"
            
        if self.normalization_method == 'standardize':
            return f"标准化: 均值={self.mean:.4f}, 标准差={self.std:.4f}"
        else:
            return f"Min-Max: 范围=[{self.min_val:.4f}, {self.max_val:.4f}] -> {self.target_range}" 
