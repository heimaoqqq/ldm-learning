#!/usr/bin/env python3
"""
VQ-VAE模型评估和可视化脚本
用于分析训练好的模型性能
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vqvae_model import create_vqvae_model
from dataset import create_dataloaders
from omegaconf import OmegaConf

class VQVAEEvaluator:
    """VQ-VAE模型评估器"""
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        初始化评估器
        
        Args:
            model_path: 模型检查点路径
            config_path: 配置文件路径
        """
        self.model_path = model_path
        self.config_path = config_path or "configs/vqvae_256_config.yaml"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载配置和模型
        self.config = OmegaConf.load(self.config_path)
        self.model = self._load_model()
        self.model.eval()
        
        print(f"模型加载成功: {model_path}")
        print(f"使用设备: {self.device}")
    
    def _load_model(self):
        """加载训练好的模型"""
        # 创建模型
        model = create_vqvae_model(**self.config.model.params)
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model.to(self.device)
    
    @torch.no_grad()
    def evaluate_reconstruction(self, dataloader, num_samples: int = 100):
        """评估重构质量"""
        print("=== 评估重构质量 ===")
        
        reconstruction_losses = []
        perceptual_losses = []
        
        count = 0
        for batch in tqdm(dataloader):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            images = images.to(self.device)
            
            # 前向传播
            reconstructed, quant_loss = self.model(images)
            
            # 计算重构损失 (MSE)
            mse_loss = torch.nn.functional.mse_loss(reconstructed, images, reduction='none')
            mse_loss = mse_loss.view(mse_loss.size(0), -1).mean(dim=1)
            reconstruction_losses.extend(mse_loss.cpu().numpy())
            
            count += images.size(0)
            if count >= num_samples:
                break
        
        reconstruction_losses = np.array(reconstruction_losses)
        
        print(f"重构损失统计:")
        print(f"  平均值: {reconstruction_losses.mean():.6f}")
        print(f"  标准差: {reconstruction_losses.std():.6f}")
        print(f"  最小值: {reconstruction_losses.min():.6f}")
        print(f"  最大值: {reconstruction_losses.max():.6f}")
        
        return reconstruction_losses
    
    @torch.no_grad()
    def analyze_codebook_usage(self, dataloader, num_samples: int = 1000):
        """分析码本使用情况"""
        print("=== 分析码本使用情况 ===")
        
        all_indices = []
        count = 0
        
        for batch in tqdm(dataloader):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
                
            images = images.to(self.device)
            
            # 编码获取索引
            _, _, indices = self.model.encode(images)
            all_indices.extend(indices.flatten().cpu().numpy())
            
            count += images.size(0)
            if count >= num_samples:
                break
        
        all_indices = np.array(all_indices)
        unique_indices, counts = np.unique(all_indices, return_counts=True)
        
        # 计算使用率
        total_codebook_size = self.config.model.params.n_embed
        usage_rate = len(unique_indices) / total_codebook_size
        
        print(f"码本使用统计:")
        print(f"  码本大小: {total_codebook_size}")
        print(f"  使用的码: {len(unique_indices)}")
        print(f"  使用率: {usage_rate:.3f} ({usage_rate*100:.1f}%)")
        print(f"  平均使用次数: {counts.mean():.2f}")
        print(f"  最大使用次数: {counts.max()}")
        
        return unique_indices, counts, usage_rate
    
    @torch.no_grad()
    def visualize_reconstructions(self, dataloader, num_images: int = 8, save_path: str = "results"):
        """可视化重构结果"""
        print("=== 生成重构可视化 ===")
        
        os.makedirs(save_path, exist_ok=True)
        
        # 获取一批图像
        batch = next(iter(dataloader))
        if isinstance(batch, (list, tuple)):
            images = batch[0][:num_images]
        else:
            images = batch[:num_images]
        
        images = images.to(self.device)
        
        # 重构图像
        reconstructed, _ = self.model(images)
        
        # 转换为可显示格式
        def denormalize(tensor):
            return (tensor + 1) / 2
        
        original = denormalize(images).cpu()
        recon = denormalize(reconstructed).cpu()
        
        # 创建对比图
        fig, axes = plt.subplots(2, num_images, figsize=(2*num_images, 4))
        
        for i in range(num_images):
            # 原图
            axes[0, i].imshow(original[i].permute(1, 2, 0).numpy())
            axes[0, i].set_title(f"Original {i+1}")
            axes[0, i].axis('off')
            
            # 重构图
            axes[1, i].imshow(recon[i].permute(1, 2, 0).numpy())
            axes[1, i].set_title(f"Reconstructed {i+1}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/reconstruction_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"重构对比图保存至: {save_path}/reconstruction_comparison.png")
    
    @torch.no_grad()
    def analyze_latent_space(self, dataloader, num_samples: int = 500, save_path: str = "results"):
        """分析潜在空间"""
        print("=== 分析潜在空间 ===")
        
        os.makedirs(save_path, exist_ok=True)
        
        # 收集潜在表示和码本索引
        latent_codes = []
        indices = []
        count = 0
        
        for batch in tqdm(dataloader):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
                
            images = images.to(self.device)
            
            # 编码
            quant, _, ind = self.model.encode(images)
            
            # 收集数据
            latent_codes.append(quant.cpu())
            indices.append(ind.cpu())
            
            count += images.size(0)
            if count >= num_samples:
                break
        
        # 合并数据
        latent_codes = torch.cat(latent_codes, dim=0)
        indices = torch.cat(indices, dim=0)
        
        # 展平特征
        B, C, H, W = latent_codes.shape
        latent_flat = latent_codes.view(B, -1).numpy()
        indices_flat = indices.view(B, -1).numpy()
        
        # 对每个样本取平均码本索引
        sample_indices = np.mean(indices_flat, axis=1).astype(int)
        
        # PCA降维可视化
        print("执行PCA分析...")
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_flat)
        
        # 绘制PCA结果
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=sample_indices, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, label='Average Codebook Index')
        plt.title('VQ-VAE Latent Space (PCA)')
        plt.xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.3f})')
        plt.savefig(f"{save_path}/latent_space_pca.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # t-SNE可视化 (如果样本数不太多)
        if len(latent_flat) <= 1000:
            print("执行t-SNE分析...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_flat)//4))
            latent_tsne = tsne.fit_transform(latent_flat)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=sample_indices, cmap='tab10', alpha=0.6)
            plt.colorbar(scatter, label='Average Codebook Index')
            plt.title('VQ-VAE Latent Space (t-SNE)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.savefig(f"{save_path}/latent_space_tsne.png", dpi=150, bbox_inches='tight')
            plt.show()
        
        print(f"潜在空间分析结果保存至: {save_path}/")
    
    def generate_codebook_visualization(self, save_path: str = "results"):
        """可视化码本向量"""
        print("=== 生成码本可视化 ===")
        
        os.makedirs(save_path, exist_ok=True)
        
        # 获取码本
        codebook = self.model.quantize.embed.weight.data.cpu().numpy()  # [n_embed, embed_dim]
        
        # PCA降维到2维
        pca = PCA(n_components=2)
        codebook_2d = pca.fit_transform(codebook)
        
        # 绘制码本分布
        plt.figure(figsize=(12, 8))
        plt.scatter(codebook_2d[:, 0], codebook_2d[:, 1], alpha=0.6, s=20)
        plt.title('VQ-VAE Codebook Distribution (PCA)')
        plt.xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.3f})')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_path}/codebook_distribution.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"码本可视化保存至: {save_path}/codebook_distribution.png")
    
    def run_full_evaluation(self, data_path: str = "data", save_path: str = "results"):
        """运行完整评估"""
        print("开始VQ-VAE模型完整评估")
        print("=" * 50)
        
        # 创建数据加载器
        print("加载数据...")
        _, val_loader, test_loader = create_dataloaders(
            data_path=data_path,
            batch_size=16,
            num_workers=2,
            image_size=256
        )
        
        # 创建结果目录
        os.makedirs(save_path, exist_ok=True)
        
        # 1. 重构质量评估
        recon_losses = self.evaluate_reconstruction(test_loader)
        
        # 2. 码本使用分析
        unique_indices, counts, usage_rate = self.analyze_codebook_usage(test_loader)
        
        # 3. 可视化重构结果
        self.visualize_reconstructions(test_loader, save_path=save_path)
        
        # 4. 潜在空间分析
        self.analyze_latent_space(test_loader, save_path=save_path)
        
        # 5. 码本可视化
        self.generate_codebook_visualization(save_path=save_path)
        
        # 生成评估报告
        self._generate_report(recon_losses, usage_rate, save_path)
        
        print("=" * 50)
        print("✅ 评估完成!")
        print(f"结果保存在: {save_path}/")
    
    def _generate_report(self, recon_losses, usage_rate, save_path: str):
        """生成评估报告"""
        report_path = os.path.join(save_path, "evaluation_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("VQ-VAE模型评估报告\n")
            f.write("=" * 30 + "\n\n")
            
            f.write(f"模型路径: {self.model_path}\n")
            f.write(f"配置文件: {self.config_path}\n\n")
            
            f.write("模型参数:\n")
            f.write(f"  码本大小: {self.config.model.params.n_embed}\n")
            f.write(f"  嵌入维度: {self.config.model.params.embed_dim}\n")
            f.write(f"  输入分辨率: {self.config.model.params.resolution}\n\n")
            
            f.write("性能指标:\n")
            f.write(f"  平均重构损失: {recon_losses.mean():.6f}\n")
            f.write(f"  重构损失标准差: {recon_losses.std():.6f}\n")
            f.write(f"  码本使用率: {usage_rate:.3f} ({usage_rate*100:.1f}%)\n\n")
            
            f.write("文件说明:\n")
            f.write("  - reconstruction_comparison.png: 重构对比图\n")
            f.write("  - latent_space_pca.png: 潜在空间PCA可视化\n")
            f.write("  - latent_space_tsne.png: 潜在空间t-SNE可视化\n")
            f.write("  - codebook_distribution.png: 码本分布图\n")
        
        print(f"评估报告保存至: {report_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VQ-VAE模型评估")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型检查点路径"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vqvae_256_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data",
        help="数据集路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="结果保存路径"
    )
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        return
    
    # 创建评估器并运行
    try:
        evaluator = VQVAEEvaluator(args.model, args.config)
        evaluator.run_full_evaluation(args.data, args.output)
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        raise

if __name__ == "__main__":
    main() 
