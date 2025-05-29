"""
集成的LDM训练脚本
支持归一化VQ-VAE，适配32×32×256潜在空间
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import sys
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 添加VAE目录到路径以导入归一化模块
sys.path.append('../VAE')
from normalized_vqvae import NormalizedVQVAE
from adv_vq_vae import AdvVQVAE

# 导入现有的CLDM模型
from cldm_paper_standard import PaperStandardCLDM

class NormalizedVQVAEWrapper:
    """
    归一化VQ-VAE包装器
    集成到LDM训练中使用
    """
    
    def __init__(self, vqvae_config, normalization_method='standardize'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载原始VQ-VAE
        self.vqvae = AdvVQVAE(
            in_channels=vqvae_config['in_channels'],
            latent_dim=vqvae_config['latent_dim'],
            num_embeddings=vqvae_config['num_embeddings'],
            beta=vqvae_config['beta'],
            decay=vqvae_config.get('decay', 0.99),
            groups=vqvae_config['groups'],
            disc_ndf=64
        ).to(self.device)
        
        # 加载预训练权重
        model_path = vqvae_config['model_path']
        if os.path.exists(model_path):
            self.vqvae.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ 已加载VQ-VAE模型: {model_path}")
        else:
            print(f"❌ 未找到VQ-VAE模型: {model_path}")
            
        # 创建归一化包装器
        self.norm_vqvae = NormalizedVQVAE(
            self.vqvae, 
            normalization_method=normalization_method
        ).to(self.device)
        
        # 尝试加载归一化统计量
        norm_model_path = f"../VAE/normalized_vqvae_{normalization_method}.pth"
        if os.path.exists(norm_model_path):
            self.norm_vqvae.load_state_dict(torch.load(norm_model_path, map_location=self.device))
            print(f"✅ 已加载归一化统计量: {norm_model_path}")
        else:
            print(f"⚠️ 未找到预计算的归一化统计量，将使用示例数据计算")
            self._fit_normalization_stats()
    
    def _fit_normalization_stats(self):
        """使用示例数据拟合归一化统计量"""
        print("🔍 使用示例数据计算归一化统计量...")
        
        # 创建示例数据
        sample_batches = []
        for _ in range(20):  # 320个样本
            batch = torch.randn(16, 3, 256, 256)
            batch = (batch - batch.min()) / (batch.max() - batch.min())
            sample_batches.append(batch)
        
        # 计算归一化统计量
        all_encodings = []
        self.vqvae.eval()
        
        with torch.no_grad():
            for batch in sample_batches:
                batch = batch.to(self.device)
                z = self.vqvae.encoder(batch)
                all_encodings.append(z.cpu())
        
        all_encodings = torch.cat(all_encodings, dim=0)
        
        if self.norm_vqvae.normalization_method == 'standardize':
            self.norm_vqvae.mean = all_encodings.mean()
            self.norm_vqvae.std = all_encodings.std()
            print(f"📊 标准化统计量: 均值={self.norm_vqvae.mean:.4f}, 标准差={self.norm_vqvae.std:.4f}")
        elif self.norm_vqvae.normalization_method == 'minmax':
            self.norm_vqvae.min_val = all_encodings.min()
            self.norm_vqvae.max_val = all_encodings.max()
            print(f"📊 Min-Max统计量: 范围=[{self.norm_vqvae.min_val:.4f}, {self.norm_vqvae.max_val:.4f}]")
        
        self.norm_vqvae.is_fitted = torch.tensor(True)
        print("✅ 归一化统计量计算完成！")
    
    def encode_for_ldm(self, images):
        """编码图像用于LDM训练"""
        with torch.no_grad():
            return self.norm_vqvae.encode_without_vq(images)
    
    def decode_from_ldm(self, z_normalized):
        """从LDM生成的编码解码回图像"""
        with torch.no_grad():
            return self.norm_vqvae.decode_from_normalized(z_normalized)

class LDMTrainer:
    """
    LDM训练器，支持归一化VQ-VAE
    """
    
    def __init__(self, config_path):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化归一化VQ-VAE
        self.vqvae_wrapper = NormalizedVQVAEWrapper(
            self.config['vqvae'],
            self.config.get('normalization_method', 'standardize')
        )
        
        # 创建CLDM模型 - 适配32×32×256潜在空间
        cldm_config = self.config['cldm']
        self.model = PaperStandardCLDM(
            latent_dim=cldm_config['latent_dim'],
            num_classes=cldm_config['num_classes'],
            num_timesteps=cldm_config['num_timesteps'],
            beta_schedule=cldm_config['beta_schedule'],
            # U-Net参数 - 针对32×32调整
            model_channels=cldm_config['model_channels'],
            time_emb_dim=cldm_config['time_emb_dim'],
            class_emb_dim=cldm_config['class_emb_dim'],
            channel_mult=cldm_config['channel_mult'],
            num_res_blocks=cldm_config['num_res_blocks'],
            attention_resolutions=cldm_config['attention_resolutions'],  # 适配32×32
            dropout=cldm_config['dropout']
        ).to(self.device)
        
        # 打印模型参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 模型参数统计:")
        print(f"  总参数数: {total_params:,}")
        print(f"  可训练参数数: {trainable_params:,}")
        
        # 设置优化器
        train_config = self.config['training']
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_config['lr'],
            weight_decay=train_config.get('weight_decay', 0.0),
            betas=(train_config.get('beta1', 0.9), train_config.get('beta2', 0.999)),
            eps=train_config.get('eps', 1e-8)
        )
        
        # 学习率调度器
        if train_config.get('use_scheduler', False):
            if train_config.get('scheduler_type') == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=train_config['epochs'],
                    eta_min=train_config.get('min_lr', 0.00001)
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None
        
        # 训练状态
        self.epoch = 0
        self.best_loss = float('inf')
        
        # 创建保存目录
        self.save_dir = train_config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        with tqdm(dataloader, desc=f"Epoch {self.epoch}") as pbar:
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 使用归一化VQ-VAE编码图像
                with torch.no_grad():
                    latents = self.vqvae_wrapper.encode_for_ldm(images)
                
                # 检查编码范围
                if batch_idx == 0:
                    print(f"🔍 潜在编码统计: 形状={latents.shape}, "
                          f"范围=[{latents.min():.4f}, {latents.max():.4f}], "
                          f"均值={latents.mean():.4f}, 标准差={latents.std():.4f}")
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(latents, labels)
                
                # 处理CLDM返回的三元组 (loss, pred_noise, noise)
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    loss, pred_noise, noise = outputs
                else:
                    raise ValueError(f"模型返回格式错误: {type(outputs)}")
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                grad_clip = self.config['training'].get('grad_clip_norm')
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 使用归一化VQ-VAE编码图像
                latents = self.vqvae_wrapper.encode_for_ldm(images)
                
                # 前向传播
                outputs = self.model(latents, labels)
                
                # 处理CLDM返回的三元组 (loss, pred_noise, noise)
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    loss, pred_noise, noise = outputs
                else:
                    raise ValueError(f"模型返回格式错误: {type(outputs)}")
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def sample_and_save(self, num_samples=16):
        """生成样本并保存"""
        self.model.eval()
        
        with torch.no_grad():
            # 随机生成类别标签
            class_labels = torch.randint(
                0, self.config['cldm']['num_classes'], 
                (num_samples,), device=self.device
            )
            
            # 生成潜在编码
            shape = (num_samples, 256, 32, 32)  # 32×32×256
            generated_latents = self.model.sample(
                class_labels, 
                self.device,
                shape=shape,
                num_inference_steps=self.config['training']['sampling_steps'],
                eta=self.config['training'].get('eta', 0.0)
            )
            
            # 解码为图像
            generated_images = self.vqvae_wrapper.decode_from_ldm(generated_latents)
            
            # 保存图像
            import torchvision.utils as vutils
            save_path = os.path.join(self.save_dir, f"samples_epoch_{self.epoch}.png")
            vutils.save_image(
                generated_images, save_path,
                nrow=4, normalize=True, value_range=(0, 1)
            )
            
            print(f"✅ 已保存生成样本: {save_path}")
    
    def save_model(self, is_best=False):
        """保存模型"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存检查点
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{self.epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"✅ 已保存最佳模型: {best_path}")
    
    def train(self, train_loader, val_loader):
        """完整训练循环"""
        print("🚀 开始训练LDM...")
        print(f"📊 配置信息:")
        print(f"  潜在空间维度: 32×32×256")
        print(f"  模型通道数: {self.config['cldm']['model_channels']}")
        print(f"  注意力分辨率: {self.config['cldm']['attention_resolutions']}")
        print(f"  归一化方法: {self.config.get('normalization_method', 'standardize')}")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # 验证
            if epoch % self.config['training']['val_interval'] == 0:
                val_loss = self.validate(val_loader)
                val_losses.append(val_loss)
                
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                
                # 保存最佳模型
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_model(is_best=True)
            
            # 生成样本
            if epoch % self.config['training']['sample_interval'] == 0:
                self.sample_and_save()
            
            # 保存检查点
            if epoch % self.config['training']['save_interval'] == 0:
                self.save_model()
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
        
        print("🎉 训练完成！")
        return train_losses, val_losses

def main():
    """主函数"""
    
    # 检查配置文件
    config_path = "config_normalized_ldm.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        print("请先创建配置文件或使用现有的config_paper_standard.yaml")
        return
    
    # 创建训练器
    trainer = LDMTrainer(config_path)
    
    # 这里需要根据你的数据集创建数据加载器
    # 示例代码（需要根据实际情况调整）
    """
    from your_dataset import YourDataset
    
    train_dataset = YourDataset(...)
    val_dataset = YourDataset(...)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=trainer.config['dataset']['batch_size'],
        shuffle=True,
        num_workers=trainer.config['dataset']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=trainer.config['dataset']['batch_size'],
        shuffle=False,
        num_workers=trainer.config['dataset']['num_workers']
    )
    
    # 开始训练
    trainer.train(train_loader, val_loader)
    """
    
    print("✅ 训练器初始化完成！")
    print("请根据你的数据集情况修改main()函数中的数据加载部分。")

if __name__ == "__main__":
    main() 
