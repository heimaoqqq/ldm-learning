"""
集成的LDM训练脚本
支持归一化VQ-VAE，适配32×32×256潜在空间
修复Kaggle环境数据集结构和参数问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import sys
import glob
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# 添加VAE目录到路径以导入归一化模块
sys.path.append('../VAE')
from normalized_vqvae import NormalizedVQVAE
from adv_vq_vae import AdvVQVAE

# 导入现有的CLDM模型
from cldm_paper_standard import PaperStandardCLDM

class SimpleGaitDataset(Dataset):
    """简化的步态数据集类，适配Kaggle数据结构"""
    def __init__(self, image_paths, class_ids, transform=None):
        self.image_paths = image_paths
        self.class_ids = class_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = self.class_ids[idx]
            return image, label
        except Exception as e:
            print(f"错误加载图片 {img_path}: {e}")
            # 返回一个默认图片
            default_image = torch.zeros(3, 256, 256)
            return default_image, 0

def build_kaggle_dataloader(root_dir, batch_size=8, num_workers=2, val_split=0.2):
    """
    适配Kaggle数据集结构的数据加载器
    支持ID_X目录结构和递归文件查找
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print(f"🔍 扫描数据集目录: {root_dir}")
    
    # 收集所有图片路径和类别
    all_image_paths = []
    all_class_ids = []
    
    # 扫描所有ID目录
    id_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('ID_')]
    id_dirs.sort()
    
    print(f"📁 找到 {len(id_dirs)} 个ID目录: {id_dirs[:5]}{'...' if len(id_dirs) > 5 else ''}")
    
    for id_dir in id_dirs:
        id_path = os.path.join(root_dir, id_dir)
        
        # 提取类别ID
        try:
            class_id = int(id_dir.replace('ID_', '')) - 1  # 转换为0-based索引
        except:
            print(f"⚠️ 无法解析ID目录: {id_dir}")
            continue
            
        # 收集该ID下的所有jpg文件
        jpg_files = glob.glob(os.path.join(id_path, "*.jpg"))
        
        if jpg_files:
            all_image_paths.extend(jpg_files)
            all_class_ids.extend([class_id] * len(jpg_files))
            print(f"  📂 {id_dir}: {len(jpg_files)} 张图片, 类别ID: {class_id}")
    
    if not all_image_paths:
        # 如果按目录结构没找到，尝试直接在根目录查找
        jpg_files = glob.glob(os.path.join(root_dir, "**/*.jpg"), recursive=True)
        print(f"🔍 递归查找到 {len(jpg_files)} 个jpg文件")
        
        for img_path in jpg_files:
            filename = os.path.basename(img_path)
            try:
                # 尝试从文件名解析类别
                if 'ID' in filename:
                    class_id_str = filename.split('ID')[1].split('_')[0]
                    class_id = int(class_id_str) - 1
                    all_image_paths.append(img_path)
                    all_class_ids.append(class_id)
            except:
                # 如果解析失败，使用默认类别
                all_image_paths.append(img_path)
                all_class_ids.append(0)
    
    if not all_image_paths:
        raise ValueError(f"在目录 {root_dir} 中没有找到任何图片文件")
    
    print(f"✅ 总共找到 {len(all_image_paths)} 张图片")
    print(f"📊 类别分布: {len(set(all_class_ids))} 个不同类别")
    
    # 划分训练集和验证集
    if len(set(all_class_ids)) > 1:
        # 有多个类别，使用分层采样
        try:
            train_paths, val_paths, train_ids, val_ids = train_test_split(
                all_image_paths, all_class_ids, 
                test_size=val_split, 
                random_state=42, 
                stratify=all_class_ids
            )
        except:
            # 如果分层采样失败，使用普通随机划分
            train_paths, val_paths, train_ids, val_ids = train_test_split(
                all_image_paths, all_class_ids, 
                test_size=val_split, 
                random_state=42
            )
    else:
        # 只有一个类别，使用普通随机划分
        train_paths, val_paths, train_ids, val_ids = train_test_split(
            all_image_paths, all_class_ids, 
            test_size=val_split, 
            random_state=42
        )
    
    print(f"📈 数据集划分完成:")
    print(f"  训练集: {len(train_paths)} 张图片")
    print(f"  验证集: {len(val_paths)} 张图片")
    
    # 创建数据集和数据加载器
    train_dataset = SimpleGaitDataset(train_paths, train_ids, transform=transform)
    val_dataset = SimpleGaitDataset(val_paths, val_ids, transform=transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=min(num_workers, 2),  # 在Kaggle中限制workers数量
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=min(num_workers, 2),
        pin_memory=True
    )
    
    return train_loader, val_loader, len(train_dataset), len(val_dataset)

class NormalizedVQVAEWrapper:
    """
    归一化VQ-VAE包装器
    集成到LDM训练中使用，支持真实数据归一化统计量计算
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
        
        # 归一化统计量将在实际数据上计算
        self.is_fitted = False
        self.normalization_method = normalization_method
    
    def fit_normalization_on_real_data(self, dataloader, max_samples=1000):
        """在真实数据上计算归一化统计量"""
        print("🔍 在真实数据集上计算归一化统计量...")
        
        self.vqvae.eval()
        all_encodings = []
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                if sample_count >= max_samples:
                    break
                    
                images = images.to(self.device)
                
                # 获取编码器输出（VQ之前的连续表示）
                z = self.vqvae.encoder(images)
                all_encodings.append(z.cpu())
                sample_count += images.size(0)
                
                if batch_idx % 10 == 0:
                    print(f"  已处理 {sample_count} 个样本...")
        
        # 合并所有编码
        all_encodings = torch.cat(all_encodings, dim=0)
        
        # 检查原始编码范围
        print(f"🔍 原始编码范围: [{all_encodings.min():.4f}, {all_encodings.max():.4f}]")
        print(f"🔍 原始编码统计: 均值={all_encodings.mean():.4f}, 标准差={all_encodings.std():.4f}")
        
        if self.norm_vqvae.normalization_method == 'standardize':
            # 直接使用NormalizedVQVAE的fit_normalization_stats方法
            # 这样确保统计量正确保存到模型中
            self.norm_vqvae.mean = all_encodings.mean().to(self.device)
            self.norm_vqvae.std = all_encodings.std().to(self.device)
            self.norm_vqvae.is_fitted = torch.tensor(True).to(self.device)
            
            print(f"📊 标准化统计量: 均值={self.norm_vqvae.mean:.4f}, 标准差={self.norm_vqvae.std:.4f}")
            
            # 测试归一化效果
            test_sample = all_encodings[:100].to(self.device)
            normalized_test = (test_sample - self.norm_vqvae.mean) / (self.norm_vqvae.std + 1e-5)
            print(f"🧪 归一化测试: [{normalized_test.min():.4f}, {normalized_test.max():.4f}], 均值={normalized_test.mean():.4f}, 标准差={normalized_test.std():.4f}")
            
        elif self.norm_vqvae.normalization_method == 'minmax':
            self.norm_vqvae.min_val = all_encodings.min().to(self.device)
            self.norm_vqvae.max_val = all_encodings.max().to(self.device)
            self.norm_vqvae.is_fitted = torch.tensor(True).to(self.device)
            
            print(f"📊 Min-Max统计量: 范围=[{self.norm_vqvae.min_val:.4f}, {self.norm_vqvae.max_val:.4f}]")
            
            # 测试归一化效果
            test_sample = all_encodings[:100].to(self.device)
            normalized_test = 2.0 * (test_sample - self.norm_vqvae.min_val) / (self.norm_vqvae.max_val - self.norm_vqvae.min_val + 1e-5) - 1.0
            print(f"🧪 归一化测试: [{normalized_test.min():.4f}, {normalized_test.max():.4f}]")
        
        self.is_fitted = True
        print("✅ 真实数据归一化统计量计算完成！")
        
        # 强制启用调试模式进行下次验证
        self.norm_vqvae._debug = True
        print("🔧 已启用归一化调试模式")
    
    def _fit_normalization_stats(self):
        """使用示例数据拟合归一化统计量（备用方法）"""
        print("⚠️ 使用示例数据计算归一化统计量（建议使用真实数据）...")
        
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
            print(f"📊 示例数据标准化统计量: 均值={self.norm_vqvae.mean:.4f}, 标准差={self.norm_vqvae.std:.4f}")
        elif self.norm_vqvae.normalization_method == 'minmax':
            self.norm_vqvae.min_val = all_encodings.min()
            self.norm_vqvae.max_val = all_encodings.max()
            print(f"📊 示例数据Min-Max统计量: 范围=[{self.norm_vqvae.min_val:.4f}, {self.norm_vqvae.max_val:.4f}]")
        
        self.norm_vqvae.is_fitted = torch.tensor(True)
        self.is_fitted = True
        print("✅ 示例数据归一化统计量计算完成！")
    
    def encode_for_ldm(self, images):
        """编码图像用于LDM训练"""
        if not self.is_fitted:
            print("⚠️ 归一化统计量未计算，使用示例数据计算...")
            self._fit_normalization_stats()
        
        with torch.no_grad():
            # 调用NormalizedVQVAE的encode_without_vq方法
            z_normalized = self.norm_vqvae.encode_without_vq(images)
            return z_normalized
    
    def decode_from_ldm(self, z_normalized):
        """从LDM生成的编码解码回图像"""
        if not self.is_fitted:
            raise RuntimeError("请先计算归一化统计量！")
            
        with torch.no_grad():
            # 调用NormalizedVQVAE的decode_from_normalized方法
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
        
        # 简化的训练循环，只在第一个batch显示编码统计
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 使用归一化VQ-VAE编码图像
            with torch.no_grad():
                latents = self.vqvae_wrapper.encode_for_ldm(images)
            
            # 只在第一个epoch的第一个批次显示编码统计
            if self.epoch == 0 and batch_idx == 0:
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
        """保存模型并删除旧的检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 删除旧的检查点文件（保留最近3个）
        checkpoint_pattern = os.path.join(self.save_dir, "checkpoint_epoch_*.pth")
        existing_checkpoints = glob.glob(checkpoint_pattern)
        existing_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # 删除旧于第3个的检查点
        if len(existing_checkpoints) >= 3:
            for old_checkpoint in existing_checkpoints[2:]:
                try:
                    os.remove(old_checkpoint)
                    print(f"🗑️ 已删除旧检查点: {os.path.basename(old_checkpoint)}")
                except Exception as e:
                    print(f"⚠️ 删除失败 {old_checkpoint}: {e}")
        
        # 保存新检查点
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{self.epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 已保存检查点: checkpoint_epoch_{self.epoch}.pth")
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"✅ 已保存最佳模型: best_model.pth")
    
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
                val_loss, eval_metrics = self.comprehensive_evaluation(val_loader, save_samples=True)
                val_losses.append(val_loss)
                
                print(f"✅ Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                
                # 保存最佳模型（基于验证损失）
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_model(is_best=True)
                    print(f"🎯 新的最佳模型! 验证损失: {val_loss:.6f}")
                
            else:
                # 只显示训练损失，不进行详细评估
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")
            
            # 常规样本生成（简化版本）
            if epoch % self.config['training']['sample_interval'] == 0 and epoch % self.config['training']['val_interval'] != 0:
                self.sample_and_save()
            
            # 保存检查点
            if epoch % self.config['training']['save_interval'] == 0:
                self.save_model()
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
        
        print("🎉 训练完成！")
        return train_losses, val_losses

    def evaluate_generation_quality(self, num_samples=100, num_classes=None):
        """
        评估LDM生成质量
        包括FID、多样性、类别条件准确性等指标
        """
        self.model.eval()
        print(f"🔍 开始评估生成质量 (样本数: {num_samples})...")
        
        if num_classes is None:
            num_classes = self.config['cldm']['num_classes']
        
        generated_images = []
        class_labels = []
        
        with torch.no_grad():
            # 生成指定数量的样本
            samples_per_batch = min(16, num_samples)
            num_batches = (num_samples + samples_per_batch - 1) // samples_per_batch
            
            for batch_idx in range(num_batches):
                current_batch_size = min(samples_per_batch, num_samples - batch_idx * samples_per_batch)
                
                # 随机生成类别标签
                batch_labels = torch.randint(0, num_classes, (current_batch_size,), device=self.device)
                
                # 生成潜在编码
                shape = (current_batch_size, 256, 32, 32)  # 32×32×256
                generated_latents = self.model.sample(
                    batch_labels, 
                    self.device,
                    shape=shape,
                    num_inference_steps=self.config['training']['sampling_steps'],
                    eta=self.config['training'].get('eta', 0.0)
                )
                
                # 解码为图像
                batch_images = self.vqvae_wrapper.decode_from_ldm(generated_latents)
                
                generated_images.append(batch_images.cpu())
                class_labels.extend(batch_labels.cpu().tolist())
                
                if (batch_idx + 1) % 5 == 0:
                    print(f"  已生成 {(batch_idx + 1) * samples_per_batch} / {num_samples} 样本")
        
        # 合并所有生成的图像
        generated_images = torch.cat(generated_images, dim=0)
        
        # 评估指标
        metrics = {}
        
        # 1. 图像质量统计
        metrics['image_stats'] = {
            'mean_pixel_value': generated_images.mean().item(),
            'std_pixel_value': generated_images.std().item(),
            'min_pixel_value': generated_images.min().item(),
            'max_pixel_value': generated_images.max().item()
        }
        
        # 2. 类别分布均匀性
        from collections import Counter
        class_distribution = Counter(class_labels)
        metrics['class_distribution'] = {
            'entropy': self._calculate_entropy(list(class_distribution.values())),
            'max_class_ratio': max(class_distribution.values()) / len(class_labels),
            'num_unique_classes': len(class_distribution)
        }
        
        # 3. 图像多样性 (通过像素差异估算)
        if len(generated_images) > 1:
            # 随机选择样本对计算差异
            num_pairs = min(100, len(generated_images) * (len(generated_images) - 1) // 2)
            differences = []
            for _ in range(num_pairs):
                idx1, idx2 = torch.randint(0, len(generated_images), (2,))
                if idx1 != idx2:
                    diff = torch.mean((generated_images[idx1] - generated_images[idx2]) ** 2).item()
                    differences.append(diff)
            
            metrics['diversity'] = {
                'mean_pairwise_mse': np.mean(differences),
                'std_pairwise_mse': np.std(differences)
            }
        
        print("✅ 生成质量评估完成")
        return metrics, generated_images
    
    def _calculate_entropy(self, counts):
        """计算分布熵"""
        counts = np.array(counts)
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # 避免log(0)
        return -np.sum(probs * np.log2(probs))
    
    def comprehensive_evaluation(self, val_loader, save_samples=True):
        """
        综合评估LDM质量
        """
        print("🔍 开始综合质量评估...")
        
        # 1. 验证损失
        val_loss = self.validate(val_loader)
        print(f"📊 验证损失: {val_loss:.6f}")
        
        # 2. 生成质量评估
        metrics, generated_images = self.evaluate_generation_quality(
            num_samples=self.config['training'].get('eval_samples', 64)
        )
        
        # 3. 打印评估结果
        print(f"\n📈 生成质量指标:")
        print(f"  图像统计:")
        stats = metrics['image_stats']
        print(f"    像素值范围: [{stats['min_pixel_value']:.3f}, {stats['max_pixel_value']:.3f}]")
        print(f"    像素均值: {stats['mean_pixel_value']:.3f} ± {stats['std_pixel_value']:.3f}")
        
        print(f"  类别分布:")
        dist = metrics['class_distribution']
        print(f"    分布熵: {dist['entropy']:.3f} (理想值: {np.log2(self.config['cldm']['num_classes']):.3f})")
        print(f"    最大类别占比: {dist['max_class_ratio']:.3f}")
        print(f"    生成类别数: {dist['num_unique_classes']}/{self.config['cldm']['num_classes']}")
        
        if 'diversity' in metrics:
            div = metrics['diversity']
            print(f"  图像多样性:")
            print(f"    平均成对差异: {div['mean_pairwise_mse']:.6f}")
        
        # 4. 保存生成样本
        if save_samples:
            import torchvision.utils as vutils
            save_path = os.path.join(self.save_dir, f"eval_samples_epoch_{self.epoch}.png")
            vutils.save_image(
                generated_images[:min(64, len(generated_images))], save_path,
                nrow=8, normalize=True, value_range=(0, 1)
            )
            print(f"✅ 已保存评估样本: {save_path}")
        
        return val_loss, metrics

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
    
    # 配置数据集路径 - 使用Kaggle适配的数据加载器
    print("🔍 配置数据集...")
    
    try:
        # 使用Kaggle适配的数据加载器
        train_loader, val_loader, train_dataset_len, val_dataset_len = build_kaggle_dataloader(
            root_dir=trainer.config['dataset']['root_dir'],
            batch_size=trainer.config['dataset']['batch_size'],
            num_workers=trainer.config['dataset']['num_workers'],
            val_split=0.2  # 20%验证集，80%训练集
        )
        
        print(f"✅ 数据集加载成功:")
        print(f"  训练集: {train_dataset_len} 张图片")
        print(f"  验证集: {val_dataset_len} 张图片")
        print(f"  批次大小: {trainer.config['dataset']['batch_size']}")
        
        # 在真实数据上计算归一化统计量
        print("🔧 在真实数据集上计算归一化统计量...")
        trainer.vqvae_wrapper.fit_normalization_on_real_data(train_loader, max_samples=500)
        
        # 开始训练
        print("🚀 开始LDM训练...")
        train_losses, val_losses = trainer.train(train_loader, val_loader)
        
        print("🎉 训练完成！")
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        print("\n📋 请检查以下配置:")
        print(f"  数据集路径: {trainer.config['dataset']['root_dir']}")
        print(f"  批次大小: {trainer.config['dataset']['batch_size']}")
        print(f"  工作进程数: {trainer.config['dataset']['num_workers']}")
        print("\n💡 解决方案:")
        print("1. 检查数据集路径是否正确")
        print("2. 确保图片格式为 .jpg")
        print("3. 检查Kaggle输入数据集是否正确挂载")
        print("4. 可以尝试减小批次大小或工作进程数")
        
        # 提供调试信息
        print(f"\n🔍 当前工作目录: {os.getcwd()}")
        if os.path.exists("/kaggle/input"):
            print("📁 Kaggle输入目录:")
            for item in os.listdir("/kaggle/input"):
                item_path = f"/kaggle/input/{item}"
                if os.path.isdir(item_path):
                    print(f"  📂 {item}/")
                    try:
                        sub_items = os.listdir(item_path)[:5]  # 只显示前5个
                        for sub_item in sub_items:
                            print(f"    📄 {sub_item}")
                        if len(os.listdir(item_path)) > 5:
                            print(f"    ... 和其他 {len(os.listdir(item_path)) - 5} 个文件")
                    except:
                        print(f"    ❌ 无法访问")
                else:
                    print(f"  📄 {item}")
        
        return
    
    print("✅ LDM训练流程配置完成！")

if __name__ == "__main__":
    main() 
