"""
Train LDM - 基于CompVis/latent-diffusion
使用微调后的AutoencoderKL训练潜在扩散模型
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import json
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# FID评估相关导入
from torchvision import models
from scipy import linalg
from torchvision.transforms.functional import resize, to_tensor

# 导入我们的模块
from ldm_model import create_ldm_model, LatentDiffusionModel
# 修改数据加载器导入
try:
    # 尝试导入VAE的数据加载器
    sys.path.append('../VAE')
    from dataset import build_dataloader as vae_build_dataloader
    print("✅ 成功导入VAE数据加载器")
except ImportError:
    print("⚠️  VAE数据加载器不可用，将使用合成数据")
    vae_build_dataloader = None

# 云服务器环境设置
print("🌐 LDM训练环境初始化...")
print(f"📂 当前工作目录: {os.getcwd()}")

# 检查CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"🚀 CUDA设备: {torch.cuda.get_device_name()}")
    print(f"💾 总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = 'cuda'
else:
    print("⚠️  CUDA不可用，使用CPU")
    device = 'cpu'

# 内存优化
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class FIDEvaluator:
    """FID评估器，用于计算生成图像的FID分数"""
    
    def __init__(self, device='cuda', inception_batch_size=64):
        self.device = device
        self.inception_batch_size = inception_batch_size
        
        # 加载预训练的Inception-v3模型
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()  # 移除最后的分类层
        self.inception.eval()
        self.inception.to(device)
        
        # 图像预处理（用于Inception-v3）
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),  # Inception-v3需要299x299输入
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ FID评估器初始化完成")
    
    def extract_features(self, images):
        """
        提取图像特征
        输入: [N, 3, H, W] 的RGB图像，范围[0, 1]
        """
        features = []
        
        with torch.no_grad():
            for i in range(0, len(images), self.inception_batch_size):
                batch = images[i:i + self.inception_batch_size]
                
                # 确保是3通道RGB图像，范围[0, 1]
                if batch.size(1) != 3:
                    raise ValueError(f"期望3通道RGB图像，得到{batch.size(1)}通道")
                
                # 检查数值范围
                if batch.min() < -0.1 or batch.max() > 1.1:
                    print(f"⚠️  图像像素范围异常: [{batch.min():.3f}, {batch.max():.3f}]")
                
                # 裁剪到[0, 1]范围
                batch = torch.clamp(batch, 0.0, 1.0)
                
                # 调整大小并标准化
                batch_processed = []
                for img in batch:
                    img_resized = resize(img, (299, 299))
                    img_normalized = self.preprocess(img_resized)
                    batch_processed.append(img_normalized)
                
                batch_processed = torch.stack(batch_processed).to(self.device)
                
                # 提取特征
                feat = self.inception(batch_processed)
                features.append(feat.cpu())
        
        return torch.cat(features, dim=0)
    
    def calculate_fid(self, real_features, fake_features):
        """计算FID分数"""
        # 转换为numpy
        real_features = real_features.numpy()
        fake_features = fake_features.numpy()
        
        # 计算均值和协方差
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        # 计算FID
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid
    
    def evaluate_fid(self, ldm_model, real_loader, vae_model, config):
        """
        评估FID分数
        
        Args:
            ldm_model: 潜在扩散模型
            real_loader: 真实图像数据加载器
            vae_model: VAE模型（实际上不需要，因为LDM.sample()已经包含解码）
            config: 配置字典
        """
        fid_config = config.get('evaluation', {}).get('fid_evaluation', {})
        num_samples = fid_config.get('num_samples', 1000)
        batch_size = fid_config.get('batch_size', 32)
        num_inference_steps = fid_config.get('num_inference_steps', 50)
        num_classes = config.get('unet', {}).get('num_classes', 31)
        
        ldm_model.eval()
        
        # 收集真实图像（RGB，范围[0,1]）
        print(f"🔍 收集真实图像 ({num_samples} 张)...")
        real_images = []
        for batch_idx, batch in enumerate(real_loader):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images = batch[0]  # [B, 3, H, W], 范围[-1, 1]
            else:
                images = batch
            
            # 转换到[0, 1]范围
            images = (images + 1.0) / 2.0
            images = torch.clamp(images, 0.0, 1.0)
            
            real_images.append(images)
            if len(real_images) * images.size(0) >= num_samples:
                break
        
        real_images = torch.cat(real_images, dim=0)[:num_samples]
        
        # 生成假图像
        print(f"🎨 生成假图像 ({num_samples} 张)...")
        fake_images = []
        
        with torch.no_grad():
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            for i in tqdm(range(num_batches), desc="生成图像"):
                current_batch_size = min(batch_size, num_samples - i * batch_size)
                
                # 随机生成类别标签
                if num_classes > 1:
                    class_labels = torch.randint(0, num_classes, (current_batch_size,), device=self.device)
                else:
                    class_labels = None
                
                # LDM生成RGB图像（已经包含了潜变量采样和VAE解码）
                generated_images = ldm_model.sample(
                    num_samples=current_batch_size,
                    class_labels=class_labels,
                    num_inference_steps=num_inference_steps,
                    eta=0.0
                )  # 返回 [B, 3, H, W]，范围[-1, 1]
                
                # 转换到[0, 1]范围
                generated_images = (generated_images + 1.0) / 2.0
                generated_images = torch.clamp(generated_images, 0.0, 1.0)
                
                fake_images.append(generated_images.cpu())
        
        fake_images = torch.cat(fake_images, dim=0)[:num_samples]
        
        # 提取特征
        print(f"🧠 提取真实图像特征...")
        real_features = self.extract_features(real_images)
        
        print(f"🧠 提取生成图像特征...")
        fake_features = self.extract_features(fake_images)
        
        # 计算FID
        fid_score = self.calculate_fid(real_features, fake_features)
        
        return fid_score, fake_images[:8]  # 返回FID分数和一些生成样本用于可视化


class DataLoaderWrapper:
    """数据加载器包装器，将VAE的(image, label)格式适配为LDM需要的格式"""
    
    def __init__(self, original_loader):
        self.original_loader = original_loader
        self.batch_size = original_loader.batch_size
        self.dataset = original_loader.dataset
    
    def __iter__(self):
        for batch in self.original_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, labels = batch[0], batch[1]
                yield images, labels
            else:
                raise ValueError(f"无法处理的批次格式: {type(batch)}")
    
    def __len__(self):
        return len(self.original_loader)


class LDMTrainer:
    """LDM训练器"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        vae_checkpoint_path: str = None,
        device: str = 'cuda'
    ):
        self.config = config
        self.device = device
        self.vae_checkpoint_path = vae_checkpoint_path
        
        # 创建输出目录
        self.output_dir = config.get('output_dir', './ldm_outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
        
        # 初始化模型
        self._init_model()
        
        # 初始化优化器
        self._init_optimizer()
        
        # 初始化数据加载器
        self._init_dataloader()
        
        # 初始化FID评估器
        fid_config = self.config.get('evaluation', {}).get('fid_evaluation', {})
        inception_batch_size = fid_config.get('inception_batch_size', 64)
        self.fid_evaluator = FIDEvaluator(device=self.device, inception_batch_size=inception_batch_size)
        
        # 训练历史
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'fid_score': [],
            'lr': []
        }
        
        # 最佳验证损失和FID
        self.best_val_loss = float('inf')
        self.best_fid_score = float('inf')
        
    def _init_model(self):
        """初始化模型"""
        print("🔧 初始化LDM模型...")
        
        # 获取配置
        unet_config = self.config.get('unet', {})
        diffusion_config = self.config.get('diffusion', {})
        
        # 创建模型
        self.model = create_ldm_model(
            vae_checkpoint_path=self.vae_checkpoint_path,
            unet_config=unet_config,
            diffusion_config=diffusion_config,
            device=self.device
        )
        
        # 保存VAE模型引用，用于FID评估时的解码
        self.vae_model = self.model.vae
        
        print(f"✅ LDM模型创建成功")
        
        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 模型参数统计:")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   训练比例: {trainable_params / total_params * 100:.1f}%")
        
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        train_config = self.config.get('training', {})
        
        # 优化器
        lr = train_config.get('learning_rate', 1e-4)
        weight_decay = train_config.get('weight_decay', 0.01)
        
        self.optimizer = self.model.configure_optimizers(lr=lr, weight_decay=weight_decay)
        
        # 学习率调度器
        scheduler_type = train_config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            # 确保eta_min是浮点数类型
            min_lr = train_config.get('min_lr', 1e-6)
            if isinstance(min_lr, str):
                min_lr = float(min_lr)
            
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config.get('epochs', 100),
                eta_min=min_lr
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_config.get('step_size', 30),
                gamma=train_config.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
            
        print(f"📈 优化器: AdamW (lr={lr}, weight_decay={weight_decay})")
        if self.scheduler:
            print(f"📉 学习率调度器: {scheduler_type}")
        
    def _init_dataloader(self):
        """初始化数据加载器"""
        data_config = self.config.get('data', {})
        
        # 数据路径
        data_dir = data_config.get('data_dir', '/kaggle/input/dataset/dataset')
        
        # 数据增强 - 保持与VAE训练一致的归一化
        train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),  # 匹配VAE训练时的尺寸
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])
        
        val_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 批次大小
        batch_size = data_config.get('batch_size', 8)
        num_workers = data_config.get('num_workers', 2)
        
        try:
            if vae_build_dataloader is not None:
                # 使用VAE的数据加载器
                print(f"🔄 使用VAE数据加载器")
                train_loader, val_loader, train_size, val_size = vae_build_dataloader(
                    root_dir=data_dir,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle_train=True,
                    shuffle_val=False,
                    val_split=0.2,  # 20%验证集
                    random_state=42
                )
                
                # 包装数据加载器以适配LDM格式
                self.train_loader = DataLoaderWrapper(train_loader)
                self.val_loader = DataLoaderWrapper(val_loader)
                
                print(f"✅ VAE数据加载器创建成功")
                print(f"   训练集: {train_size} 样本, {len(self.train_loader)} 批次")
                print(f"   验证集: {val_size} 样本, {len(self.val_loader)} 批次")
                print(f"   批次大小: {batch_size}")
            else:
                # 回退到合成数据
                self._create_synthetic_dataloaders(batch_size, num_workers)
                
        except Exception as e:
            print(f"❌ 数据加载器创建失败: {e}")
            print(f"🔄 回退到合成数据...")
            self._create_synthetic_dataloaders(batch_size, num_workers)
    
    def _create_synthetic_dataloaders(self, batch_size: int, num_workers: int):
        """创建合成数据加载器作为备用方案"""
        from torch.utils.data import Dataset, DataLoader
        
        class SyntheticDataset(Dataset):
            def __init__(self, num_samples: int, num_classes: int):
                self.num_samples = num_samples
                self.num_classes = num_classes
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                # 生成随机图像，范围[-1, 1]
                image = torch.randn(3, 256, 256)
                label = torch.randint(0, self.num_classes, (1,)).item()
                return image, label
        
        num_classes = self.config.get('unet', {}).get('num_classes', 31)
        
        train_dataset = SyntheticDataset(1000, num_classes)
        val_dataset = SyntheticDataset(200, num_classes)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # 合成数据不需要多进程
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )
        
        print(f"✅ 合成数据集创建成功")
        print(f"   训练集: {len(train_dataset)} 样本, {len(self.train_loader)} 批次")
        print(f"   验证集: {len(val_dataset)} 样本, {len(self.val_loader)} 批次")
        print(f"   类别数: {num_classes}")

    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1} - Training",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 解包批次数据
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, class_labels = batch[0], batch[1]
            else:
                images = batch
                class_labels = None
            
            images = images.to(self.device)
            if class_labels is not None:
                class_labels = class_labels.to(self.device)
            
            # 前向传播
            loss_dict = self.model(images, class_labels)
            loss = loss_dict['loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.unet.parameters(), max_norm=1.0)
            
            # 优化器步进
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{avg_loss:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 清理显存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1} - Validation",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 解包批次数据
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, class_labels = batch[0], batch[1]
            else:
                images = batch
                class_labels = None
                
            images = images.to(self.device)
            if class_labels is not None:
                class_labels = class_labels.to(self.device)
            
            # 前向传播
            loss_dict = self.model(images, class_labels)
            loss = loss_dict['loss']
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            progress_bar.set_postfix({'Val Loss': f'{avg_loss:.4f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate_fid(self, epoch: int) -> float:
        """评估FID分数"""
        print(f"🎯 开始FID评估 (Epoch {epoch + 1})...")
        
        try:
            fid_score, sample_images = self.fid_evaluator.evaluate_fid(
                ldm_model=self.model,
                real_loader=self.val_loader,
                vae_model=None,  # 不再需要，因为model.sample()已经包含解码
                config=self.config
            )
            
            print(f"📊 FID Score: {fid_score:.2f}")
            
            # 保存FID评估的样本图像
            if len(sample_images) > 0:
                sample_images = (sample_images + 1.0) / 2.0  # [-1,1] -> [0,1]
                sample_images = torch.clamp(sample_images, 0.0, 1.0)
                
                fig, axes = plt.subplots(2, 4, figsize=(12, 6))
                axes = axes.flatten()
                
                for i in range(min(8, len(sample_images))):
                    img = sample_images[i].permute(1, 2, 0).cpu().numpy()
                    axes[i].imshow(img)
                    axes[i].axis('off')
                    axes[i].set_title(f'Generated')
                
                plt.tight_layout()
                fid_sample_path = os.path.join(self.output_dir, 'samples', f'fid_epoch_{epoch+1:03d}.png')
                plt.savefig(fid_sample_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"📸 FID样本图像已保存: {fid_sample_path}")
            
            return fid_score
            
        except Exception as e:
            print(f"❌ FID评估失败: {e}")
            return float('inf')
    
    @torch.no_grad()
    def generate_samples(self, epoch: int, num_samples: int = 4):
        """生成样本图像"""
        self.model.eval()
        
        # 从配置读取参数
        sample_config = self.config.get('evaluation', {}).get('sample_generation', {})
        inference_steps = sample_config.get('inference_steps', 50)
        
        # 随机选择类别标签
        if self.model.unet.num_classes and self.model.unet.num_classes > 1:
            class_labels = torch.randint(0, self.model.unet.num_classes, (num_samples,), device=self.device)
        else:
            class_labels = None
        
        # 生成图像
        generated_images = self.model.sample(
            num_samples=num_samples,
            class_labels=class_labels,
            num_inference_steps=inference_steps,
            eta=0.0  # DDIM确定性采样
        )
        
        # 转换到[0, 1]范围
        generated_images = (generated_images + 1.0) / 2.0
        generated_images = torch.clamp(generated_images, 0.0, 1.0)
        
        # 保存图像
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            img = generated_images[i].cpu().permute(1, 2, 0).numpy()
            axes[i].imshow(img)
            axes[i].axis('off')
            if class_labels is not None:
                axes[i].set_title(f'Class {class_labels[i].item()}')
        
        plt.tight_layout()
        sample_path = os.path.join(self.output_dir, 'samples', f'epoch_{epoch+1:03d}.png')
        plt.savefig(sample_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📸 样本图像已保存: {sample_path}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, is_best_fid: bool = False):
        """保存检查点"""
        # 保存最新检查点
        checkpoint_path = os.path.join(self.output_dir, 'checkpoints', 'latest.pth')
        self.model.save_checkpoint(
            filepath=checkpoint_path,
            epoch=epoch,
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
            metrics={'train_history': self.train_history}
        )
        
        # 保存最佳验证损失模型
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoints', 'best_val_loss.pth')
            
            # 删除旧的最佳验证损失模型
            old_best_path = os.path.join(self.output_dir, 'checkpoints', 'best_val_loss.pth')
            if os.path.exists(old_best_path):
                try:
                    os.remove(old_best_path)
                    print(f"🗑️  删除旧的最佳验证损失模型")
                except Exception as e:
                    print(f"⚠️  删除旧模型失败: {e}")
            
            self.model.save_checkpoint(
                filepath=best_path,
                epoch=epoch,
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
                metrics={'train_history': self.train_history}
            )
            print(f"🏆 最佳验证损失模型已保存: {best_path}")
        
        # 保存最佳FID模型
        if is_best_fid:
            best_fid_path = os.path.join(self.output_dir, 'checkpoints', 'best_fid.pth')
            
            # 删除旧的最佳FID模型
            old_best_fid_path = os.path.join(self.output_dir, 'checkpoints', 'best_fid.pth')
            if os.path.exists(old_best_fid_path):
                try:
                    os.remove(old_best_fid_path)
                    print(f"🗑️  删除旧的最佳FID模型")
                except Exception as e:
                    print(f"⚠️  删除旧FID模型失败: {e}")
            
            self.model.save_checkpoint(
                filepath=best_fid_path,
                epoch=epoch,
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
                metrics={'train_history': self.train_history}
            )
            print(f"🎯 最佳FID模型已保存: {best_fid_path}")
        
        # 定期保存（根据配置）
        save_every_epochs = self.config.get('save_every_epochs', 10)
        if (epoch + 1) % save_every_epochs == 0:
            periodic_path = os.path.join(self.output_dir, 'checkpoints', f'epoch_{epoch+1:03d}.pth')
            self.model.save_checkpoint(
                filepath=periodic_path,
                epoch=epoch,
                metrics={'train_history': self.train_history}
            )
            
            # 删除旧的定期检查点（保留最近3个）
            checkpoints_dir = os.path.join(self.output_dir, 'checkpoints')
            try:
                # 获取所有epoch检查点文件
                epoch_files = []
                for filename in os.listdir(checkpoints_dir):
                    if filename.startswith('epoch_') and filename.endswith('.pth'):
                        try:
                            epoch_num = int(filename.split('_')[1].split('.')[0])
                            epoch_files.append((epoch_num, filename))
                        except (ValueError, IndexError):
                            continue
                
                # 按epoch编号排序
                epoch_files.sort(key=lambda x: x[0])
                
                # 删除旧的检查点，保留最近3个
                if len(epoch_files) > 3:
                    files_to_delete = epoch_files[:-3]  # 除了最后3个
                    for epoch_num, filename in files_to_delete:
                        old_path = os.path.join(checkpoints_dir, filename)
                        if os.path.exists(old_path):
                            os.remove(old_path)
                            print(f"🗑️  删除旧检查点: {filename}")
                            
            except Exception as e:
                print(f"⚠️  清理旧检查点时出错: {e}")
        
        print(f"💾 检查点已保存: {checkpoint_path}")
    
    def train(self, start_epoch: int = 0):
        """主训练循环"""
        train_config = self.config.get('training', {})
        epochs = train_config.get('epochs', 100)
        
        print(f"🚀 开始LDM训练...")
        print(f"   总轮次: {epochs}")
        print(f"   起始轮次: {start_epoch + 1}")
        print(f"   输出目录: {self.output_dir}")
        
        for epoch in range(start_epoch, epochs):
            print(f"\n📅 Epoch {epoch + 1}/{epochs}")
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate_epoch(epoch)
            
            # FID评估（根据配置）
            fid_config = self.config.get('evaluation', {}).get('fid_evaluation', {})
            sample_config = self.config.get('evaluation', {}).get('sample_generation', {})
            
            fid_score = None
            if fid_config.get('enabled', True) and (epoch + 1) % fid_config.get('eval_every_epochs', 5) == 0:
                fid_score = self.evaluate_fid(epoch)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 记录历史
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['fid_score'].append(fid_score if fid_score is not None else None)
            self.train_history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # 打印统计
            print(f"📊 Train Loss: {train_loss:.4f}")
            print(f"📊 Val Loss: {val_loss:.4f}")
            if fid_score is not None:
                print(f"📊 FID Score: {fid_score:.2f}")
            print(f"📊 Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 检查是否为最佳模型
            is_best = val_loss < self.best_val_loss
            is_best_fid = False
            
            if is_best:
                self.best_val_loss = val_loss
                print(f"🎯 新的最佳验证损失: {val_loss:.4f}")
            
            if fid_score is not None and fid_score < self.best_fid_score:
                self.best_fid_score = fid_score
                is_best_fid = True
                print(f"🎯 新的最佳FID分数: {fid_score:.2f}")
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best, is_best_fid)
            
            # 生成样本（根据配置）
            if (epoch + 1) % sample_config.get('sample_every_epochs', 5) == 0:
                num_samples = sample_config.get('num_sample_images', 4)
                self.generate_samples(epoch, num_samples)
            
            # 清理显存
            torch.cuda.empty_cache()
        
        print(f"\n✅ 训练完成!")
        print(f"🏆 最佳验证损失: {self.best_val_loss:.4f}")
        print(f"🎯 最佳FID分数: {self.best_fid_score:.2f}")
        
        # 保存训练历史
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        print(f"📈 训练历史已保存: {history_path}")


def load_config(config_path: str = None) -> Dict[str, Any]:
    """加载配置文件"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        print(f"✅ 配置文件加载成功: {config_path}")
    else:
        # 默认配置
        config = {
            'data': {
                'data_dir': '/kaggle/input/dataset/dataset',
                'batch_size': 8,  # 保守的批次大小
                'num_workers': 2
            },
            'unet': {
                'in_channels': 4,
                'out_channels': 4,
                'model_channels': 320,
                'num_res_blocks': 2,
                'attention_resolutions': [4, 2, 1],
                'channel_mult': [1, 2, 4, 4],
                'num_classes': 31,
                'dropout': 0.1,
                'num_heads': 8,
                'use_scale_shift_norm': True,
                'resblock_updown': True
            },
            'diffusion': {
                'timesteps': 1000,
                'beta_schedule': 'cosine',
                'beta_start': 0.0001,  # 0.0001
                'beta_end': 0.02
            },
            'training': {
                'epochs': 50,
                'learning_rate': 0.0001,  # 1e-4
                'weight_decay': 0.01,
                'scheduler': 'cosine',
                'min_lr': 0.000001  # 1e-6
            },
            'output_dir': './ldm_outputs'
        }
        print("⚠️  使用默认配置")
    
    return config


def main():
    """主函数"""
    # 配置路径
    config_path = "ldm_config.yaml"  # 如果存在的话
    vae_checkpoint_path = None  # 将根据实际情况设置
    
    # 查找VAE检查点
    possible_vae_paths = [
        "/kaggle/input/vae-model/vae_finetuned_epoch_23.pth",  # 用户提供的预训练模型
        "VAE/vae_model_best.pth",
        "VAE/vae_model_final.pth",
        "VAE/checkpoints/best_vae_checkpoint.pth",
        "VAE/checkpoints/vae_epoch_025.pth",
        "../VAE/vae_model_best.pth"
    ]
    
    for path in possible_vae_paths:
        if os.path.exists(path):
            vae_checkpoint_path = path
            print(f"🔍 找到VAE检查点: {path}")
            break
    
    if vae_checkpoint_path is None:
        print("⚠️  未找到VAE检查点，将使用预训练的Stable Diffusion VAE")
    
    # 加载配置
    config = load_config(config_path)
    
    # 创建训练器
    trainer = LDMTrainer(
        config=config,
        vae_checkpoint_path=vae_checkpoint_path,
        device=device
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main() 
