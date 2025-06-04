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
from pathlib import Path
import shutil

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
    
    def __init__(self, device='cuda', inception_batch_size=32):  # 减少默认批次大小
        self.device = device
        self.inception_batch_size = inception_batch_size
        
        # Inception模型将在需要时加载，用完就卸载
        self.inception = None
        self.preprocess = None
        
        print("✅ FID评估器初始化完成（延迟加载Inception模型）")
    
    def _load_inception_model(self):
        """延迟加载Inception模型"""
        if self.inception is None:
            print("🔄 加载Inception-v3模型...")
            
            # 清理显存
            torch.cuda.empty_cache()
            
            # 加载预训练的Inception-v3模型
            self.inception = models.inception_v3(pretrained=True, transform_input=False)
            self.inception.fc = nn.Identity()  # 移除最后的分类层
            self.inception.eval()
            self.inception.to(self.device)
            
            # 图像预处理（用于Inception-v3）
            self.preprocess = transforms.Compose([
                transforms.Resize((299, 299)),  # Inception-v3需要299x299输入
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print("✅ Inception-v3模型加载完成")
    
    def _unload_inception_model(self):
        """卸载Inception模型释放内存"""
        if self.inception is not None:
            print("🗑️  卸载Inception-v3模型...")
            self.inception = None
            self.preprocess = None
            torch.cuda.empty_cache()
            print("✅ Inception-v3模型已卸载")
    
    def _check_memory(self, stage=""):
        """检查GPU内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"💾 {stage} GPU内存: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB")
    
    def extract_features(self, images):
        """
        提取图像特征
        输入: [N, 3, H, W] 的RGB图像，范围[0, 1]
        """
        self._load_inception_model()
        features = []
        
        # 使用更小的批次大小避免内存问题
        effective_batch_size = min(self.inception_batch_size, 16)  # 最大16
        
        with torch.no_grad():
            for i in range(0, len(images), effective_batch_size):
                batch = images[i:i + effective_batch_size]
                
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
                
                # 及时清理中间结果
                del batch_processed, feat
                
                # 每处理几个批次清理一次内存
                if (i // effective_batch_size) % 5 == 0:
                    torch.cuda.empty_cache()
        
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
        self._check_memory("FID评估开始")
        
        fid_config = config.get('evaluation', {}).get('fid_evaluation', {})
        num_samples = fid_config.get('num_samples', 500)  # 减少默认样本数
        batch_size = fid_config.get('batch_size', 16)  # 减少批次大小
        num_inference_steps = fid_config.get('num_inference_steps', 50)
        num_classes = config.get('unet', {}).get('num_classes', 31)
        
        # 根据可用内存动态调整参数
        available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        free_memory = available_memory - allocated_memory
        
        if free_memory < 3:  # 小于3GB可用内存
            num_samples = min(num_samples, 200)  # 进一步减少样本数
            batch_size = min(batch_size, 8)    # 进一步减少批次大小
            print(f"⚠️  可用显存不足 ({free_memory:.1f}GB)，调整参数：samples={num_samples}, batch_size={batch_size}")
        
        ldm_model.eval()
        
        try:
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
                
                real_images.append(images.cpu())  # 立即移到CPU
                if len(real_images) * images.size(0) >= num_samples:
                    break
            
            real_images = torch.cat(real_images, dim=0)[:num_samples]
            self._check_memory("真实图像收集完成")
            
            # 生成假图像
            print(f"🎨 生成假图像 ({num_samples} 张)...")
            fake_images = []
            
            # 禁用采样过程中的进度条
            old_tqdm_disable = os.environ.get('TQDM_DISABLE', '0')
            os.environ['TQDM_DISABLE'] = '1'
            
            try:
                with torch.no_grad():
                    num_batches = (num_samples + batch_size - 1) // batch_size
                    
                    # 简化进度条显示，不显示具体描述只显示总体进度
                    for i in range(num_batches):
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
                        
                        fake_images.append(generated_images.cpu())  # 立即移到CPU
                        
                        # 简化进度显示
                        if (i + 1) % max(1, num_batches // 4) == 0 or i == num_batches - 1:
                            print(f"   📊 已生成 {min((i + 1) * batch_size, num_samples)}/{num_samples} 张图像")
                        
                        # 清理显存
                        del generated_images
                        if class_labels is not None:
                            del class_labels
                        torch.cuda.empty_cache()
            finally:
                # 恢复tqdm设置
                os.environ['TQDM_DISABLE'] = old_tqdm_disable
            
            fake_images = torch.cat(fake_images, dim=0)[:num_samples]
            self._check_memory("假图像生成完成")
            
            # 提取特征
            print(f"🧠 提取真实图像特征...")
            real_features = self.extract_features(real_images)
            
            # 清理真实图像内存
            del real_images
            torch.cuda.empty_cache()
            self._check_memory("真实图像特征提取完成")
            
            print(f"🧠 提取生成图像特征...")
            fake_features = self.extract_features(fake_images)
            sample_images = fake_images[:8].clone()  # 只保留8张用于可视化
            
            # 清理假图像内存
            del fake_images
            torch.cuda.empty_cache()
            
            # 卸载Inception模型
            self._unload_inception_model()
            self._check_memory("Inception模型卸载完成")
            
            # 计算FID
            fid_score = self.calculate_fid(real_features, fake_features)
            
            return fid_score, sample_images
            
        except Exception as e:
            # 出错时确保清理资源
            print(f"❌ FID评估过程中出错: {e}")
            self._unload_inception_model()
            
            # 强制清理所有可能的变量
            try:
                if 'real_images' in locals():
                    del real_images
                if 'fake_images' in locals():
                    del fake_images
                if 'real_features' in locals():
                    del real_features
                if 'fake_features' in locals():
                    del fake_features
            except:
                pass
            
            torch.cuda.empty_cache()
            self._check_memory("错误清理完成")
            raise e


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

    @torch.no_grad()
    def compute_vae_scaling_factor(self, num_samples: int = 1000) -> float:
        """
        计算微调后VAE的最佳缩放因子
        
        Args:
            num_samples: 用于统计的样本数量
            
        Returns:
            optimal_scaling_factor: 最佳缩放因子
        """
        print(f"🔍 计算VAE缩放因子 (使用 {num_samples} 个样本)...")
        
        self.vae_model.eval()
        all_latents = []
        sample_count = 0
        
        # Get the VAE model to be used for encoding
        # It should be the one loaded by LDMTrainer, typically already on the correct device
        temp_vae = self.vae_model 
        temp_vae.eval() # Ensure it's in evaluation mode

        print(f"🔍 计算缩放因子：从 {len(self.train_loader)} 个批次中处理最多 {num_samples} 个样本...")
        with torch.no_grad(): # Crucial: no gradients needed here
            for batch_idx, batch_data in enumerate(self.train_loader):
                # Try to gracefully handle different batch structures
                if isinstance(batch_data, (list, tuple)) and len(batch_data) > 0:
                    images = batch_data[0]
                elif torch.is_tensor(batch_data):
                    images = batch_data
                else:
                    print(f"⚠️  跳过无法解析的批次数据类型: {type(batch_data)}")
                    continue
                
                images = images.to(self.device) # Move images to the correct device

                # Encode using the VAE
                # The VAE (AutoencoderKL from diffusers) encode method returns an object
                # with a `latent_dist` attribute, which is a DiagonalGaussianDistribution.
                posterior = temp_vae.encode(images).latent_dist
                
                # 🔧 重要修复：为了与实际编码方式保持一致，这里也使用 .sample()
                # 但收集多个样本来获得更稳定的统计估计
                latents = posterior.sample()  # 使用与encode_to_latent相同的方法
                
                # 📊 添加调试信息：比较 mode 和 sample 的差异
                if batch_idx == 0:  # 只在第一个批次打印
                    latents_mode = posterior.mode()
                    print(f"   调试信息 - Mode vs Sample:")
                    print(f"     Mode:   范围[{latents_mode.min():.3f}, {latents_mode.max():.3f}], std={latents_mode.std():.3f}")
                    print(f"     Sample: 范围[{latents.min():.3f}, {latents.max():.3f}], std={latents.std():.3f}")
                    
                    # 检查方差(logvar)的大小，这能解释为什么sample与mode差异很大
                    logvar = posterior.logvar
                    print(f"     LogVar: 范围[{logvar.min():.3f}, {logvar.max():.3f}], mean={logvar.mean():.3f}")
                    print(f"     Var:    范围[{torch.exp(logvar).min():.3f}, {torch.exp(logvar).max():.3f}], mean={torch.exp(logvar).mean():.3f}")

                all_latents.append(latents.cpu()) # Move to CPU to save GPU memory
                sample_count += latents.shape[0]
                
                if sample_count >= num_samples:
                    print(f"   已收集 {sample_count} 个潜变量样本，停止收集。")
                    break
        
        if not all_latents:
            print("⚠️  无法收集潜变量样本，使用默认缩放因子")
            return 0.18215
        
        # 合并所有样本
        all_latents = torch.cat(all_latents, dim=0)[:num_samples]
        
        # 计算统计信息
        mean = all_latents.mean()
        std = all_latents.std()
        
        # 最佳缩放因子 = 1 / std
        # 目标是使缩放后的潜变量标准差接近1
        optimal_scaling_factor = 1.0 / std.item()
        
        print(f"📊 VAE潜变量统计:")
        print(f"   样本数量: {len(all_latents)}")
        print(f"   均值: {mean:.6f}")
        print(f"   标准差: {std:.6f}")
        print(f"   建议缩放因子: {optimal_scaling_factor:.6f}")
        print(f"   当前缩放因子: {self.model.scaling_factor:.6f}")
        
        # 检查是否需要更新
        current_factor = self.model.scaling_factor
        factor_diff = abs(optimal_scaling_factor - current_factor) / current_factor
        
        if factor_diff > 0.1:  # 差异超过10%
            print(f"⚠️  缩放因子差异较大 ({factor_diff*100:.1f}%)，建议更新")
        else:
            print(f"✅ 缩放因子差异较小 ({factor_diff*100:.1f}%)，当前设置合适")
        
        return optimal_scaling_factor

    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # 获取总epoch数
        total_epochs = self.config.get('training', {}).get('epochs', 100)
        
        # 从配置读取详细监控设置
        monitoring_config = self.config.get('training', {}).get('detailed_monitoring', {})
        detailed_monitoring = (
            monitoring_config.get('enabled', True) and 
            epoch < monitoring_config.get('monitor_epochs', 10)
        )
        monitor_batches = monitoring_config.get('monitor_batches', 5)
        summary_interval = monitoring_config.get('summary_interval', 50)
        
        if detailed_monitoring:
            print(f"\n🔍 第 {epoch+1} 轮详细监控:")
            latent_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
            noise_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
            pred_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
            loss_components = []
            gradient_norms = []
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{total_epochs}",
            leave=False,
            ncols=80,  # 限制进度条宽度
            ascii=True  # 使用ASCII字符，避免编码问题
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
            
            # 详细监控：输入图像统计
            if detailed_monitoring and batch_idx < monitor_batches:
                img_min, img_max = images.min().item(), images.max().item()
                img_mean, img_std = images.mean().item(), images.std().item()
                print(f"  批次 {batch_idx+1} - 输入图像: 范围[{img_min:.3f}, {img_max:.3f}], 均值{img_mean:.3f}, 标准差{img_std:.3f}")
            
            # 前向传播
            loss_dict = self.model(images, class_labels)
            loss = loss_dict['loss']
            
            # 详细监控：获取中间结果
            if detailed_monitoring and batch_idx < monitor_batches:
                with torch.no_grad():
                    # 获取潜变量
                    if hasattr(self.model, 'vae') and self.model.vae is not None:
                        try:
                            # VAE编码
                            posterior = self.model.vae.encode(images)
                            latents = posterior.latent_dist.sample() * self.model.scaling_factor
                            
                            # 统计潜变量
                            lat_min, lat_max = latents.min().item(), latents.max().item()
                            lat_mean, lat_std = latents.mean().item(), latents.std().item()
                            latent_stats['min'].append(lat_min)
                            latent_stats['max'].append(lat_max)
                            latent_stats['mean'].append(lat_mean)
                            latent_stats['std'].append(lat_std)
                            
                            print(f"    潜变量(缩放后): 范围[{lat_min:.3f}, {lat_max:.3f}], 均值{lat_mean:.3f}, 标准差{lat_std:.3f}")
                            
                            # 生成随机噪音时间步
                            batch_size = latents.shape[0]
                            timesteps = torch.randint(0, self.model.diffusion.timesteps, (batch_size,), device=self.device)
                            
                            # 添加噪音
                            noise = torch.randn_like(latents)
                            noisy_latents = self.model.diffusion.q_sample(latents, timesteps, noise)
                            
                            # 统计噪音和加噪后的潜变量
                            noise_min, noise_max = noise.min().item(), noise.max().item()
                            noise_mean, noise_std = noise.mean().item(), noise.std().item()
                            noise_stats['min'].append(noise_min)
                            noise_stats['max'].append(noise_max)
                            noise_stats['mean'].append(noise_mean)
                            noise_stats['std'].append(noise_std)
                            
                            noisy_min, noisy_max = noisy_latents.min().item(), noisy_latents.max().item()
                            noisy_mean, noisy_std = noisy_latents.mean().item(), noisy_latents.std().item()
                            
                            print(f"    噪音: 范围[{noise_min:.3f}, {noise_max:.3f}], 均值{noise_mean:.3f}, 标准差{noise_std:.3f}")
                            print(f"    加噪潜变量: 范围[{noisy_min:.3f}, {noisy_max:.3f}], 均值{noisy_mean:.3f}, 标准差{noisy_std:.3f}")
                            
                            # U-Net预测
                            model_pred = self.model.unet(noisy_latents, timesteps, class_labels)
                            pred_min, pred_max = model_pred.min().item(), model_pred.max().item()
                            pred_mean, pred_std = model_pred.mean().item(), model_pred.std().item()
                            pred_stats['min'].append(pred_min)
                            pred_stats['max'].append(pred_max)
                            pred_stats['mean'].append(pred_mean)
                            pred_stats['std'].append(pred_std)
                            
                            print(f"    模型预测: 范围[{pred_min:.3f}, {pred_max:.3f}], 均值{pred_mean:.3f}, 标准差{pred_std:.3f}")
                            
                        except Exception as e:
                            print(f"    ⚠️ 详细监控失败: {e}")
            
            # 记录损失组件
            if detailed_monitoring:
                loss_components.append(loss.item())
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 详细监控：梯度统计
            if detailed_monitoring and batch_idx < monitor_batches:
                total_norm = 0
                param_count = 0
                for p in self.model.unet.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += p.numel()
                total_norm = total_norm ** (1. / 2)
                gradient_norms.append(total_norm)
                print(f"    梯度范数: {total_norm:.6f}, 参数数量: {param_count:,}")
            
            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.unet.parameters(), max_norm=1.0)
            
            # 优化器步进
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{avg_loss:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'GradNorm': f'{grad_norm:.3f}' if isinstance(grad_norm, (int, float)) else 'N/A'
            })
            
            # 清理显存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            # 详细监控：定期总结
            if detailed_monitoring and (batch_idx + 1) % summary_interval == 0:
                self._print_monitoring_summary(epoch, batch_idx, latent_stats, noise_stats, pred_stats, loss_components, gradient_norms)
        
        # 详细监控：epoch结束总结
        if detailed_monitoring:
            self._print_epoch_monitoring_summary(epoch, latent_stats, noise_stats, pred_stats, loss_components, gradient_norms)
        
        return total_loss / num_batches
    
    def _print_monitoring_summary(self, epoch, batch_idx, latent_stats, noise_stats, pred_stats, loss_components, gradient_norms):
        """打印监控数据摘要"""
        print(f"\n  📊 批次 1-{batch_idx+1} 统计摘要:")
        
        if latent_stats['mean']:
            lat_mean_avg = np.mean(latent_stats['mean'])
            lat_std_avg = np.mean(latent_stats['std'])
            print(f"    潜变量: 平均均值={lat_mean_avg:.4f}, 平均标准差={lat_std_avg:.4f}")
        
        if noise_stats['mean']:
            noise_mean_avg = np.mean(noise_stats['mean'])
            noise_std_avg = np.mean(noise_stats['std'])
            print(f"    噪音: 平均均值={noise_mean_avg:.4f}, 平均标准差={noise_std_avg:.4f}")
        
        if pred_stats['mean']:
            pred_mean_avg = np.mean(pred_stats['mean'])
            pred_std_avg = np.mean(pred_stats['std'])
            print(f"    预测: 平均均值={pred_mean_avg:.4f}, 平均标准差={pred_std_avg:.4f}")
        
        if loss_components:
            loss_avg = np.mean(loss_components)
            loss_std = np.std(loss_components)
            print(f"    损失: 平均={loss_avg:.4f}, 标准差={loss_std:.4f}")
        
        if gradient_norms:
            grad_avg = np.mean(gradient_norms)
            grad_std = np.std(gradient_norms)
            print(f"    梯度: 平均范数={grad_avg:.6f}, 标准差={grad_std:.6f}")
    
    def _print_epoch_monitoring_summary(self, epoch, latent_stats, noise_stats, pred_stats, loss_components, gradient_norms):
        """打印epoch监控总结"""
        print(f"\n🎯 第 {epoch+1} 轮完整统计:")
        print(f"  缩放因子: {self.model.scaling_factor:.6f}")
        
        if latent_stats['mean']:
            print(f"  潜变量统计:")
            print(f"    均值范围: [{min(latent_stats['mean']):.4f}, {max(latent_stats['mean']):.4f}]")
            print(f"    标准差范围: [{min(latent_stats['std']):.4f}, {max(latent_stats['std']):.4f}]")
            print(f"    数值范围: [{min(latent_stats['min']):.4f}, {max(latent_stats['max']):.4f}]")
        
        if noise_stats['mean']:
            print(f"  噪音统计:")
            print(f"    均值范围: [{min(noise_stats['mean']):.4f}, {max(noise_stats['mean']):.4f}]")
            print(f"    标准差范围: [{min(noise_stats['std']):.4f}, {max(noise_stats['std']):.4f}]")
        
        if pred_stats['mean']:
            print(f"  预测统计:")
            print(f"    均值范围: [{min(pred_stats['mean']):.4f}, {max(pred_stats['mean']):.4f}]")
            print(f"    标准差范围: [{min(pred_stats['std']):.4f}, {max(pred_stats['std']):.4f}]")
        
        if loss_components:
            print(f"  损失统计:")
            print(f"    损失范围: [{min(loss_components):.4f}, {max(loss_components):.4f}]")
            print(f"    平均损失: {np.mean(loss_components):.4f}")
        
        if gradient_norms:
            print(f"  梯度统计:")
            print(f"    梯度范数范围: [{min(gradient_norms):.6f}, {max(gradient_norms):.6f}]")
            print(f"    平均梯度范数: {np.mean(gradient_norms):.6f}")
        
        # 健康检查
        print(f"  🔍 健康检查:")
        checks = []
        
        if latent_stats['std']:
            avg_latent_std = np.mean(latent_stats['std'])
            if 0.8 <= avg_latent_std <= 1.2:
                checks.append("✅ 潜变量标准差正常")
            else:
                checks.append(f"⚠️ 潜变量标准差异常: {avg_latent_std:.3f} (期望: ~1.0)")
        
        if noise_stats['std']:
            avg_noise_std = np.mean(noise_stats['std'])
            if 0.9 <= avg_noise_std <= 1.1:
                checks.append("✅ 噪音标准差正常")
            else:
                checks.append(f"⚠️ 噪音标准差异常: {avg_noise_std:.3f} (期望: ~1.0)")
        
        if loss_components:
            latest_loss = loss_components[-5:]  # 最后5个批次
            avg_recent_loss = np.mean(latest_loss)
            if avg_recent_loss < 1.0:
                checks.append("✅ 损失值合理")
            elif avg_recent_loss > 5.0:
                checks.append(f"⚠️ 损失值过高: {avg_recent_loss:.3f}")
            else:
                checks.append(f"📊 损失值: {avg_recent_loss:.3f}")
        
        if gradient_norms:
            avg_grad_norm = np.mean(gradient_norms)
            if avg_grad_norm < 0.001:
                checks.append("⚠️ 梯度可能过小")
            elif avg_grad_norm > 10.0:
                checks.append("⚠️ 梯度可能过大")
            else:
                checks.append("✅ 梯度范数正常")
        
        for check in checks:
            print(f"    {check}")
        
        print("=" * 60)
    
    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        # 获取总epoch数
        total_epochs = self.config.get('training', {}).get('epochs', 100)
        
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Validation {epoch+1}/{total_epochs}",
            leave=False,
            ncols=80,
            ascii=True
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
        
        # 检查磁盘空间
        output_dir_path = Path(self.output_dir)
        try:
            total, used, free = shutil.disk_usage(output_dir_path)
            free_gb = free / (1024**3)
            print(f"💾 可用磁盘空间: {free_gb:.1f}GB")
            
            if free_gb < 5:  # 小于5GB可用空间
                print(f"❌ 磁盘空间不足 ({free_gb:.1f}GB)，跳过FID评估")
                return float('inf')
        except Exception as e:
            print(f"⚠️  无法检查磁盘空间: {e}")
        
        # 预检查显存是否足够
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated() / 1024**3
            free_memory = available_memory - allocated_memory
            
            if free_memory < 1.5:  # 小于1.5GB可用内存
                print(f"⚠️  可用显存不足 ({free_memory:.1f}GB < 1.5GB)，跳过FID评估")
                return float('inf')
        
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
                sample_images = torch.clamp(sample_images, 0.0, 1.0)  # 确保在[0,1]范围
                
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
            # 强制清理内存
            torch.cuda.empty_cache()
            return float('inf')
    
    @torch.no_grad()
    def generate_samples(self, epoch: int, num_samples: int = 8):
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
        
        print(f"🎨 正在生成 {num_samples} 张样本图像...")
        
        # 临时禁用采样过程的进度条
        old_tqdm_disable = os.environ.get('TQDM_DISABLE', '0')
        os.environ['TQDM_DISABLE'] = '1'  # 禁用所有tqdm进度条
        
        try:
            generated_images = self.model.sample(
                num_samples=num_samples,
                class_labels=class_labels,
                num_inference_steps=inference_steps,
                eta=0.0  # DDIM确定性采样
            )
        finally:
            # 恢复tqdm设置
            os.environ['TQDM_DISABLE'] = old_tqdm_disable
        
        print(f"✅ 样本生成完成")
        
        # 转换到[0, 1]范围
        generated_images = (generated_images + 1.0) / 2.0
        generated_images = torch.clamp(generated_images, 0.0, 1.0)
        
        # 保存图像 - 使用2x4布局
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        for i in range(num_samples):
            img = generated_images[i].cpu().permute(1, 2, 0).numpy()
            axes[i].imshow(img)
            axes[i].axis('off')
            if class_labels is not None:
                axes[i].set_title(f'类别 {class_labels[i].item()}', fontsize=10)
        
        plt.tight_layout()
        sample_path = os.path.join(self.output_dir, 'samples', f'epoch_{epoch+1:03d}.png')
        plt.savefig(sample_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📸 样本图像已保存: {sample_path}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, is_best_fid: bool = False):
        """保存检查点"""
        
        # 检查磁盘空间
        output_dir_path = Path(self.output_dir)
        try:
            total, used, free = shutil.disk_usage(output_dir_path)
            free_gb = free / (1024**3)
            required_gb = 4.0  # 预估需要4GB空间（检查点约3.2GB + 缓冲）
            
            print(f"💾 磁盘空间检查: 可用 {free_gb:.1f}GB, 需要 {required_gb:.1f}GB")
            
            if free_gb < required_gb:
                print(f"❌ 磁盘空间不足！可用 {free_gb:.1f}GB < 需要 {required_gb:.1f}GB")
                print(f"🗑️  尝试清理旧检查点...")
                
                # 紧急清理：删除除了latest.pth之外的所有旧检查点
                checkpoints_dir = Path(self.output_dir) / 'checkpoints'
                if checkpoints_dir.exists():
                    deleted_files = []
                    for file in checkpoints_dir.glob("*.pth"):
                        if file.name not in ['latest.pth']:  # 保留latest.pth
                            try:
                                file_size_gb = file.stat().st_size / (1024**3)
                                file.unlink()
                                deleted_files.append(f"{file.name} ({file_size_gb:.1f}GB)")
                            except Exception as e:
                                print(f"   ⚠️  删除 {file.name} 失败: {e}")
                    
                    if deleted_files:
                        print(f"   🗑️  已删除: {', '.join(deleted_files)}")
                        
                        # 重新检查空间
                        total, used, free = shutil.disk_usage(output_dir_path)
                        free_gb = free / (1024**3)
                        print(f"   💾 清理后可用空间: {free_gb:.1f}GB")
                        
                        if free_gb < required_gb:
                            print(f"   ❌ 清理后空间仍不足，跳过保存")
                            return
                    else:
                        print(f"   ⚠️  无可清理文件，跳过保存")
                        return
        except Exception as e:
            print(f"⚠️  磁盘空间检查失败: {e}")
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.output_dir, 'checkpoints', 'latest.pth')
        try:
            self.model.save_checkpoint(
                filepath=checkpoint_path,
                epoch=epoch,
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
                metrics={'train_history': self.train_history}
            )
            print(f"💾 检查点已保存: {checkpoint_path}")
        except Exception as e:
            print(f"❌ 保存检查点失败: {e}")
            return
    
    def train(self, start_epoch: int = 0):
        """主训练循环"""
        train_config = self.config.get('training', {})
        epochs = train_config.get('epochs', 100)
        
        print(f"🚀 开始LDM训练...")
        print(f"   总轮次: {epochs}")
        print(f"   起始轮次: {start_epoch + 1}")
        print(f"   输出目录: {self.output_dir}")
        
        # 检查并计算VAE缩放因子
        if start_epoch == 0:  # 只在训练开始时检查
            print("\n" + "="*60)
            print("🔍 VAE缩放因子检查与验证")
            print("="*60)
            
            optimal_scaling_factor = self.compute_vae_scaling_factor()
            
            current_factor = self.model.scaling_factor
            factor_diff = abs(optimal_scaling_factor - current_factor) / current_factor
            
            print(f"\n📊 缩放因子分析:")
            print(f"   当前缩放因子: {current_factor:.6f}")
            print(f"   建议缩放因子: {optimal_scaling_factor:.6f}")
            print(f"   相对差异: {factor_diff*100:.1f}%")
            
            if factor_diff > 0.1:  # 差异超过10%
                print(f"\n⚠️  缩放因子差异较大，建议更新")
                
                # 自动更新缩放因子（也可以手动确认）
                auto_update = train_config.get('auto_update_scaling_factor', True)
                if auto_update:
                    print(f"🔄 自动更新缩放因子...")
                    old_factor = self.model.scaling_factor
                    self.model.scaling_factor = optimal_scaling_factor
                    print(f"✅ 缩放因子已更新: {old_factor:.6f} → {optimal_scaling_factor:.6f}")
                    
                    # 验证更新效果
                    print(f"\n🔬 验证更新效果:")
                    with torch.no_grad():
                        # 测试一个小批次的编码
                        test_batch = next(iter(self.train_loader))
                        if isinstance(test_batch, (list, tuple)):
                            test_images = test_batch[0][:4]  # 取4张图片测试
                        else:
                            test_images = test_batch[:4]
                        
                        test_images = test_images.to(self.device)
                        posterior = self.vae_model.encode(test_images)
                        test_latents = posterior.latent_dist.sample() * self.model.scaling_factor
                        
                        test_mean = test_latents.mean().item()
                        test_std = test_latents.std().item()
                        print(f"   编码测试结果: 均值={test_mean:.4f}, 标准差={test_std:.4f}")
                        
                        if 0.8 <= test_std <= 1.2:
                            print(f"   ✅ 缩放效果良好 (标准差接近1.0)")
                        elif test_std < 0.8:
                            print(f"   ⚠️  标准差偏小 ({test_std:.4f})，可能影响生成质量")
                        else:
                            print(f"   ⚠️  标准差偏大 ({test_std:.4f})，可能影响训练稳定性")
                else:
                    print(f"⚠️  建议手动更新配置中的缩放因子")
            else:
                print(f"✅ 缩放因子合适，无需更新")
            
            print("="*60)
            print("✅ 缩放因子检查完成，开始训练...")
            print("="*60 + "\n")
        
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
                num_samples = sample_config.get('num_sample_images', 8)
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
