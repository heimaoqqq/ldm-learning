"""
VAE Fine-tuning - 云服务器完整版本
针对微多普勒时频图优化AutoencoderKL
数据集路径：/kaggle/input/dataset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt

# 云环境初始化
print("🌐 云服务器VAE Fine-tuning环境初始化...")
print(f"📂 当前工作目录: {os.getcwd()}")

# 内存优化设置
torch.backends.cudnn.benchmark = True
# 添加内存分配优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"🚀 CUDA设备: {torch.cuda.get_device_name()}")
    print(f"💾 总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 检查并安装依赖
def check_and_install_dependencies():
    """检查并安装必要的依赖"""
    try:
        import diffusers
        print("✅ diffusers 已安装")
    except ImportError:
        print("📦 安装 diffusers...")
        os.system("pip install diffusers transformers accelerate -q")
    
    try:
        import torchvision
        print("✅ torchvision 已安装")
    except ImportError:
        print("📦 安装 torchvision...")
        os.system("pip install torchvision -q")

check_and_install_dependencies()

# 导入必要模块
from diffusers import AutoencoderKL
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

# 添加混合精度训练支持
try:
    import torch
    # 检查PyTorch版本并使用正确的autocast
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        from torch.amp import autocast
        from torch.cuda.amp import GradScaler
        AUTOCAST_DEVICE = 'cuda'
        MIXED_PRECISION_AVAILABLE = True
        print("✅ 新版混合精度训练可用")
    else:
        from torch.cuda.amp import autocast, GradScaler
        AUTOCAST_DEVICE = None
        MIXED_PRECISION_AVAILABLE = True
        print("✅ 传统混合精度训练可用")
except ImportError:
    MIXED_PRECISION_AVAILABLE = False
    print("⚠️  混合精度训练不可用")

class CloudGaitDataset(Dataset):
    """云环境数据集类"""
    
    def __init__(self, image_paths, class_ids, transform=None):
        self.image_paths = image_paths
        self.class_ids = class_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.class_ids[idx]
        
        # 确保标签在有效范围内
        if label < 0 or label > 30:
            label = max(0, min(30, label))
        
        return image, label

def build_cloud_dataloader(root_dir, batch_size=8, num_workers=0, val_split=0.3):
    """内存优化的云环境数据加载器"""
    print(f"🔍 扫描云数据集目录: {root_dir}")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    all_image_paths = []
    all_class_ids = []
    
    # 检查目录结构
    if not os.path.exists(root_dir):
        raise ValueError(f"❌ 数据集目录不存在: {root_dir}")
    
    # 支持ID_1, ID_2, ... ID_31文件夹结构
    id_folders = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and item.startswith('ID_'):
            id_folders.append(item)
    
    if id_folders:
        print(f"📁 发现ID文件夹结构: {len(id_folders)} 个文件夹")
        
        for folder_name in sorted(id_folders):
            folder_path = os.path.join(root_dir, folder_name)
            
            try:
                # 提取类别ID
                class_id = int(folder_name.split('_')[1]) - 1  # 转换为0-based
                
                if not (0 <= class_id <= 30):
                    print(f"⚠️  跳过超出范围的类别: {folder_name}")
                    continue
                
                # 收集图片
                folder_images = []
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                    folder_images.extend(glob.glob(os.path.join(folder_path, ext)))
                
                print(f"  {folder_name}: {len(folder_images)} 张图片")
                
                all_image_paths.extend(folder_images)
                all_class_ids.extend([class_id] * len(folder_images))
                
            except (ValueError, IndexError) as e:
                print(f"⚠️  无法解析文件夹名 {folder_name}: {e}")
                continue

    if not all_image_paths:
        raise ValueError(f"❌ 在 {root_dir} 中没有找到图片文件")

    print(f"📊 数据集统计:")
    print(f"  总图片数: {len(all_image_paths)}")
    print(f"  类别数: {len(set(all_class_ids))}")

    # 数据集划分
    train_paths, val_paths, train_ids, val_ids = train_test_split(
        all_image_paths, all_class_ids, 
        test_size=val_split, 
        random_state=42,
        stratify=all_class_ids if len(set(all_class_ids)) > 1 else None
    )

    print(f"  训练集: {len(train_paths)} 张")
    print(f"  验证集: {len(val_paths)} 张")

    # 创建数据集和加载器
    train_dataset = CloudGaitDataset(train_paths, train_ids, transform=transform)
    val_dataset = CloudGaitDataset(val_paths, val_ids, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False)
    
    return train_loader, val_loader, len(train_dataset), len(val_dataset)

class CloudVAEFineTuner:
    """云环境VAE Fine-tuning器"""
    
    def __init__(self, data_dir='/kaggle/input/dataset'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 云VAE Fine-tuning 设备: {self.device}")
        
        # 内存优化配置
        self.config = {
            'batch_size': 8,  # 减小batch size以节省显存
            'gradient_accumulation_steps': 2,  # 使用梯度累积保持有效batch size
            'learning_rate': 1e-5,  # 较小学习率保护预训练权重
            'weight_decay': 0.01,
            'max_epochs': 25,  # 完整训练
            'data_dir': data_dir,
            'save_dir': './vae_finetuned',
            'reconstruction_weight': 1.0,
            'kl_weight': 0.1,  # 降低KL权重，关注重建质量
            'perceptual_weight': 0.3,  # 恢复合理的感知损失权重
            'use_perceptual_loss': True,  # 保持感知损失开启
            'save_every_epochs': 5,  # 每5个epoch保存一次
            'eval_every_epochs': 2,  # 每2个epoch评估一次
            'use_gradient_checkpointing': True,  # 启用梯度检查点
            'mixed_precision': MIXED_PRECISION_AVAILABLE,  # 混合精度训练
            'safe_mixed_precision': True,  # 安全模式：遇到问题时自动禁用混合精度
        }
        
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # 1. 创建数据加载器 - 减少num_workers
        print("📁 加载云数据集...")
        self.train_loader, self.val_loader, train_size, val_size = build_cloud_dataloader(
            root_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            num_workers=0,  # 减少workers以节省内存
            val_split=0.3
        )
        print(f"   训练集: {len(self.train_loader)} batches ({train_size} samples)")
        print(f"   验证集: {len(self.val_loader)} batches ({val_size} samples)")
        print(f"   有效batch size: {self.config['batch_size'] * self.config['gradient_accumulation_steps']}")
        
        # 2. 加载预训练VAE
        print("📦 加载预训练AutoencoderKL...")
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="vae"
        ).to(self.device)
        
        # 启用梯度检查点
        if self.config['use_gradient_checkpointing']:
            if hasattr(self.vae, 'enable_gradient_checkpointing'):
                self.vae.enable_gradient_checkpointing()
                print("✅ VAE梯度检查点已启用")
        
        # 解冻VAE参数用于fine-tuning
        for param in self.vae.parameters():
            param.requires_grad = True
        
        print(f"🔓 VAE参数已解冻用于fine-tuning")
        
        # 3. 配置优化器
        self.setup_optimizer()
        
        # 4. 损失函数
        self.mse_loss = nn.MSELoss()
        self.setup_perceptual_loss()
        
        # 5. 混合精度训练
        if self.config['mixed_precision']:
            self.scaler = GradScaler()
            print("✅ 混合精度训练已启用")
        
        # 记录训练历史
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'perceptual_loss': [],
            'val_mse': []
        }
        
        # 清理初始化后的内存
        self.clear_memory()
        
        print("✅ 云VAE Fine-tuning器初始化完成!")
    
    def setup_optimizer(self):
        """设置分层优化器"""
        # 只fine-tune decoder + encoder的最后几层
        finetune_params = []
        
        # Decoder - 全部fine-tune
        finetune_params.extend(self.vae.decoder.parameters())
        
        # Encoder - 只fine-tune最后2个block
        encoder_blocks = list(self.vae.encoder.down_blocks)
        for block in encoder_blocks[-2:]:  # 最后2个block
            finetune_params.extend(block.parameters())
        
        # Post quant conv
        finetune_params.extend(self.vae.quant_conv.parameters())
        finetune_params.extend(self.vae.post_quant_conv.parameters())
        
        total_params = sum(p.numel() for p in finetune_params)
        print(f"🎯 Fine-tune参数数量: {total_params:,}")
        
        self.optimizer = torch.optim.AdamW(
            finetune_params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['max_epochs']
        )
    
    def setup_perceptual_loss(self):
        """设置完整感知损失"""
        if not self.config['use_perceptual_loss']:
            print("🚫 感知损失已关闭以节省内存")
            self.perceptual_net = None
            return
            
        try:
            from torchvision.models import vgg16, VGG16_Weights
            # 使用完整的VGG16特征层，保持感知损失质量
            self.perceptual_net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(self.device)
            for param in self.perceptual_net.parameters():
                param.requires_grad = False
            self.perceptual_net.eval()
            print("✅ 完整VGG16感知损失已设置")
        except:
            print("⚠️  VGG16不可用，跳过感知损失")
            self.perceptual_net = None
    
    def compute_perceptual_loss(self, real, fake):
        """计算完整感知损失"""
        if self.perceptual_net is None:
            return torch.tensor(0.0, device=self.device)
        
        # 确保输入是3通道
        if real.size(1) == 1:
            real = real.repeat(1, 3, 1, 1)
        if fake.size(1) == 1:
            fake = fake.repeat(1, 3, 1, 1)
        
        # 计算VGG特征，真实图像用no_grad优化内存
        with torch.no_grad():
            real_features = self.perceptual_net(real)
        fake_features = self.perceptual_net(fake)
        
        return F.mse_loss(real_features, fake_features)
    
    def vae_loss(self, images):
        """VAE损失函数"""
        # 确保输入图像为正确的数据类型
        if images.dtype != torch.float32 and not self.config.get('mixed_precision', False):
            images = images.float()
        
        # 编码
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample()
        
        # 解码
        reconstructed = self.vae.decode(latents).sample
        
        # 重建损失 - 确保数据类型匹配
        recon_loss = self.mse_loss(reconstructed.float(), images.float())
        
        # KL散度损失
        kl_loss = posterior.kl().mean()
        
        # 感知损失 - 在autocast外计算以避免数据类型问题
        if self.config.get('mixed_precision', False):
            with torch.cuda.amp.autocast(enabled=False):
                perceptual_loss = self.compute_perceptual_loss(images.float(), reconstructed.float())
        else:
            perceptual_loss = self.compute_perceptual_loss(images, reconstructed)
        
        # 总损失
        total_loss = (
            self.config['reconstruction_weight'] * recon_loss +
            self.config['kl_weight'] * kl_loss +
            self.config['perceptual_weight'] * perceptual_loss
        )
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'perceptual_loss': perceptual_loss,
            'reconstructed': reconstructed,
            'latents': latents
        }
    
    def clear_memory(self):
        """增强内存清理"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def train_epoch(self, epoch: int):
        """内存优化的训练epoch"""
        self.vae.train()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_perceptual = 0
        num_batches = 0
        
        # 重置梯度累积
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Fine-tune Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            images, _ = batch  # 忽略标签
            images = images.to(self.device, non_blocking=True)
            
            # 混合精度训练
            if self.config['mixed_precision']:
                if AUTOCAST_DEVICE:
                    # 新版PyTorch
                    with autocast(AUTOCAST_DEVICE, dtype=torch.float16):
                        loss_dict = self.vae_loss(images)
                        loss = loss_dict['total_loss'] / self.config['gradient_accumulation_steps']
                else:
                    # 传统版本
                    with autocast():
                        loss_dict = self.vae_loss(images)
                        loss = loss_dict['total_loss'] / self.config['gradient_accumulation_steps']
                
                # 缩放损失并反向传播
                self.scaler.scale(loss).backward()
            else:
                # 前向传播
                loss_dict = self.vae_loss(images)
                loss = loss_dict['total_loss'] / self.config['gradient_accumulation_steps']
                
                # 反向传播
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                if self.config['mixed_precision']:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # 记录损失（恢复到原始scale）
            actual_loss = loss.item() * self.config['gradient_accumulation_steps']
            total_loss += actual_loss
            total_recon += loss_dict['recon_loss'].item()
            total_kl += loss_dict['kl_loss'].item()
            total_perceptual += loss_dict['perceptual_loss'].item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{actual_loss:.4f}',
                'Recon': f'{loss_dict["recon_loss"].item():.4f}',
                'KL': f'{loss_dict["kl_loss"].item():.4f}',
                'Perc': f'{loss_dict["perceptual_loss"].item():.4f}'
            })
            
            # 保存第一个batch的重建样本
            if batch_idx == 0:
                self.save_reconstruction_samples(
                    images, loss_dict['reconstructed'], epoch
                )
            
            # 更频繁的内存清理
            if batch_idx % 20 == 0:  # 每20个batch清理一次
                del images, loss_dict, loss
                self.clear_memory()
        
        # 处理最后不完整的梯度累积
        if (len(self.train_loader)) % self.config['gradient_accumulation_steps'] != 0:
            if self.config['mixed_precision']:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        # 更新学习率
        self.scheduler.step()
        
        avg_metrics = {
            'avg_loss': total_loss / num_batches,
            'avg_recon': total_recon / num_batches,
            'avg_kl': total_kl / num_batches,
            'avg_perceptual': total_perceptual / num_batches
        }
        
        self.clear_memory()
        return avg_metrics
    
    def save_reconstruction_samples(self, original, reconstructed, epoch):
        """保存重建样本对比"""
        with torch.no_grad():
            # 反归一化
            def denormalize(tensor):
                return torch.clamp((tensor + 1) / 2, 0, 1)
            
            orig = denormalize(original[:8]).cpu()
            recon = denormalize(reconstructed[:8]).cpu()
            
            fig, axes = plt.subplots(2, 8, figsize=(24, 6))
            
            for i in range(8):
                # 原图
                axes[0, i].imshow(orig[i].permute(1, 2, 0))
                axes[0, i].set_title(f'Original {i+1}')
                axes[0, i].axis('off')
                
                # 重建图
                axes[1, i].imshow(recon[i].permute(1, 2, 0))
                axes[1, i].set_title(f'Reconstructed {i+1}')
                axes[1, i].axis('off')
            
            plt.suptitle(f'VAE Reconstruction - Epoch {epoch+1}')
            plt.tight_layout()
            plt.savefig(f'{self.config["save_dir"]}/reconstruction_epoch_{epoch+1}.png',
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    @torch.no_grad()
    def evaluate_reconstruction_quality(self, val_loader):
        """内存优化的评估函数"""
        self.vae.eval()
        
        total_mse = 0
        total_samples = 0
        latent_stats = []
        
        # 限制评估样本数量
        max_eval_batches = min(20, len(val_loader))
        
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_eval_batches:
                break
                
            images, _ = batch
            images = images.to(self.device, non_blocking=True)
            
            # VAE重建
            if self.config['mixed_precision']:
                if AUTOCAST_DEVICE:
                    # 新版PyTorch
                    with autocast(AUTOCAST_DEVICE, dtype=torch.float16):
                        posterior = self.vae.encode(images).latent_dist
                        latents = posterior.sample()
                        reconstructed = self.vae.decode(latents).sample
                else:
                    # 传统版本
                    with autocast():
                        posterior = self.vae.encode(images).latent_dist
                        latents = posterior.sample()
                        reconstructed = self.vae.decode(latents).sample
            else:
                posterior = self.vae.encode(images).latent_dist
                latents = posterior.sample()
                reconstructed = self.vae.decode(latents).sample
            
            # 计算MSE
            mse = F.mse_loss(reconstructed, images, reduction='sum')
            total_mse += mse.item()
            total_samples += images.size(0)
            
            # 收集潜在表示统计
            latent_stats.append({
                'mean': latents.mean().item(),
                'std': latents.std().item(),
                'min': latents.min().item(),
                'max': latents.max().item()
            })
            
            # 清理内存
            del images, posterior, latents, reconstructed, mse
            
            if batch_idx % 5 == 0:
                self.clear_memory()
        
        avg_mse = total_mse / total_samples
        
        # 计算平均潜在空间统计
        avg_latent_stats = {
            'mean': np.mean([s['mean'] for s in latent_stats]),
            'std': np.mean([s['std'] for s in latent_stats]),
            'min': np.min([s['min'] for s in latent_stats]),
            'max': np.max([s['max'] for s in latent_stats])
        }
        
        print(f"📊 重建质量MSE: {avg_mse:.6f}")
        print(f"🧠 潜在空间统计: 均值={avg_latent_stats['mean']:.4f}, "
              f"标准差={avg_latent_stats['std']:.4f}, "
              f"范围=[{avg_latent_stats['min']:.2f}, {avg_latent_stats['max']:.2f}]")
        
        return avg_mse, avg_latent_stats
    
    def save_finetuned_vae(self, epoch, metrics):
        """保存fine-tuned VAE"""
        save_path = f'{self.config["save_dir"]}/vae_finetuned_epoch_{epoch+1}.pth'
        
        torch.save({
            'vae_state_dict': self.vae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'config': self.config,
            'train_history': self.train_history
        }, save_path)
        
        print(f"💾 Fine-tuned VAE保存: {save_path}")
    
    def plot_training_history(self):
        """绘制训练历史"""
        if len(self.train_history['epoch']) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 总损失
        axes[0, 0].plot(self.train_history['epoch'], self.train_history['train_loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # 重建损失
        axes[0, 1].plot(self.train_history['epoch'], self.train_history['recon_loss'])
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE Loss')
        axes[0, 1].grid(True)
        
        # KL损失
        axes[1, 0].plot(self.train_history['epoch'], self.train_history['kl_loss'])
        axes[1, 0].set_title('KL Divergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Loss')
        axes[1, 0].grid(True)
        
        # 验证MSE
        if self.train_history['val_mse']:
            axes[1, 1].plot(self.train_history['epoch'][::self.config['eval_every_epochs']], 
                           self.train_history['val_mse'])
            axes[1, 1].set_title('Validation MSE')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MSE')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.config["save_dir"]}/training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def finetune(self):
        """主要fine-tuning流程"""
        print("🚀 开始云VAE Fine-tuning完整训练...")
        print("=" * 60)
        
        best_recon_loss = float('inf')
        
        for epoch in range(self.config['max_epochs']):
            print(f"\n📅 Epoch {epoch+1}/{self.config['max_epochs']}")
            
            # 训练一个epoch
            train_metrics = self.train_epoch(epoch)
            
            # 记录训练历史
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_metrics['avg_loss'])
            self.train_history['recon_loss'].append(train_metrics['avg_recon'])
            self.train_history['kl_loss'].append(train_metrics['avg_kl'])
            self.train_history['perceptual_loss'].append(train_metrics['avg_perceptual'])
            
            # 定期评估重建质量
            if epoch % self.config['eval_every_epochs'] == 0 or epoch == self.config['max_epochs'] - 1:
                val_mse, latent_stats = self.evaluate_reconstruction_quality(self.val_loader)
                self.train_history['val_mse'].append(val_mse)
            
            print(f"📊 Epoch {epoch+1} 结果:")
            print(f"   总损失: {train_metrics['avg_loss']:.4f}")
            print(f"   重建损失: {train_metrics['avg_recon']:.4f}")
            print(f"   KL损失: {train_metrics['avg_kl']:.4f}")
            print(f"   感知损失: {train_metrics['avg_perceptual']:.4f}")
            print(f"   学习率: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # 保存最佳模型
            if train_metrics['avg_recon'] < best_recon_loss:
                best_recon_loss = train_metrics['avg_recon']
                self.save_finetuned_vae(epoch, train_metrics)
                print(f"✅ 保存最佳模型 (重建损失: {best_recon_loss:.4f})")
            
            # 定期保存检查点
            if epoch % self.config['save_every_epochs'] == 0:
                self.save_finetuned_vae(epoch, train_metrics)
                self.plot_training_history()
            
            print(f"🏆 当前最佳重建损失: {best_recon_loss:.4f}")
            
            # 每个epoch后清理内存
            self.clear_memory()
        
        # 训练完成
        print("\n" + "=" * 60)
        print("🎉 云VAE Fine-tuning 完整训练完成!")
        print(f"🏆 最佳重建损失: {best_recon_loss:.4f}")
        
        # 最终评估
        final_mse, final_latent_stats = self.evaluate_reconstruction_quality(self.val_loader)
        print(f"🎯 最终重建MSE: {final_mse:.6f}")
        
        # 绘制最终训练历史
        self.plot_training_history()
        
        return best_recon_loss

def create_emergency_low_memory_finetuner(data_dir):
    """创建极低内存配置的Fine-tuner"""
    print("🆘 启动应急低内存模式...")
    
    class EmergencyCloudVAEFineTuner(CloudVAEFineTuner):
        def __init__(self, data_dir):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🚀 应急模式 - 云VAE Fine-tuning 设备: {self.device}")
            
            # 极端内存优化配置
            self.config = {
                'batch_size': 4,  # 极小batch size
                'gradient_accumulation_steps': 4,  # 增加梯度累积
                'learning_rate': 1e-5,
                'weight_decay': 0.01,
                'max_epochs': 15,  # 减少训练轮次
                'data_dir': data_dir,
                'save_dir': './vae_finetuned',
                'reconstruction_weight': 1.0,
                'kl_weight': 0.05,  # 进一步降低KL权重
                'perceptual_weight': 0.1,  # 保持轻量感知损失
                'use_perceptual_loss': True,  # 保持感知损失开启
                'save_every_epochs': 10,
                'eval_every_epochs': 5,
                'use_gradient_checkpointing': True,
                'mixed_precision': MIXED_PRECISION_AVAILABLE,
            }
            
            # 直接初始化父类的组件，跳过父类__init__
            os.makedirs(self.config['save_dir'], exist_ok=True)
            
            # 数据加载器
            print("📁 加载应急模式数据集...")
            self.train_loader, self.val_loader, train_size, val_size = build_cloud_dataloader(
                root_dir=self.config['data_dir'],
                batch_size=self.config['batch_size'],
                num_workers=0,
                val_split=0.3
            )
            print(f"   应急模式batch size: {self.config['batch_size']}")
            print(f"   有效batch size: {self.config['batch_size'] * self.config['gradient_accumulation_steps']}")
            
            # 加载VAE
            print("📦 加载预训练AutoencoderKL (应急模式)...")
            self.vae = AutoencoderKL.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                subfolder="vae"
            ).to(self.device)
            
            # 启用梯度检查点
            if hasattr(self.vae, 'enable_gradient_checkpointing'):
                self.vae.enable_gradient_checkpointing()
                print("✅ VAE梯度检查点已启用")
            
            for param in self.vae.parameters():
                param.requires_grad = True
            
            self.setup_optimizer()
            self.mse_loss = nn.MSELoss()
            self.setup_perceptual_loss()  # 会被配置跳过
            
            if self.config['mixed_precision']:
                self.scaler = GradScaler()
                print("✅ 混合精度训练已启用")
            
            self.train_history = {
                'epoch': [], 'train_loss': [], 'recon_loss': [],
                'kl_loss': [], 'perceptual_loss': [], 'val_mse': []
            }
            
            self.clear_memory()
            print("✅ 应急模式Fine-tuning器初始化完成!")
        
        def setup_perceptual_loss(self):
            """应急模式轻量级感知损失"""
            try:
                from torchvision.models import vgg16, VGG16_Weights
                # 应急模式使用更少的VGG层
                self.perceptual_net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:8].to(self.device)
                for param in self.perceptual_net.parameters():
                    param.requires_grad = False
                self.perceptual_net.eval()
                print("✅ 应急模式轻量级VGG感知损失已设置")
            except:
                print("⚠️  VGG16不可用，应急模式跳过感知损失")
                self.perceptual_net = None
    
    return EmergencyCloudVAEFineTuner(data_dir)

def main():
    """主函数"""
    print("🌐 启动云VAE Fine-tuning完整训练...")
    
    # 设置CUDA内存分配策略
    if torch.cuda.is_available():
        # 启用内存分片管理
        torch.cuda.set_per_process_memory_fraction(0.95)  # 使用95%的显存
        torch.cuda.empty_cache()
        print(f"🔧 CUDA内存优化设置完成")
    
    # 云环境数据路径
    data_dir = '/kaggle/input/dataset'
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据集目录不存在: {data_dir}")
        return None
    
    try:
        # 创建fine-tuner
        finetuner = CloudVAEFineTuner(data_dir=data_dir)
        
        # 开始完整fine-tuning
        best_loss = finetuner.finetune()
        
        print(f"\n🎯 VAE Fine-tuning完成，最佳重建损失: {best_loss:.4f}")
        print("💡 VAE已经适配微多普勒时频图域，可以用于LDM训练")
        
        return best_loss
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ 显存不足错误: {e}")
            print("🆘 启动应急低内存模式...")
            
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            try:
                # 尝试应急模式
                emergency_finetuner = create_emergency_low_memory_finetuner(data_dir)
                best_loss = emergency_finetuner.finetune()
                
                print(f"\n🎯 应急模式VAE Fine-tuning完成，最佳重建损失: {best_loss:.4f}")
                return best_loss
                
            except Exception as emergency_e:
                print(f"❌ 应急模式也失败了: {emergency_e}")
                print("💡 最终建议:")
                print("   1. 重启内核清理所有内存")
                print("   2. 手动设置batch_size=2")
                print("   3. 使用CPU训练（非常慢）")
                return None
        elif "Unsupported dtype" in str(e) or "dtype" in str(e).lower():
            print(f"❌ 数据类型错误: {e}")
            print("🔄 禁用混合精度训练重试...")
            
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            try:
                # 创建禁用混合精度的fine-tuner
                print("📦 创建FP32精度训练器...")
                
                class SafeCloudVAEFineTuner(CloudVAEFineTuner):
                    def __init__(self, data_dir):
                        # 临时禁用混合精度
                        global MIXED_PRECISION_AVAILABLE
                        original_mp = MIXED_PRECISION_AVAILABLE
                        MIXED_PRECISION_AVAILABLE = False
                        
                        super().__init__(data_dir)
                        self.config['mixed_precision'] = False
                        print("✅ FP32精度训练器创建完成")
                        
                        # 恢复原始设置
                        MIXED_PRECISION_AVAILABLE = original_mp
                
                safe_finetuner = SafeCloudVAEFineTuner(data_dir)
                best_loss = safe_finetuner.finetune()
                
                print(f"\n🎯 FP32模式VAE Fine-tuning完成，最佳重建损失: {best_loss:.4f}")
                return best_loss
                
            except Exception as safe_e:
                print(f"❌ FP32模式也失败了: {safe_e}")
                return None
        else:
            print(f"❌ 运行时错误: {e}")
        
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return None
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return None

if __name__ == "__main__":
    main() 
