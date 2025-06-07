"""
VAE Fine-tuning - 云服务器完整版本
针对Oxford-IIIT Pet数据集优化AutoencoderKL
数据集路径：/data
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL
from transformers import get_scheduler

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
from transformers import get_scheduler

# 添加混合精度训练支持
try:
    import torch
    # 检查PyTorch版本并使用正确的autocast和GradScaler
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        from torch.amp import autocast, GradScaler
        AUTOCAST_DEVICE = 'cuda'
        GRADSCALER_DEVICE = 'cuda'
        MIXED_PRECISION_AVAILABLE = True
        print("✅ 新版混合精度训练可用")
    else:
        from torch.cuda.amp import autocast, GradScaler
        AUTOCAST_DEVICE = None
        GRADSCALER_DEVICE = None
        MIXED_PRECISION_AVAILABLE = True
        print("✅ 传统混合精度训练可用")
except ImportError:
    MIXED_PRECISION_AVAILABLE = False
    print("⚠️  混合精度训练不可用")

print("DEBUG PY: Reached point 1 - After initial setup and mixed precision check.")

class OxfordPetDataset(Dataset):
    """Oxford-IIIT Pet数据集类"""
    
    def __init__(self, images_dir, annotation_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        
        # 加载标注文件
        self.samples = []
        with open(annotation_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue  # 跳过注释行
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_id = parts[0]  # 图像ID
                    class_id = int(parts[1]) - 1  # 类别ID (转为0-indexed)
                    self.samples.append((image_id, class_id))
        
        print(f"加载了 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_id, class_id = self.samples[idx]
        img_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        
        # 有些图像可能没有扩展名，尝试其他可能的扩展名
        if not os.path.exists(img_path):
            for ext in ['.jpeg', '.png', '.JPEG', '.PNG']:
                alt_path = os.path.join(self.images_dir, f"{image_id}{ext}")
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, class_id
        except Exception as e:
            print(f"错误加载图像 {img_path}: {e}")
            # 返回一个占位图像和类别
            placeholder = torch.zeros((3, 256, 256))
            return placeholder, class_id

def build_pet_dataloader(images_dir, annotations_dir, batch_size=8, num_workers=0, val_split=0.2):
    """构建Oxford Pet数据集的数据加载器"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 更新日志并确保使用 list.txt
    print(f"📁 加载Oxford-IIIT Pet数据集...")
    annotation_file = os.path.join(annotations_dir, "list.txt")
    
    dataset = OxfordPetDataset(
        images_dir=images_dir,
        annotation_file=annotation_file,
        transform=transform
    )
    
    # 分割数据集
    train_size = int(len(dataset) * (1 - val_split))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"  训练集: {len(train_dataset)} 张图片")
    print(f"  验证集: {len(val_dataset)} 张图片")
    
    return train_loader, val_loader, len(train_dataset), len(val_dataset)

class VAEFineTuner:
    """优化的VAE微调器，专为P100 GPU设计"""
    def __init__(self, images_dir, annotations_dir, output_dir='/kaggle/working/vae_finetuned'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 VAE Fine-tuning 设备: {self.device}")
        
        # 针对P100 GPU优化的配置
        self.config = {
            'batch_size': 12,
            'gradient_accumulation_steps': 2,
            'learning_rate': 1e-5,
            'weight_decay': 0.01,
            'max_epochs': 16,
            'images_dir': images_dir,
            'annotations_dir': annotations_dir,
            'save_dir': output_dir,
            'reconstruction_weight': 1.0,
            'kl_weight': 1e-7,  # 降低KL权重，更注重重建质量
            'perceptual_weight': 0.1,  # 降低感知损失权重
            'use_perceptual_loss': True,
            'save_every_epochs': 2,
            'eval_every_epochs': 2,
            'use_gradient_checkpointing': True,
            'mixed_precision': MIXED_PRECISION_AVAILABLE,
            'image_size': 256,
        }
        self.best_model_checkpoint_path = None
        
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # 数据加载器
        self.train_loader, self.val_loader, train_size, val_size = build_pet_dataloader(
            images_dir=self.config['images_dir'],
            annotations_dir=self.config['annotations_dir'],
            batch_size=self.config['batch_size'],
            num_workers=2,
            val_split=0.2
        )
        
        print(f"  有效batch size: {self.config['batch_size'] * self.config['gradient_accumulation_steps']}")
        print(f"  图像尺寸: {self.config['image_size']}x{self.config['image_size']}")
        
        # 加载VAE
        print("📦 加载预训练AutoencoderKL...")
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="vae"
        ).to(self.device).float()
        
        # 启用梯度检查点
        if hasattr(self.vae, 'enable_gradient_checkpointing'):
            self.vae.enable_gradient_checkpointing()
            print("✅ VAE梯度检查点已启用")
        
        # 解冻参数
        for param in self.vae.parameters():
            param.requires_grad = True
        
        self.setup_optimizer()
        self.mse_loss = nn.MSELoss()
        self.setup_perceptual_loss()
        
        if self.config['mixed_precision']:
            self.scaler = GradScaler()
            print("✅ 混合精度训练已启用")
        
        # 训练历史记录
        self.train_history = {
            'epoch': [], 'train_loss': [], 'recon_loss': [],
            'kl_loss': [], 'perceptual_loss': [], 'val_mse': [], 'val_epoch': []
        }
        
        self.clear_memory()
        print("✅ VAE微调器初始化完成!")
    
    def setup_optimizer(self):
        """设置分层优化器，并使用带预热的学习率调度器"""
        finetune_params = list(self.vae.decoder.parameters()) + \
                        list(self.vae.encoder.down_blocks[-2].parameters()) + \
                        list(self.vae.encoder.down_blocks[-1].parameters()) + \
                        list(self.vae.quant_conv.parameters()) + \
                        list(self.vae.post_quant_conv.parameters())
        
        total_params = sum(p.numel() for p in finetune_params)
        print(f"🎯 Fine-tune参数数量: {total_params:,}")
        
        self.optimizer = torch.optim.AdamW(
            finetune_params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 计算总训练步数
        num_update_steps_per_epoch = len(self.train_loader) // self.config['gradient_accumulation_steps']
        num_training_steps = self.config['max_epochs'] * num_update_steps_per_epoch
        # 设置预热步数，总步数的10%
        num_warmup_steps = int(num_training_steps * 0.1)

        print(f"🔄 优化器设置 - 使用带预热的调度器:")
        print(f"   总训练步数: {num_training_steps}")
        print(f"   预热步数: {num_warmup_steps} (占总步数的10%)")
        print(f"   权重配置: KL损失={self.config['kl_weight']}, 感知损失={self.config['perceptual_weight']}")

        # 使用 transformers 提供的带预热的调度器
        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    def setup_perceptual_loss(self):
        """设置VGG感知损失"""
        if not self.config['use_perceptual_loss']:
            print("🚫 感知损失已关闭以节省内存")
            self.perceptual_net = None
            return
            
        try:
            from torchvision.models import vgg16, VGG16_Weights
            self.perceptual_net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(self.device).float()
            for param in self.perceptual_net.parameters():
                param.requires_grad = False
            self.perceptual_net.eval()
            print("✅ VGG16感知损失已设置 (使用前16层)")
        except:
            print("⚠️  VGG16不可用，跳过感知损失")
            self.perceptual_net = None
    
    def compute_perceptual_loss(self, real, fake):
        """计算感知损失"""
        if self.perceptual_net is None:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        self.perceptual_net.float()
        real = real.float()
        fake = fake.float()
        
        if real.size(1) == 1:
            real = real.repeat(1, 3, 1, 1)
        if fake.size(1) == 1:
            fake = fake.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            real_features = self.perceptual_net(real)
        
        fake_features = self.perceptual_net(fake)
        return F.mse_loss(real_features.float(), fake_features.float())
    
    def vae_loss(self, images):
        """VAE损失函数计算"""
        try:
            images = images.float()
            posterior = self.vae.encode(images).latent_dist
            latents = posterior.sample().float()
            reconstructed = self.vae.decode(latents).sample.float()

            # 重建损失
            recon_loss = self.mse_loss(reconstructed, images)
            
            # KL散度损失
            try:
                kl_raw = posterior.kl()
                if torch.isnan(kl_raw).any() or torch.isinf(kl_raw).any():
                    print("⚠️ 检测到KL散度中的NaN/Inf值，使用替代计算方法")
                    mean, var = posterior.mean, posterior.var
                    eps = 1e-8
                    kl_loss = 0.5 * torch.mean(mean.pow(2) + var - torch.log(var + eps) - 1)
                else:
                    kl_loss = kl_raw.mean()
                if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                    print("⚠️ KL损失仍然包含NaN/Inf，使用常数替代")
                    kl_loss = torch.tensor(0.1, device=self.device, dtype=torch.float32)
            except Exception as kl_e:
                print(f"⚠️ KL散度计算错误: {kl_e}，使用替代值")
                kl_loss = torch.tensor(0.1, device=self.device, dtype=torch.float32)
            
            # 感知损失
            if self.perceptual_net:
                perceptual_loss = self.compute_perceptual_loss(images, reconstructed)
            else:
                perceptual_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            
            # 总损失
            total_loss = (
                self.config['reconstruction_weight'] * recon_loss +
                self.config['kl_weight'] * kl_loss +
                self.config['perceptual_weight'] * perceptual_loss
            )
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("⚠️ 总损失包含NaN/Inf，仅使用重建损失")
                total_loss = self.config['reconstruction_weight'] * recon_loss
                
            return {
                'total_loss': total_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'perceptual_loss': perceptual_loss,
                'reconstructed': reconstructed,
                'latents': latents
            }
        except Exception as e:
            print(f"🔥 VAE_LOSS内部发生错误，类型: {type(e)}, 内容: {e} 🔥")
            import traceback
            traceback.print_exc()
            raise e
    
    def clear_memory(self):
        """清理内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def train_epoch(self, epoch: int):
        """优化的训练epoch (使用更好的学习率调度策略)"""
        self.vae.train()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_perceptual = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Fine-tune Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            images, _ = batch
            images = images.to(self.device, non_blocking=True)
            
            if self.config['mixed_precision']:
                if AUTOCAST_DEVICE:
                    with autocast(AUTOCAST_DEVICE, dtype=torch.float16):
                        loss_dict = self.vae_loss(images)
                        loss = loss_dict['total_loss'] / self.config['gradient_accumulation_steps']
                else:
                    with autocast():
                        loss_dict = self.vae_loss(images)
                        loss = loss_dict['total_loss'] / self.config['gradient_accumulation_steps']
                
                self.scaler.scale(loss.float()).backward()
            else:
                loss_dict = self.vae_loss(images)
                loss = loss_dict['total_loss'] / self.config['gradient_accumulation_steps']
                loss.backward()
            
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                try:
                    if self.config['mixed_precision']:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                        self.optimizer.step()
                    
                    self.scheduler.step() # 每步更新学习率调度器
                    self.optimizer.zero_grad()
                    
                except RuntimeError as grad_e:
                    if "dtype" in str(grad_e).lower() or "type" in str(grad_e).lower():
                        print(f"⚠️ 梯度步骤数据类型错误，跳过此步骤: {grad_e}")
                        self.optimizer.zero_grad()
                        continue
                    else:
                        raise grad_e
            
            actual_loss = loss.item() * self.config['gradient_accumulation_steps']
            total_loss += actual_loss
            total_recon += loss_dict['recon_loss'].item()
            total_kl += loss_dict['kl_loss'].item()
            total_perceptual += loss_dict['perceptual_loss'].item()
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{actual_loss:.4f}',
                'Recon': f'{loss_dict["recon_loss"].item():.4f}',
                'KL': f'{loss_dict["kl_loss"].item():.4f}',
                'Perc': f'{loss_dict["perceptual_loss"].item():.4f}'
            })
            
            if batch_idx == 0:
                self.save_reconstruction_samples(
                    images, loss_dict['reconstructed'], epoch
                )
            
            if batch_idx % 20 == 0:
                del images, loss_dict, loss
                self.clear_memory()
        
        if (len(self.train_loader)) % self.config['gradient_accumulation_steps'] != 0:
            if self.config['mixed_precision']:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            self.scheduler.step() # 确保最后一次更新学习率
            self.optimizer.zero_grad()
        
        avg_metrics = {
            'avg_loss': total_loss / num_batches,
            'avg_recon': total_recon / num_batches,
            'avg_kl': total_kl / num_batches,
            'avg_perceptual': total_perceptual / num_batches
        }
        
        self.clear_memory()
        return avg_metrics
    
    def save_reconstruction_samples(self, original, reconstructed, epoch):
        """保存重建样本"""
        try:
            def denormalize(tensor):
                tensor = tensor.clone().detach().cpu()
                tensor = tensor * 0.5 + 0.5  # 反归一化
                tensor = tensor.clamp(0, 1)
                return tensor
            
            with torch.no_grad():
                # 选择最多8张图像用于可视化
                num_images = min(8, original.size(0))
                original = denormalize(original[:num_images])
                reconstructed = denormalize(reconstructed[:num_images])
                
                # 创建网格
                grid = torch.zeros((3, original.size(2) * 2, original.size(3) * num_images))
                for i in range(num_images):
                    grid[:, :original.size(2), i*original.size(3):(i+1)*original.size(3)] = original[i]
                    grid[:, original.size(2):, i*original.size(3):(i+1)*original.size(3)] = reconstructed[i]
                
                # 保存图像
                from torchvision.utils import save_image
                save_image(grid, f"{self.config['save_dir']}/recon_epoch_{epoch+1}.png")
                print(f"✅ 保存了重建样本 epoch {epoch+1}")
        except Exception as e:
            print(f"⚠️ 保存重建样本失败: {e}")
    
    @torch.no_grad()
    def evaluate_reconstruction_quality(self, val_loader):
        """评估重建质量"""
        self.vae.eval()
        total_mse = 0
        total_samples = 0
        latent_means = []
        latent_stds = []
        
        for batch in tqdm(val_loader, desc="评估重建质量"):
            images, _ = batch
            images = images.to(self.device)
            
            try:
                posterior = self.vae.encode(images).latent_dist
                latents = posterior.sample()
                reconstructions = self.vae.decode(latents).sample
            
                # 计算MSE
                mse = F.mse_loss(reconstructions, images, reduction='none').mean([1, 2, 3])
                total_mse += mse.sum().item()
                total_samples += images.size(0)
            
                # 收集潜在空间统计信息
                latent_means.append(posterior.mean.cpu().flatten(1).mean(0).numpy())
                latent_stds.append(posterior.var.sqrt().cpu().flatten(1).mean(0).numpy())
            except RuntimeError as e:
                print(f"⚠️ 评估中的CUDA错误: {e}")
                self.clear_memory()
        
        avg_mse = total_mse / total_samples if total_samples > 0 else float('inf')
        
        # 汇总潜在空间统计信息
        latent_stats = {
            'mean': np.mean(latent_means, axis=0) if latent_means else None,
            'std': np.mean(latent_stds, axis=0) if latent_stds else None
        }
        
        print(f"📊 验证集平均MSE: {avg_mse:.6f}")
        self.clear_memory()
        
        return avg_mse, latent_stats
    
    def save_finetuned_vae(self, epoch):
        """以diffusers兼容的格式保存微调后的VAE模型"""
        try:
            # 模型将保存到一个目录中, e.g., /.../vae_finetuned_epoch_3/
            model_path = os.path.join(self.config['save_dir'], f"vae_finetuned_epoch_{epoch+1}")
            self.vae.save_pretrained(model_path)
            print(f"💾 VAE模型已保存到目录: {model_path}")
            return model_path
        except Exception as e:
            print(f"❌ 保存模型失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(14, 10))
        axes = plt.subplot(2, 2, 1)
        plt.plot(self.train_history['epoch'], self.train_history['train_loss'])
        plt.title('总损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(self.train_history['epoch'], self.train_history['kl_loss'])
        plt.title('KL散度损失')
        plt.xlabel('Epoch')
        plt.ylabel('KL Loss')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(self.train_history['epoch'], self.train_history['perceptual_loss'])
        plt.title('感知损失')
        plt.xlabel('Epoch')
        plt.ylabel('Perceptual Loss')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        if self.train_history['val_epoch']:
            plt.plot(self.train_history['val_epoch'], self.train_history['val_mse'])
            plt.title('验证集MSE')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.config["save_dir"]}/training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def finetune(self):
        """优化版本的fine-tuning流程，添加更好的学习率调度策略"""
        print("🚀 开始优化版VAE Fine-tuning训练...")
        print("=" * 60)
        
        if torch.cuda.is_available():
            torch.autograd.set_detect_anomaly(True)
            print("⚠️ PyTorch 异常检测已启用 (用于调试，可能影响速度)")
        
        best_recon_loss = float('inf')
        
        for epoch in range(self.config['max_epochs']):
            print(f"\n📅 Epoch {epoch+1}/{self.config['max_epochs']}")
            
            # 调用新的train_epoch方法
            train_metrics = self.train_epoch(epoch)
            
            # 记录训练历史
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_metrics['avg_loss'])
            self.train_history['recon_loss'].append(train_metrics['avg_recon'])
            self.train_history['kl_loss'].append(train_metrics['avg_kl'])
            self.train_history['perceptual_loss'].append(train_metrics['avg_perceptual'])
            
            # 进行验证
            if epoch % self.config['eval_every_epochs'] == 0 or epoch == self.config['max_epochs'] - 1:
                val_mse, latent_stats = self.evaluate_reconstruction_quality(self.val_loader)
                self.train_history['val_mse'].append(val_mse)
                self.train_history['val_epoch'].append(epoch + 1)
            
            # 学习率是每步更新的，这里获取最新的学习率值
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"📊 Epoch {epoch+1} 结果:")
            print(f"   总损失: {train_metrics['avg_loss']:.4f}")
            print(f"   重建损失: {train_metrics['avg_recon']:.4f}")
            print(f"   KL损失: {train_metrics['avg_kl']:.4f}")
            print(f"   感知损失: {train_metrics['avg_perceptual']:.4f}")
            print(f"   当前学习率: {current_lr:.2e}")
            
            # 保存最佳模型逻辑
            is_best_model_save = False
            if train_metrics['avg_recon'] < best_recon_loss:
                best_recon_loss = train_metrics['avg_recon']
                
                # 尝试删除上一个被标记为"最佳"的模型目录
                if self.best_model_checkpoint_path and os.path.isdir(self.best_model_checkpoint_path):
                    try:
                        import shutil
                        # 从目录名提取旧的epoch号
                        old_epoch_str = self.best_model_checkpoint_path.split('_epoch_')[-1]
                        old_epoch_idx = int(old_epoch_str) - 1
                        
                        is_periodic_save = (old_epoch_idx % self.config['save_every_epochs'] == 0)
                        
                        if not is_periodic_save:
                            print(f"🗑️ 删除旧的最佳模型目录: {self.best_model_checkpoint_path}")
                            shutil.rmtree(self.best_model_checkpoint_path)
                        else:
                            print(f"ℹ️ 保留旧的最佳模型 (同时也是定期保存点): {self.best_model_checkpoint_path}")
                    except Exception as e_remove:
                        print(f"⚠️ 无法删除或检查旧的最佳模型目录: {e_remove}")

                saved_path = self.save_finetuned_vae(epoch)  # 保存当前模型
                if saved_path:
                    self.best_model_checkpoint_path = saved_path  # 更新最佳模型路径
                    print(f"✅ 保存新的最佳模型 (重建损失: {best_recon_loss:.4f}): {self.best_model_checkpoint_path}")
                is_best_model_save = True
            
            if epoch % self.config['save_every_epochs'] == 0:
                if not is_best_model_save:  # 如果此epoch已作为最佳模型保存，则不再重复保存
                    self.save_finetuned_vae(epoch)
                self.plot_training_history()
            
            print(f"🏆 当前最佳重建损失: {best_recon_loss:.4f}")
            self.clear_memory()
        
        # 训练完成
        print("\n" + "=" * 60)
        print("🎉 优化版VAE Fine-tuning训练完成!")
        print(f"🏆 最佳重建损失: {best_recon_loss:.4f}")
        
        # 最终评估
        final_mse, final_latent_stats = self.evaluate_reconstruction_quality(self.val_loader)
        print(f"🎯 最终重建MSE: {final_mse:.6f}")
        
        # 绘制最终训练历史
        self.plot_training_history()
        
        return best_recon_loss

# 主函数
def run_vae_training(images_dir=None, annotations_dir=None, output_dir=None):
    """运行VAE微调训练"""
    print("🚀 开始VAE微调训练 (针对P100 GPU优化)...")

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.cuda.empty_cache()
        print(f"🔧 CUDA内存优化设置完成")

    # 默认路径设置
    if images_dir is None:
        images_dir = '/kaggle/input/dataset-test/images/images'
    if annotations_dir is None:
        annotations_dir = '/kaggle/input/dataset-test/annotations/annotations'
    if output_dir is None:
        output_dir = '/kaggle/working/vae_finetuned'

    print(f"📊 数据集路径配置:")
    print(f"  图像目录: {images_dir}")
    print(f"  标注目录: {annotations_dir}")
    print(f"  输出目录: {output_dir}")

    try:
        print("💡 使用P100 GPU优化配置...")
        trainer = VAEFineTuner(images_dir, annotations_dir, output_dir)
        best_loss = trainer.finetune()
        print(f"\n🎯 VAE微调完成，最佳重建损失: {best_loss:.4f}")
        return best_loss
    except Exception as e:
        print("🔥 训练过程中发生错误:")
        import traceback
        traceback.print_exc()
        return None

# 如果直接运行此脚本
if __name__ == "__main__":
    print("VAE微调脚本已加载，开始执行训练...")
    
    # Kaggle环境默认路径
    images_dir = '/kaggle/input/dataset-test/images/images'
    annotations_dir = '/kaggle/input/dataset-test/annotations/annotations'
    output_dir = '/kaggle/working/vae_finetuned'
    
    # 自动开始训练
    run_vae_training(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        output_dir=output_dir
    )
