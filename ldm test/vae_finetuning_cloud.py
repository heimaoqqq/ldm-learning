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

class OxfordPetDataset(Dataset):
    """Oxford-IIIT Pet数据集类"""
    
    def __init__(self, data_root, annotation_file, transform=None):
        self.data_root = data_root
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
        img_path = os.path.join(self.data_root, "images", f"{image_id}.jpg")
        
        # 有些图像可能没有扩展名，尝试其他可能的扩展名
        if not os.path.exists(img_path):
            for ext in ['.jpeg', '.png', '.JPEG', '.PNG']:
                alt_path = os.path.join(self.data_root, "images", f"{image_id}{ext}")
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

def build_pet_dataloader(root_dir, batch_size=8, num_workers=0, val_split=0.2):
    """构建Oxford-IIIT Pet数据集的加载器"""
    print(f"🔍 加载Oxford-IIIT Pet数据集: {root_dir}")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 检查数据集目录结构
    train_annotation_file = os.path.join(root_dir, "annotations", "trainval.txt")
    test_annotation_file = os.path.join(root_dir, "annotations", "test.txt")
    
    if not os.path.exists(root_dir):
        raise ValueError(f"❌ 数据集目录不存在: {root_dir}")
    
    if not os.path.exists(train_annotation_file) or not os.path.exists(test_annotation_file):
        raise ValueError(f"❌ 标注文件不存在")
    
    # 加载训练和验证集
    train_val_dataset = OxfordPetDataset(
        data_root=root_dir,
        annotation_file=train_annotation_file,
        transform=transform
    )
    
    # 分割训练集和验证集
    train_size = int(len(train_val_dataset) * (1 - val_split))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"📊 数据集统计:")
    print(f"  训练集: {len(train_dataset)} 张图片")
    print(f"  验证集: {len(val_dataset)} 张图片")
    
    return train_loader, val_loader, len(train_dataset), len(val_dataset)

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
    
    def __init__(self, data_dir='./data'):
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
            'kl_weight': 1e-5,  # 大幅降低KL权重，更关注重建质量
            'perceptual_weight': 0.3,  # 恢复合理的感知损失权重
            'use_perceptual_loss': True,  # 保持感知损失开启
            'save_every_epochs': 5,  # 每5个epoch保存一次
            'eval_every_epochs': 2,  # 每2个epoch评估一次
            'use_gradient_checkpointing': True,  # 启用梯度检查点
            'mixed_precision': MIXED_PRECISION_AVAILABLE,  # 混合精度训练
            'safe_mixed_precision': True,  # 安全模式：遇到问题时自动禁用混合精度
        }
        self.best_model_checkpoint_path = None # 用于跟踪要删除的旧最佳模型
        
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # 1. 创建数据加载器 - 减少num_workers
        print("📁 加载Oxford-IIIT Pet数据集...")
        self.train_loader, self.val_loader, train_size, val_size = build_pet_dataloader(
            root_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            num_workers=0,  # 减少workers以节省内存
            val_split=0.2
        )
        print(f"   训练集: {len(self.train_loader)} batches ({train_size} samples)")
        print(f"   验证集: {len(self.val_loader)} batches ({val_size} samples)")
        print(f"   有效batch size: {self.config['batch_size'] * self.config['gradient_accumulation_steps']}")
        
        # 2. 加载预训练VAE
        print("📦 加载预训练AutoencoderKL...")
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="vae"
        ).to(self.device).float()
        
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
            if GRADSCALER_DEVICE:
                self.scaler = GradScaler(GRADSCALER_DEVICE)
            else:
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
            self.perceptual_net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(self.device).float()
            for param in self.perceptual_net.parameters():
                param.requires_grad = False
            self.perceptual_net.eval()
            print("✅ 完整VGG16感知损失已设置 (FP32)")
        except:
            print("⚠️  VGG16不可用，跳过感知损失")
            self.perceptual_net = None
    
    def compute_perceptual_loss(self, real, fake):
        """计算完整感知损失"""
        if self.perceptual_net is None:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        # Ensure network and inputs are float32
        self.perceptual_net.float()
        real = real.float()
        fake = fake.float()
        
        # 确保输入是3通道
        if real.size(1) == 1:
            real = real.repeat(1, 3, 1, 1)
        if fake.size(1) == 1:
            fake = fake.repeat(1, 3, 1, 1)
        
        # 计算VGG特征，完全避免autocast的影响
        with torch.no_grad():
            real_features = self.perceptual_net(real)
        
        # 确保fake_features计算也使用正确的数据类型
        fake_features = self.perceptual_net(fake)
        
        # Ensure MSE损失计算使用相同的数据类型
        return F.mse_loss(real_features.float(), fake_features.float())
    
    def vae_loss(self, images):
        """VAE损失函数 - VAE操作强制FP32，其余依赖外部autocast"""
        # images 输入时，在混合精度模式下已在 train_epoch 中被转换为 .float(), 
        # 并且此函数在顶层 autocast(dtype=torch.float16) 上下文中被调用。

        try:
            # 1. VAE操作：强制在FP32下执行，局部覆盖外部autocast
            if AUTOCAST_DEVICE: # 新版PyTorch
                with autocast(AUTOCAST_DEVICE, enabled=False, dtype=torch.float32):
                    posterior = self.vae.encode(images.float()).latent_dist
                    latents = posterior.sample().float()
                    reconstructed = self.vae.decode(latents).sample.float()
            else: # 传统版本 PyTorch (autocast 指 torch.cuda.amp.autocast)
                with autocast(enabled=False):
                    posterior = self.vae.encode(images.float()).latent_dist
                    latents = posterior.sample().float()
                    reconstructed = self.vae.decode(latents).sample.float()

            # 2. 重建损失：输入确保是FP32
            recon_loss = self.mse_loss(reconstructed.float(), images.float()) # images 已是 .float()
            
            # 3. KL散度损失：结果转换为FP32
            kl_loss = posterior.kl().mean().float()
            
            # 4. 感知损失：输入确保是FP32。compute_perceptual_loss内部已处理好网络精度。
            if self.perceptual_net:
                perceptual_loss = self.compute_perceptual_loss(images.float(), reconstructed.float())
            else:
                perceptual_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            
            # 5. 总损失：所有组件都应是FP32
            total_loss = (
                self.config['reconstruction_weight'] * recon_loss +
                self.config['kl_weight'] * kl_loss +
                self.config['perceptual_weight'] * perceptual_loss
            )
            
            # 6. 返回值：确保所有返回的张量是FP32
            return {
                'total_loss': total_loss.float(),
                'recon_loss': recon_loss.float(),
                'kl_loss': kl_loss.float(),
                'perceptual_loss': perceptual_loss.float(),
                'reconstructed': reconstructed.float(),
                'latents': latents.float()
            }
            
        except Exception as e:
            error_str = str(e).lower()
            if "dtype" in error_str or "type" in error_str or "expected scalar type" in error_str:
                print(f"⚠️ VAE损失计算中的数据类型错误: {e}")
                return {
                    'total_loss': torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=True),
                    'recon_loss': torch.tensor(0.0, device=self.device, dtype=torch.float32),
                    'kl_loss': torch.tensor(0.0, device=self.device, dtype=torch.float32),
                    'perceptual_loss': torch.tensor(0.0, device=self.device, dtype=torch.float32),
                    'reconstructed': images.float(),
                    'latents': torch.zeros_like(images, dtype=torch.float32)
                }
            else:
                print(f"🔥 VAE_LOSS内部发生意外错误，类型: {type(e)}, 内容: {e} 🔥")
                import traceback
                traceback.print_exc()
                raise e
    
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
                    with autocast(AUTOCAST_DEVICE, dtype=torch.float16):
                        loss_dict = self.vae_loss(images)
                        loss = loss_dict['total_loss'] / self.config['gradient_accumulation_steps']
                else:
                    with autocast(): # Old API for mixed precision context
                        loss_dict = self.vae_loss(images)
                        loss = loss_dict['total_loss'] / self.config['gradient_accumulation_steps']
                
                # 确保损失是 float32 类型再进行缩放和反向传播
                self.scaler.scale(loss.float()).backward()
            else:
                # 前向传播 (无混合精度)
                loss_dict = self.vae_loss(images)
                loss = loss_dict['total_loss'] / self.config['gradient_accumulation_steps']
                loss.backward() # 直接反向传播
            
            # 梯度累积
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                try:
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
                    
                except RuntimeError as grad_e:
                    if "dtype" in str(grad_e).lower() or "type" in str(grad_e).lower():
                        print(f"⚠️ 梯度步骤数据类型错误，跳过此步骤: {grad_e}")
                        self.optimizer.zero_grad()
                        continue
                    else:
                        raise grad_e
            
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
            
            batch_size = original.size(0)  # 获取实际的batch大小
            
            # 确保不超过batch size的范围
            orig = denormalize(original[:batch_size]).cpu()
            recon = denormalize(reconstructed[:batch_size]).cpu()
            
            # 动态调整图表大小
            fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 3, 6))
            
            # 处理单样本情况（batch_size=1）
            if batch_size == 1:
                # 原图
                axes[0].imshow(orig[0].permute(1, 2, 0))
                axes[0].set_title(f'Original')
                axes[0].axis('off')
                
                # 重建图
                axes[1].imshow(recon[0].permute(1, 2, 0))
                axes[1].set_title(f'Reconstructed')
                axes[1].axis('off')
            else:
                # 多样本情况
                for i in range(batch_size):
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
        
        if torch.cuda.is_available(): # 确保只在CUDA可用时设置
            torch.autograd.set_detect_anomaly(True)
            print("⚠️ PyTorch 异常检测已启用 (用于调试，可能影响速度)")
        
        best_recon_loss = float('inf')
        
        for epoch in range(self.config['max_epochs']):
            print(f"\n📅 Epoch {epoch+1}/{self.config['max_epochs']}")
            
            train_metrics = self.train_epoch(epoch)
            
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_metrics['avg_loss'])
            self.train_history['recon_loss'].append(train_metrics['avg_recon'])
            self.train_history['kl_loss'].append(train_metrics['avg_kl'])
            self.train_history['perceptual_loss'].append(train_metrics['avg_perceptual'])
            
            if epoch % self.config['eval_every_epochs'] == 0 or epoch == self.config['max_epochs'] - 1:
                val_mse, latent_stats = self.evaluate_reconstruction_quality(self.val_loader)
                self.train_history['val_mse'].append(val_mse)
            
            print(f"📊 Epoch {epoch+1} 结果:")
            print(f"   总损失: {train_metrics['avg_loss']:.4f}")
            print(f"   重建损失: {train_metrics['avg_recon']:.4f}")
            print(f"   KL损失: {train_metrics['avg_kl']:.4f}")
            print(f"   感知损失: {train_metrics['avg_perceptual']:.4f}")
            print(f"   学习率: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # 保存最佳模型逻辑修改
            current_epoch_model_path = f'{self.config["save_dir"]}/vae_finetuned_epoch_{epoch+1}.pth'
            is_best_model_save = False
            if train_metrics['avg_recon'] < best_recon_loss:
                best_recon_loss = train_metrics['avg_recon']
                
                # 尝试删除上一个被标记为"最佳"的模型文件
                if self.best_model_checkpoint_path and os.path.exists(self.best_model_checkpoint_path):
                    # 检查这个旧的最佳模型是否也是一个定期保存点
                    try:
                        # 从文件名提取旧的epoch号
                        old_epoch_str = self.best_model_checkpoint_path.split('_epoch_')[-1].split('.pth')[0]
                        old_epoch_idx = int(old_epoch_str) - 1 # epoch号转为0-indexed
                        is_periodic_save = (old_epoch_idx % self.config['save_every_epochs'] == 0)
                        
                        if not is_periodic_save:
                            print(f"🗑️ 删除旧的最佳模型: {self.best_model_checkpoint_path}")
                            os.remove(self.best_model_checkpoint_path)
                        else:
                            print(f"ℹ️ 保留旧的最佳模型 (同时也是定期保存点): {self.best_model_checkpoint_path}")
                    except Exception as e_remove:
                        print(f"⚠️ 无法删除或检查旧的最佳模型: {e_remove}")

                self.save_finetuned_vae(epoch, train_metrics) # 保存当前模型
                self.best_model_checkpoint_path = current_epoch_model_path # 更新最佳模型路径
                print(f"✅ 保存新的最佳模型 (重建损失: {best_recon_loss:.4f}): {current_epoch_model_path}")
                is_best_model_save = True
            
            if epoch % self.config['save_every_epochs'] == 0:
                if not is_best_model_save: # 如果此epoch已作为最佳模型保存，则不再重复保存
                    self.save_finetuned_vae(epoch, train_metrics)
                self.plot_training_history()
            
            print(f"🏆 当前最佳重建损失: {best_recon_loss:.4f}")
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

class PureFP32CloudVAEFineTuner(CloudVAEFineTuner):
    """纯FP32精度训练器，完全避免混合精度问题"""
    
    def __init__(self, data_dir):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 纯FP32模式 - 云VAE Fine-tuning 设备: {self.device}")
        
        # 纯FP32模式配置
        self.config = {
            'batch_size': 6,  # 稍微减小batch size
            'gradient_accumulation_steps': 3,  # 增加梯度累积
            'learning_rate': 1e-5,
            'weight_decay': 0.01,
            'max_epochs': 20,
            'data_dir': data_dir,
            'save_dir': './vae_finetuned',
            'reconstruction_weight': 1.0,
            'kl_weight': 1e-5,
            'perceptual_weight': 0.3,
            'use_perceptual_loss': True,
            'save_every_epochs': 5,
            'eval_every_epochs': 2,
            'use_gradient_checkpointing': True,
            'mixed_precision': False,  # 强制禁用混合精度
            'safe_mixed_precision': False,
        }
        self.best_model_checkpoint_path = None
        
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # 创建数据加载器
        print("📁 加载Oxford-IIIT Pet数据集 (纯FP32模式)...")
        self.train_loader, self.val_loader, train_size, val_size = build_pet_dataloader(
            root_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            num_workers=0,
            val_split=0.2
        )
        print(f"   训练集: {len(self.train_loader)} batches ({train_size} samples)")
        print(f"   验证集: {len(self.val_loader)} batches ({val_size} samples)")
        
        # 加载预训练VAE (纯FP32模式)
        print("📦 加载预训练AutoencoderKL (纯FP32模式)...")
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="vae"
        ).to(self.device).float()
        
        # 启用梯度检查点
        if hasattr(self.vae, 'enable_gradient_checkpointing'):
            self.vae.enable_gradient_checkpointing()
            print("✅ VAE梯度检查点已启用")
        
        for param in self.vae.parameters():
            param.requires_grad = True
        
        # 配置优化器
        self.setup_optimizer()
        self.mse_loss = nn.MSELoss()
        self.setup_perceptual_loss()
        
        # 记录训练历史
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'perceptual_loss': [],
            'val_mse': []
        }
        
        self.clear_memory()
        print("✅ 纯FP32模式Fine-tuning器初始化完成!")
    
    # 覆盖vae_loss函数，确保完全使用FP32
    def vae_loss(self, images):
        """纯FP32 VAE损失函数"""
        # 确保输入为float32
        images = images.float()
        
        # 前向传播 (纯FP32)
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample().float()
        reconstructed = self.vae.decode(latents).sample.float()
        
        # 损失计算 (纯FP32)
        recon_loss = self.mse_loss(reconstructed, images)
        
        # 更稳定的KL散度计算，防止NaN
        try:
            # 原始KL散度计算可能导致数值不稳定
            kl_raw = posterior.kl()
            
            # 检查并处理NaN或无穷大值
            if torch.isnan(kl_raw).any() or torch.isinf(kl_raw).any():
                print("⚠️ 检测到KL散度中的NaN/Inf值，使用替代计算方法")
                # 替代计算方法 - 基于均值和方差的直接计算
                mean, var = posterior.mean, posterior.var
                # 添加小的epsilon防止log(0)
                eps = 1e-8
                # 标准KL散度公式: KL = 0.5 * (μ² + σ² - log(σ²) - 1)
                kl_loss = 0.5 * torch.mean(mean.pow(2) + var - torch.log(var + eps) - 1)
            else:
                kl_loss = kl_raw.mean()
                
            # 额外的安全检查
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                print("⚠️ KL损失仍然包含NaN/Inf，使用常数替代")
                kl_loss = torch.tensor(0.1, device=self.device, dtype=torch.float32)
        except Exception as kl_e:
            print(f"⚠️ KL散度计算错误: {kl_e}，使用替代值")
            kl_loss = torch.tensor(0.1, device=self.device, dtype=torch.float32)
        
        # 不使用感知损失
        perceptual_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        # 计算总损失
        total_loss = (
            self.config['reconstruction_weight'] * recon_loss +
            self.config['kl_weight'] * kl_loss
        )
        
        # 最终安全检查
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

def create_ultra_low_memory_finetuner(data_dir, images_dir=None, annotations_dir=None, use_cpu=False):
    """创建超低内存配置的Fine-tuner，适配Kaggle环境"""
    print("🆘 启动超低内存模式..." + ("(CPU训练)" if use_cpu else ""))
    
    # 如果没有提供images_dir和annotations_dir，则使用默认路径
    if images_dir is None:
        # 本地环境使用data_dir内部的images目录
        images_dir = os.path.join(data_dir, "images")
    if annotations_dir is None:
        # 本地环境使用data_dir内部的annotations目录
        annotations_dir = os.path.join(data_dir, "annotations")
    
    class UltraLowMemoryDataset(Dataset):
        """超低内存数据集类，适配Kaggle环境"""
        
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
                placeholder = torch.zeros((3, 128, 128))
                return placeholder, class_id
    
    class UltraLowMemoryFineTuner(CloudVAEFineTuner):
        def __init__(self, data_dir, images_dir, annotations_dir, use_cpu=False):
            self.device = 'cpu' if use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"🚀 超低内存模式 - VAE Fine-tuning 设备: {self.device}")
            
            # 超低内存配置
            self.config = {
                'batch_size': 4 if self.device == 'cuda' else 2,  # GPU使用更大batch，CPU更小
                'gradient_accumulation_steps': 4,
                'learning_rate': 1e-5,
                'weight_decay': 0.01,
                'max_epochs': 15 if self.device == 'cuda' else 5,  # CPU模式减少训练轮次
                'data_dir': data_dir,
                'images_dir': images_dir,
                'annotations_dir': annotations_dir,
                'save_dir': '/kaggle/working/vae_finetuned' if os.path.exists('/kaggle') else './vae_finetuned',
                'reconstruction_weight': 1.0,
                'kl_weight': 0.05,
                'perceptual_weight': 0.0,  # 禁用感知损失以节省内存
                'use_perceptual_loss': False,
                'save_every_epochs': 5,
                'eval_every_epochs': 5,
                'use_gradient_checkpointing': self.device == 'cuda',  # CPU模式不启用
                'mixed_precision': False if self.device == 'cpu' else MIXED_PRECISION_AVAILABLE,
                'image_size': 128,  # 降低图像尺寸
            }
            
            os.makedirs(self.config['save_dir'], exist_ok=True)
            
            # 适应更小图像尺寸的数据转换
            transform = transforms.Compose([
                transforms.Resize((self.config['image_size'], self.config['image_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            # 加载训练和验证集
            print("📁 加载超低内存模式数据集...")
            train_annotation_file = os.path.join(annotations_dir, "trainval.txt")
            train_val_dataset = UltraLowMemoryDataset(
                images_dir=images_dir,
                annotation_file=train_annotation_file,
                transform=transform
            )
            
            # 分割训练集和验证集
            train_size = int(len(train_val_dataset) * 0.8)  # 使用80%作为训练集
            val_size = len(train_val_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_val_dataset, [train_size, val_size], 
                generator=torch.Generator().manual_seed(42)
            )
            
            # 创建数据加载器
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=True, 
                num_workers=0, 
                pin_memory=False,
                drop_last=True
            )
            
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=False, 
                num_workers=0,
                pin_memory=False
            )
            
            print(f"  训练集: {len(train_dataset)} 张图片")
            print(f"  验证集: {len(val_dataset)} 张图片")
            print(f"  图像尺寸: {self.config['image_size']}x{self.config['image_size']}")
            print(f"  有效batch size: {self.config['batch_size'] * self.config['gradient_accumulation_steps']}")
            
            # 加载VAE
            print(f"📦 加载预训练AutoencoderKL (超低内存模式)...")
            self.vae = AutoencoderKL.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                subfolder="vae"
            ).to(self.device).float()
            
            # 启用梯度检查点
            if self.device != 'cpu' and hasattr(self.vae, 'enable_gradient_checkpointing'):
                self.vae.enable_gradient_checkpointing()
                print("✅ VAE梯度检查点已启用")
            
            for param in self.vae.parameters():
                param.requires_grad = True
            
            self.setup_optimizer()
            self.mse_loss = nn.MSELoss()
            # 超低内存模式不使用感知损失
            self.perceptual_net = None
            
            if self.config['mixed_precision'] and self.device != 'cpu':
                self.scaler = GradScaler()
                print("✅ 混合精度训练已启用")
                
            self.train_history = {
                'epoch': [], 'train_loss': [], 'recon_loss': [],
                'kl_loss': [], 'perceptual_loss': [], 'val_mse': []
            }
            
            self.clear_memory()
            print("✅ 超低内存模式Fine-tuning器初始化完成!")
        
        def setup_perceptual_loss(self):
            """设置感知损失"""
            if not self.config['use_perceptual_loss']:
                print("🚫 感知损失已禁用")
                self.perceptual_net = None
                return
                
            try:
                from torchvision.models import vgg16, VGG16_Weights
                self.perceptual_net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(self.device).float()
                for param in self.perceptual_net.parameters():
                    param.requires_grad = False
                self.perceptual_net.eval()
                print("✅ VGG16感知损失已设置 (FP32)")
            except:
                print("⚠️ VGG16不可用，跳过感知损失")
                self.perceptual_net = None
                
        def compute_perceptual_loss(self, real, fake):
            """计算感知损失"""
            if self.perceptual_net is None:
                return torch.tensor(0.0, device=self.device, dtype=torch.float32)
            
            # 确保网络和输入是float32
            self.perceptual_net.float()
            real = real.float()
            fake = fake.float()
            
            # 确保输入是3通道
            if real.size(1) == 1:
                real = real.repeat(1, 3, 1, 1)
            if fake.size(1) == 1:
                fake = fake.repeat(1, 3, 1, 1)
            
            # 计算VGG特征
            with torch.no_grad():
                real_features = self.perceptual_net(real)
            
            fake_features = self.perceptual_net(fake)
            
            # 计算MSE损失
            return F.mse_loss(real_features.float(), fake_features.float())
            
        def vae_loss(self, images):
            """超低内存VAE损失函数"""
            try:
                # 确保输入为float32
                images = images.float()
                
                # VAE前向传播
                posterior = self.vae.encode(images).latent_dist
                latents = posterior.sample().float()
                reconstructed = self.vae.decode(latents).sample.float()
                
                # 重建损失
                recon_loss = self.mse_loss(reconstructed, images)
                
                # 更稳定的KL散度计算，防止NaN
                try:
                    # 原始KL散度计算可能导致数值不稳定
                    kl_raw = posterior.kl()
                    
                    # 检查并处理NaN或无穷大值
                    if torch.isnan(kl_raw).any() or torch.isinf(kl_raw).any():
                        print("⚠️ 检测到KL散度中的NaN/Inf值，使用替代计算方法")
                        # 替代计算方法
                        mean, var = posterior.mean, posterior.var
                        eps = 1e-8
                        kl_loss = 0.5 * torch.mean(mean.pow(2) + var - torch.log(var + eps) - 1)
                    else:
                        kl_loss = kl_raw.mean()
                        
                    # 额外的安全检查
                    if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                        print("⚠️ KL损失仍然包含NaN/Inf，使用常数替代")
                        kl_loss = torch.tensor(0.1, device=self.device, dtype=torch.float32)
                except Exception as kl_e:
                    print(f"⚠️ KL散度计算错误: {kl_e}，使用替代值")
                    kl_loss = torch.tensor(0.1, device=self.device, dtype=torch.float32)
                
                # 不使用感知损失
                perceptual_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                
                # 计算总损失
                total_loss = (
                    self.config['reconstruction_weight'] * recon_loss +
                    self.config['kl_weight'] * kl_loss
                )
                
                # 最终安全检查
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
                print(f"🔥 超低内存VAE_LOSS内部发生错误，类型: {type(e)}, 内容: {e} 🔥")
                import traceback
                traceback.print_exc()
                raise e
    
    return UltraLowMemoryFineTuner(data_dir, images_dir, annotations_dir, use_cpu=use_cpu)

def create_kaggle_trainer(data_dir, images_dir, annotations_dir):
    """创建适用于Kaggle P100 GPU的高性能训练器"""
    print("🚀 配置Kaggle P100 GPU训练器...")
    
    class KaggleOxfordPetDataset(Dataset):
        """适配Kaggle路径的Oxford-IIIT Pet数据集类"""
        
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
    
    class KaggleVAEFineTuner(CloudVAEFineTuner):
        def __init__(self, data_dir, images_dir, annotations_dir):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🚀 Kaggle P100 - VAE Fine-tuning 设备: {self.device}")
            
            # P100配置 - 16GB显存
            self.config = {
                'batch_size': 12,  # 增大batch size
                'gradient_accumulation_steps': 2,  # 保持适度的梯度累积
                'learning_rate': 1e-5,
                'weight_decay': 0.01,
                'max_epochs': 20,  # 增加训练轮次
                'data_dir': data_dir,
                'images_dir': images_dir,
                'annotations_dir': annotations_dir,
                'save_dir': '/kaggle/working/vae_finetuned',
                'reconstruction_weight': 1.0,
                'kl_weight': 1e-5,  # 保持标准KL权重
                'perceptual_weight': 0.3,  # 启用感知损失
                'use_perceptual_loss': True,  # 启用感知损失
                'save_every_epochs': 2,  # 每2个epoch保存一次
                'eval_every_epochs': 2,  # 每2个epoch评估一次
                'use_gradient_checkpointing': True,
                'mixed_precision': MIXED_PRECISION_AVAILABLE,
                'image_size': 256,  # 使用全尺寸图像
            }
            
            os.makedirs(self.config['save_dir'], exist_ok=True)
            
            # 高性能图像转换
            transform = transforms.Compose([
                transforms.Resize((self.config['image_size'], self.config['image_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            # 加载训练和验证集
            print("📁 加载Kaggle路径的Oxford-IIIT Pet数据集...")
            train_annotation_file = os.path.join(annotations_dir, "trainval.txt")
            train_val_dataset = KaggleOxfordPetDataset(
                images_dir=images_dir,
                annotation_file=train_annotation_file,
                transform=transform
            )
            
            # 分割训练集和验证集
            train_size = int(len(train_val_dataset) * 0.8)
            val_size = len(train_val_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_val_dataset, [train_size, val_size], 
                generator=torch.Generator().manual_seed(42)
            )
            
            # 创建高性能数据加载器
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=True, 
                num_workers=2,  # Kaggle环境有多个CPU核心
                pin_memory=True,  # 启用pin_memory加速数据传输
                drop_last=True
            )
            
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=False, 
                num_workers=2,
                pin_memory=True
            )
            
            print(f"  训练集: {len(train_dataset)} 张图片")
            print(f"  验证集: {len(val_dataset)} 张图片")
            print(f"  图像尺寸: {self.config['image_size']}x{self.config['image_size']}")
            print(f"  有效batch size: {self.config['batch_size'] * self.config['gradient_accumulation_steps']}")
            
            # 加载预训练VAE
            print(f"📦 加载预训练AutoencoderKL...")
            self.vae = AutoencoderKL.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                subfolder="vae"
            ).to(self.device).float()
            
            # 启用梯度检查点
            if hasattr(self.vae, 'enable_gradient_checkpointing'):
                self.vae.enable_gradient_checkpointing()
                print("✅ VAE梯度检查点已启用")
            
            for param in self.vae.parameters():
                param.requires_grad = True
            
            # 配置优化器
            self.setup_optimizer()
            self.mse_loss = nn.MSELoss()
            self.setup_perceptual_loss()
            
            if self.config['mixed_precision']:
                self.scaler = GradScaler()
                print("✅ 混合精度训练已启用")
                
            self.train_history = {
                'epoch': [], 'train_loss': [], 'recon_loss': [],
                'kl_loss': [], 'perceptual_loss': [], 'val_mse': []
            }
            
            self.clear_memory()
            print("✅ Kaggle P100训练器初始化完成!")
            
        def setup_perceptual_loss(self):
            """设置感知损失"""
            if not self.config['use_perceptual_loss']:
                print("🚫 感知损失已禁用")
                self.perceptual_net = None
                return
                
            try:
                from torchvision.models import vgg16, VGG16_Weights
                self.perceptual_net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(self.device).float()
                for param in self.perceptual_net.parameters():
                    param.requires_grad = False
                self.perceptual_net.eval()
                print("✅ VGG16感知损失已设置 (FP32)")
            except:
                print("⚠️ VGG16不可用，跳过感知损失")
                self.perceptual_net = None
                
        def compute_perceptual_loss(self, real, fake):
            """计算感知损失"""
            if self.perceptual_net is None:
                return torch.tensor(0.0, device=self.device, dtype=torch.float32)
            
            # 确保网络和输入是float32
            self.perceptual_net.float()
            real = real.float()
            fake = fake.float()
            
            # 确保输入是3通道
            if real.size(1) == 1:
                real = real.repeat(1, 3, 1, 1)
            if fake.size(1) == 1:
                fake = fake.repeat(1, 3, 1, 1)
            
            # 计算VGG特征
            with torch.no_grad():
                real_features = self.perceptual_net(real)
            
            fake_features = self.perceptual_net(fake)
            
            # 计算MSE损失
            return F.mse_loss(real_features.float(), fake_features.float())
            
        def vae_loss(self, images):
            """稳定版VAE损失函数"""
            try:
                # 确保输入为float32
                images = images.float()
                
                # VAE前向传播
                posterior = self.vae.encode(images).latent_dist
                latents = posterior.sample().float()
                reconstructed = self.vae.decode(latents).sample.float()
                
                # 重建损失
                recon_loss = self.mse_loss(reconstructed, images)
                
                # 更稳定的KL散度计算，防止NaN
                try:
                    # 原始KL散度计算可能导致数值不稳定
                    kl_raw = posterior.kl()
                    
                    # 检查并处理NaN或无穷大值
                    if torch.isnan(kl_raw).any() or torch.isinf(kl_raw).any():
                        print("⚠️ 检测到KL散度中的NaN/Inf值，使用替代计算方法")
                        # 替代计算方法
                        mean, var = posterior.mean, posterior.var
                        eps = 1e-8
                        kl_loss = 0.5 * torch.mean(mean.pow(2) + var - torch.log(var + eps) - 1)
                    else:
                        kl_loss = kl_raw.mean()
                        
                    # 额外的安全检查
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
                
                # 计算总损失
                total_loss = (
                    self.config['reconstruction_weight'] * recon_loss +
                    self.config['kl_weight'] * kl_loss +
                    self.config['perceptual_weight'] * perceptual_loss
                )
                
                # 最终安全检查
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
    
    return KaggleVAEFineTuner(data_dir, images_dir, annotations_dir)

def main():
    """主函数"""
    print("🌐 启动云VAE Fine-tuning完整训练 (Oxford-IIIT Pet数据集)...")
    
    # 设置CUDA内存分配策略
    if torch.cuda.is_available():
        # 启用内存分片管理
        torch.cuda.set_per_process_memory_fraction(0.95)  # 使用95%的显存
        torch.cuda.empty_cache()
        print(f"🔧 CUDA内存优化设置完成")
    
    # Kaggle环境数据目录
    data_dir = '/kaggle/input/dataset-test'
    images_dir = '/kaggle/input/dataset-test/images/images'
    annotations_dir = '/kaggle/input/dataset-test/annotations/annotations'
    
    print(f"📊 数据集路径配置:")
    print(f"  主目录: {data_dir}")
    print(f"  图像目录: {images_dir}")
    print(f"  标注目录: {annotations_dir}")
    
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        print(f"❌ 数据集目录不存在: {images_dir} 或 {annotations_dir}")
        return None
    
    try:
        # 使用适合P100 16GB显存的高性能配置
        print("💡 使用P100 GPU高性能配置...")
        kaggle_trainer = create_kaggle_trainer(data_dir, images_dir, annotations_dir)
        best_loss = kaggle_trainer.finetune()
        print(f"\n🎯 P100 GPU VAE Fine-tuning完成，最佳重建损失: {best_loss:.4f}")
        print("💡 VAE已经适配Oxford-IIIT Pet数据集，可以用于LDM训练")
        return best_loss
    
    except Exception as e:
        print("🔥 Top-level exception caught in main. Full traceback follows: 🔥")
        import traceback
        traceback.print_exc()

        error_str_lower = str(e).lower()
        print(f"🕵️‍♀️ 捕获到异常详情: 类型={type(e)}, 内容='{str(e)}'")

        if "out of memory" in error_str_lower:
            print("💡 尝试使用降级配置进行训练...")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # 降级配置训练
                fallback_trainer = create_ultra_low_memory_finetuner(data_dir, use_cpu=False)
                best_loss = fallback_trainer.finetune()
                print(f"\n🎯 降级配置VAE Fine-tuning完成，最佳重建损失: {best_loss:.4f}")
                return best_loss
            except Exception as fallback_e:
                print(f"❌ 降级配置也失败了: {fallback_e}")
                traceback.print_exc()
                return None
        else:
            print(f"❌ 训练过程中出现未分类的错误: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return None

if __name__ == "__main__":
    main() 
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
    
    def __init__(self, data_dir='/kaggle/input/dataset/dataset'):
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
            'kl_weight': 1e-5,  # 大幅降低KL权重，更关注重建质量
            'perceptual_weight': 0.3,  # 恢复合理的感知损失权重
            'use_perceptual_loss': True,  # 保持感知损失开启
            'save_every_epochs': 5,  # 每5个epoch保存一次
            'eval_every_epochs': 2,  # 每2个epoch评估一次
            'use_gradient_checkpointing': True,  # 启用梯度检查点
            'mixed_precision': MIXED_PRECISION_AVAILABLE,  # 混合精度训练
            'safe_mixed_precision': True,  # 安全模式：遇到问题时自动禁用混合精度
        }
        self.best_model_checkpoint_path = None # 用于跟踪要删除的旧最佳模型
        
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
        ).to(self.device).float()
        
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
            if GRADSCALER_DEVICE:
                self.scaler = GradScaler(GRADSCALER_DEVICE)
            else:
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
            self.perceptual_net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(self.device).float()
            for param in self.perceptual_net.parameters():
                param.requires_grad = False
            self.perceptual_net.eval()
            print("✅ 完整VGG16感知损失已设置 (FP32)")
        except:
            print("⚠️  VGG16不可用，跳过感知损失")
            self.perceptual_net = None
    
    def compute_perceptual_loss(self, real, fake):
        """计算完整感知损失"""
        if self.perceptual_net is None:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        # Ensure network and inputs are float32
        self.perceptual_net.float()
        real = real.float()
        fake = fake.float()
        
        # 确保输入是3通道
        if real.size(1) == 1:
            real = real.repeat(1, 3, 1, 1)
        if fake.size(1) == 1:
            fake = fake.repeat(1, 3, 1, 1)
        
        # 计算VGG特征，完全避免autocast的影响
        with torch.no_grad():
            real_features = self.perceptual_net(real)
        
        # 确保fake_features计算也使用正确的数据类型
        fake_features = self.perceptual_net(fake)
        
        # Ensure MSE损失计算使用相同的数据类型
        return F.mse_loss(real_features.float(), fake_features.float())
    
    def vae_loss(self, images):
        """VAE损失函数 - VAE操作强制FP32，其余依赖外部autocast"""
        # images 输入时，在混合精度模式下已在 train_epoch 中被转换为 .float(), 
        # 并且此函数在顶层 autocast(dtype=torch.float16) 上下文中被调用。

        try:
            # 1. VAE操作：强制在FP32下执行，局部覆盖外部autocast
            if AUTOCAST_DEVICE: # 新版PyTorch
                with autocast(AUTOCAST_DEVICE, enabled=False, dtype=torch.float32):
                    posterior = self.vae.encode(images.float()).latent_dist
                    latents = posterior.sample().float()
                    reconstructed = self.vae.decode(latents).sample.float()
            else: # 传统版本 PyTorch (autocast 指 torch.cuda.amp.autocast)
                with autocast(enabled=False):
                    posterior = self.vae.encode(images.float()).latent_dist
                    latents = posterior.sample().float()
                    reconstructed = self.vae.decode(latents).sample.float()

            # 2. 重建损失：输入确保是FP32
            recon_loss = self.mse_loss(reconstructed.float(), images.float()) # images 已是 .float()
            
            # 3. KL散度损失：结果转换为FP32
            kl_loss = posterior.kl().mean().float()
            
            # 4. 感知损失：输入确保是FP32。compute_perceptual_loss内部已处理好网络精度。
            if self.perceptual_net:
                perceptual_loss = self.compute_perceptual_loss(images.float(), reconstructed.float())
            else:
                perceptual_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            
            # 5. 总损失：所有组件都应是FP32
            total_loss = (
                self.config['reconstruction_weight'] * recon_loss +
                self.config['kl_weight'] * kl_loss +
                self.config['perceptual_weight'] * perceptual_loss
            )
            
            # 6. 返回值：确保所有返回的张量是FP32
            return {
                'total_loss': total_loss.float(),
                'recon_loss': recon_loss.float(),
                'kl_loss': kl_loss.float(),
                'perceptual_loss': perceptual_loss.float(),
                'reconstructed': reconstructed.float(),
                'latents': latents.float()
            }
            
        except Exception as e:
            error_str = str(e).lower()
            if "dtype" in error_str or "type" in error_str or "expected scalar type" in error_str:
                print(f"⚠️ VAE损失计算中的数据类型错误: {e}")
                return {
                    'total_loss': torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=True),
                    'recon_loss': torch.tensor(0.0, device=self.device, dtype=torch.float32),
                    'kl_loss': torch.tensor(0.0, device=self.device, dtype=torch.float32),
                    'perceptual_loss': torch.tensor(0.0, device=self.device, dtype=torch.float32),
                    'reconstructed': images.float(),
                    'latents': torch.zeros_like(images, dtype=torch.float32)
                }
            else:
                print(f"🔥 VAE_LOSS内部发生意外错误，类型: {type(e)}, 内容: {e} 🔥")
                import traceback
                traceback.print_exc()
                raise e
    
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
                    with autocast(AUTOCAST_DEVICE, dtype=torch.float16):
                        loss_dict = self.vae_loss(images)
                        loss = loss_dict['total_loss'] / self.config['gradient_accumulation_steps']
                else:
                    with autocast(): # Old API for mixed precision context
                        loss_dict = self.vae_loss(images)
                        loss = loss_dict['total_loss'] / self.config['gradient_accumulation_steps']
                
                # 确保损失是 float32 类型再进行缩放和反向传播
                self.scaler.scale(loss.float()).backward()
            else:
                # 前向传播 (无混合精度)
                loss_dict = self.vae_loss(images)
                loss = loss_dict['total_loss'] / self.config['gradient_accumulation_steps']
                loss.backward() # 直接反向传播
            
            # 梯度累积
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                try:
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
                    
                except RuntimeError as grad_e:
                    if "dtype" in str(grad_e).lower() or "type" in str(grad_e).lower():
                        print(f"⚠️ 梯度步骤数据类型错误，跳过此步骤: {grad_e}")
                        self.optimizer.zero_grad()
                        continue
                    else:
                        raise grad_e
            
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
        
        if torch.cuda.is_available(): # 确保只在CUDA可用时设置
            torch.autograd.set_detect_anomaly(True)
            print("⚠️ PyTorch 异常检测已启用 (用于调试，可能影响速度)")
        
        best_recon_loss = float('inf')
        
        for epoch in range(self.config['max_epochs']):
            print(f"\n📅 Epoch {epoch+1}/{self.config['max_epochs']}")
            
            train_metrics = self.train_epoch(epoch)
            
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_metrics['avg_loss'])
            self.train_history['recon_loss'].append(train_metrics['avg_recon'])
            self.train_history['kl_loss'].append(train_metrics['avg_kl'])
            self.train_history['perceptual_loss'].append(train_metrics['avg_perceptual'])
            
            if epoch % self.config['eval_every_epochs'] == 0 or epoch == self.config['max_epochs'] - 1:
                val_mse, latent_stats = self.evaluate_reconstruction_quality(self.val_loader)
                self.train_history['val_mse'].append(val_mse)
            
            print(f"📊 Epoch {epoch+1} 结果:")
            print(f"   总损失: {train_metrics['avg_loss']:.4f}")
            print(f"   重建损失: {train_metrics['avg_recon']:.4f}")
            print(f"   KL损失: {train_metrics['avg_kl']:.4f}")
            print(f"   感知损失: {train_metrics['avg_perceptual']:.4f}")
            print(f"   学习率: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # 保存最佳模型逻辑修改
            current_epoch_model_path = f'{self.config["save_dir"]}/vae_finetuned_epoch_{epoch+1}.pth'
            is_best_model_save = False
            if train_metrics['avg_recon'] < best_recon_loss:
                best_recon_loss = train_metrics['avg_recon']
                
                # 尝试删除上一个被标记为"最佳"的模型文件
                if self.best_model_checkpoint_path and os.path.exists(self.best_model_checkpoint_path):
                    # 检查这个旧的最佳模型是否也是一个定期保存点
                    try:
                        # 从文件名提取旧的epoch号
                        old_epoch_str = self.best_model_checkpoint_path.split('_epoch_')[-1].split('.pth')[0]
                        old_epoch_idx = int(old_epoch_str) - 1 # epoch号转为0-indexed
                        is_periodic_save = (old_epoch_idx % self.config['save_every_epochs'] == 0)
                        
                        if not is_periodic_save:
                            print(f"🗑️ 删除旧的最佳模型: {self.best_model_checkpoint_path}")
                            os.remove(self.best_model_checkpoint_path)
                        else:
                            print(f"ℹ️ 保留旧的最佳模型 (同时也是定期保存点): {self.best_model_checkpoint_path}")
                    except Exception as e_remove:
                        print(f"⚠️ 无法删除或检查旧的最佳模型: {e_remove}")

                self.save_finetuned_vae(epoch, train_metrics) # 保存当前模型
                self.best_model_checkpoint_path = current_epoch_model_path # 更新最佳模型路径
                print(f"✅ 保存新的最佳模型 (重建损失: {best_recon_loss:.4f}): {current_epoch_model_path}")
                is_best_model_save = True
            
            if epoch % self.config['save_every_epochs'] == 0:
                if not is_best_model_save: # 如果此epoch已作为最佳模型保存，则不再重复保存
                    self.save_finetuned_vae(epoch, train_metrics)
                self.plot_training_history()
            
            print(f"🏆 当前最佳重建损失: {best_recon_loss:.4f}")
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
            ).to(self.device).float()
            
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
    data_dir = '/kaggle/input/dataset/dataset'
    
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

    except Exception as e: # Catch broader exceptions for dtype issues first
        print("🔥 Top-level exception caught in main. Full traceback follows: 🔥")
        import traceback
        traceback.print_exc() # PRINT FULL TRACEBACK IMMEDIATELY

        error_str_lower = str(e).lower()
        # Using original error string for debug print for more context
        print(f"🕵️‍♀️ 捕获到异常详情: 类型={type(e)}, 内容='{str(e)}'")

        dtype_error_keywords = ["unsupported dtype", "dtype", "expected scalar type", "type mismatch"]
        is_dtype_error = any(keyword in error_str_lower for keyword in dtype_error_keywords)

        if is_dtype_error:
            print(f"❌ 数据类型相关错误诊断 (已由 traceback.print_exc() 显示详细信息): {e}")
            print("🔄 检测到数据类型问题，尝试完全禁用混合精度训练重试...")
            
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            try:
                print("📦 创建纯FP32精度训练器...")
                pure_fp32_finetuner = PureFP32CloudVAEFineTuner(data_dir)
                best_loss = pure_fp32_finetuner.finetune()
                print(f"\n🎯 纯FP32模式VAE Fine-tuning完成，最佳重建损失: {best_loss:.4f}")
                return best_loss
            except Exception as fp32_e:
                print(f"❌ 纯FP32模式也失败了. Full traceback follows: {fp32_e}")
                traceback.print_exc()
                return None

        elif "out of memory" in error_str_lower:
            print(f"❌ 显存不足错误 (已由 traceback.print_exc() 显示详细信息): {e}")
            # ... (rest of OOM handling remains the same, including its own traceback on failure)
            print("🆘 启动应急低内存模式...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            try:
                emergency_finetuner = create_emergency_low_memory_finetuner(data_dir)
                best_loss = emergency_finetuner.finetune()
                print(f"\n🎯 应急模式VAE Fine-tuning完成，最佳重建损失: {best_loss:.4f}")
                return best_loss
            except Exception as emergency_e:
                print(f"❌ 应急模式也失败了. Full traceback follows: {emergency_e}")
                traceback.print_exc()
                print("💡 最终建议: 1. 重启内核清理所有内存 2. 手动设置batch_size=2 3. 使用CPU训练（非常慢）")
                return None
        else:
            print(f"❌ 训练过程中出现未分类的错误 (已由 traceback.print_exc() 显示详细信息): {e}")
            # Fallback for any other exception, already printed by initial traceback
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return None

if __name__ == "__main__":
    main() 
