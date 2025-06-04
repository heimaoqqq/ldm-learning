#!/usr/bin/env python3
"""
Kaggle环境VQ-VAE设置脚本
专门为Kaggle P100 16GB GPU优化
"""

import os
import sys
import subprocess
import torch

def setup_kaggle_environment():
    """设置Kaggle环境"""
    print("🔧 设置Kaggle VQ-VAE训练环境")
    print("=" * 50)
    
    # 检查环境
    print("📊 环境信息:")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print()
    
    # 安装必要依赖
    print("📦 安装依赖包...")
    packages = [
        "omegaconf",
        "einops", 
        "lpips",
        "transformers",
        "albumentations",
        "kornia"  # 用于数据增强
    ]
    
    for package in packages:
        try:
            print(f"安装 {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ])
        except subprocess.CalledProcessError as e:
            print(f"安装 {package} 失败: {e}")
    
    # 克隆taming-transformers
    print("📥 克隆taming-transformers...")
    if not os.path.exists("/kaggle/working/taming-transformers"):
        try:
            subprocess.check_call([
                "git", "clone", 
                "https://github.com/CompVis/taming-transformers.git",
                "/kaggle/working/taming-transformers"
            ])
            print("✅ taming-transformers克隆成功")
        except subprocess.CalledProcessError:
            print("❌ 克隆失败，使用备用方案")
            return False
    
    # 安装taming-transformers
    try:
        os.chdir("/kaggle/working/taming-transformers")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"])
        os.chdir("/kaggle/working")
        print("✅ taming-transformers安装成功")
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        return False
    
    # 创建目录结构
    directories = [
        "/kaggle/working/checkpoints",
        "/kaggle/working/logs", 
        "/kaggle/working/results",
        "/kaggle/working/configs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 创建目录: {directory}")
    
    print("\n🎉 Kaggle环境设置完成!")
    return True

def create_kaggle_dataset_loader():
    """创建Kaggle数据集加载器"""
    kaggle_dataset_code = '''
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import numpy as np

class KaggleImageDataset(Dataset):
    """Kaggle环境专用数据集加载器"""
    
    def __init__(self, data_path="/kaggle/input/dataset", split="train", image_size=256):
        self.data_path = data_path
        self.split = split
        self.image_size = image_size
        
        # 获取所有图像文件
        self.image_paths = self._get_image_paths()
        print(f"Kaggle数据集 - {split}: {len(self.image_paths)} 张图像")
        
        # 图像变换
        self.transform = self._get_transforms()
    
    def _get_image_paths(self):
        """获取图像路径"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        
        # 搜索数据集目录
        for ext in extensions:
            paths = glob.glob(os.path.join(self.data_path, "**", ext), recursive=True)
            image_paths.extend(paths)
        
        return sorted(image_paths)
    
    def _get_transforms(self):
        """定义变换"""
        if self.split == "train":
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"加载图像失败 {self.image_paths[idx]}: {e}")
            return torch.randn(3, self.image_size, self.image_size)

def create_kaggle_dataloaders(data_path="/kaggle/input/dataset", batch_size=12):
    """创建Kaggle数据加载器"""
    # 创建数据集
    full_dataset = KaggleImageDataset(data_path, "train")
    
    # 分割数据集
    total_size = len(full_dataset)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1) 
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器 (P100优化配置)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)
    
    print(f"Kaggle数据加载器创建完成:")
    print(f"  训练集: {len(train_dataset)} 张")
    print(f"  验证集: {len(val_dataset)} 张") 
    print(f"  测试集: {len(test_dataset)} 张")
    print(f"  批次大小: {batch_size}")
    
    return train_loader, val_loader, test_loader
'''
    
    with open("/kaggle/working/kaggle_dataset.py", "w") as f:
        f.write(kaggle_dataset_code)
    
    print("✅ Kaggle数据集加载器创建完成")

def create_p100_optimized_config():
    """创建P100 GPU优化配置"""
    p100_config = """# Kaggle P100 16GB GPU优化配置
# 专门针对您的硬件和数据集路径

model:
  target: taming.models.vqgan.VQModel
  params:
    # 模型参数 - P100优化
    embed_dim: 256
    n_embed: 8192          # 适中的码本大小，平衡质量和内存
    
    # 编码器/解码器 - 内存优化
    ddconfig:
      double_z: false
      z_channels: 256
      resolution: 256
      in_channels: 3
      out_ch: 3
      ch: 128              # 基础通道数
      ch_mult: [1,1,2,4]   # 减少层数节省内存
      num_res_blocks: 2
      attn_resolutions: [32]  # 只在32x32使用注意力
      dropout: 0.0
    
    # 损失函数配置
    lossconfig:
      target: taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator
      params:
        disc_conditional: false
        disc_in_channels: 3
        disc_start: 30001      # 较早启动判别器
        disc_weight: 0.6       # 适中的判别器权重
        codebook_weight: 1.0
        pixelloss_weight: 1.0
        perceptual_weight: 1.0

# 数据配置 - Kaggle路径
data:
  target: kaggle_dataset.create_kaggle_dataloaders
  params:
    data_path: "/kaggle/input/dataset"
    batch_size: 12         # P100 16GB最优批次大小
    image_size: 256

# 训练器配置 - P100优化
trainer:
  accelerator: "gpu"
  devices: 1
  precision: 16          # 混合精度训练节省内存
  max_epochs: 80
  check_val_every_n_epoch: 2
  accumulate_grad_batches: 2  # 梯度累积增加有效批次大小
  gradient_clip_val: 1.0
  log_every_n_steps: 50
  enable_progress_bar: true
  benchmark: true
  deterministic: false

# 优化器配置
lightning_module_config:
  learning_rate: 4.5e-6
  weight_decay: 0.0

# 回调配置
callbacks:
  checkpoint:
    target: pytorch_lightning.callbacks.ModelCheckpoint
    params:
      dirpath: "/kaggle/working/checkpoints"
      filename: "vqvae-p100-{epoch:02d}-{val_loss:.4f}"
      monitor: "val/rec_loss"
      mode: "min"
      save_top_k: 3
      save_last: true
      every_n_epochs: 5

# 日志配置
logger:
  target: pytorch_lightning.loggers.TensorBoardLogger
  params:
    save_dir: "/kaggle/working/logs"
    name: "vqvae_kaggle_p100"

# P100特定优化
p100_optimizations:
  # 内存优化建议
  - "使用precision: 16 混合精度训练"
  - "batch_size: 12 是P100的最优设置"
  - "accumulate_grad_batches: 2 增加有效批次大小"
  - "ch_mult: [1,1,2,4] 减少内存使用"
  - "attn_resolutions: [32] 仅在必要分辨率使用注意力"
  
# 预期性能
expected_performance:
  training_time_per_epoch: "15-25分钟"
  memory_usage: "~14GB"
  reconstruction_loss: "0.1-0.3"
  codebook_usage: "70-85%"
"""
    
    with open("/kaggle/working/configs/kaggle_p100_config.yaml", "w") as f:
        f.write(p100_config)
    
    print("✅ P100优化配置创建完成")

def create_kaggle_training_script():
    """创建Kaggle训练脚本"""
    training_script = '''#!/usr/bin/env python3
"""
Kaggle VQ-VAE训练脚本
针对P100 16GB GPU优化
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

# 添加taming-transformers到路径
sys.path.append("/kaggle/working/taming-transformers")

def main():
    print("🚀 开始Kaggle VQ-VAE训练")
    print("=" * 50)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("❌ 未检测到GPU")
        return
    
    # 加载配置
    config_path = "/kaggle/working/configs/kaggle_p100_config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
        
    config = OmegaConf.load(config_path)
    print("✅ 配置加载成功")
    
    # 导入所需模块
    try:
        from taming.models.vqgan import VQModel
        from kaggle_dataset import create_kaggle_dataloaders
        print("✅ 模块导入成功")
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return
    
    # 创建数据加载器
    try:
        train_loader, val_loader, test_loader = create_kaggle_dataloaders(
            data_path="/kaggle/input/dataset",
            batch_size=config.data.params.batch_size
        )
        print("✅ 数据加载器创建成功")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 创建模型
    try:
        model = VQModel(**config.model.params)
        print(f"✅ 模型创建成功")
        print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return
    
    # 创建回调
    checkpoint_callback = ModelCheckpoint(
        dirpath="/kaggle/working/checkpoints",
        filename="vqvae-kaggle-{epoch:02d}-{val_loss:.4f}",
        monitor="val/rec_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=5
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=16,
        max_epochs=config.trainer.max_epochs,
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger("/kaggle/working/logs", name="vqvae_kaggle"),
        log_every_n_steps=50,
        accumulate_grad_batches=2,
        gradient_clip_val=1.0
    )
    
    print("✅ 训练器创建成功")
    
    # 开始训练
    try:
        print("🚀 开始训练...")
        trainer.fit(model, train_loader, val_loader)
        print("✅ 训练完成!")
        
        # 测试模型
        print("📊 开始测试...")
        trainer.test(model, test_loader, ckpt_path="best")
        print("✅ 测试完成!")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    
    with open("/kaggle/working/train_kaggle.py", "w") as f:
        f.write(training_script)
    
    os.chmod("/kaggle/working/train_kaggle.py", 0o755)
    print("✅ Kaggle训练脚本创建完成")

def main():
    """主函数"""
    print("🎯 Kaggle VQ-VAE环境设置")
    
    # 1. 环境设置
    if not setup_kaggle_environment():
        print("❌ 环境设置失败")
        return
    
    # 2. 创建数据加载器
    create_kaggle_dataset_loader()
    
    # 3. 创建P100优化配置
    create_p100_optimized_config()
    
    # 4. 创建训练脚本
    create_kaggle_training_script()
    
    print("\n🎉 Kaggle环境设置完成!")
    print("\n📋 使用方法:")
    print("1. 在Kaggle Notebook中运行:")
    print("   !python /kaggle/working/kaggle_setup.py")
    print("2. 开始训练:")
    print("   !python /kaggle/working/train_kaggle.py")
    print("\n💡 P100 16GB优化特性:")
    print("- 批次大小: 12 (最优设置)")
    print("- 混合精度训练节省内存")
    print("- 梯度累积增加有效批次大小")
    print("- 码本大小: 8192 (平衡质量和内存)")

if __name__ == "__main__":
    main() 
