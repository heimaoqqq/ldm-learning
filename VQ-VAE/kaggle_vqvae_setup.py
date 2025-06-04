#!/usr/bin/env python3
"""
基于官方taming-transformers的Kaggle VQ-VAE训练环境
完全兼容官方实现，针对P100 16GB GPU优化
"""

import os
import sys
import subprocess
import torch
import shutil
from pathlib import Path

def setup_kaggle_taming_environment():
    """设置Kaggle环境 - 基于官方taming-transformers"""
    print("🔧 基于官方taming-transformers设置Kaggle VQ-VAE环境")
    print("=" * 60)
    
    # 检查环境
    print("📊 环境信息:")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print()
    
    # 检查taming-transformers-master是否存在
    taming_dir = Path("../taming-transformers-master").resolve()
    if not taming_dir.exists():
        print(f"❌ 未找到taming-transformers-master目录: {taming_dir}")
        print("请确保taming-transformers-master目录在正确位置")
        return False
    
    print(f"✅ 找到taming-transformers: {taming_dir}")
    
    # 安装必要依赖
    print("📦 安装依赖包...")
    packages = [
        "omegaconf",
        "einops", 
        "lpips",
        "transformers",
        "albumentations",
        "kornia",
        "pytorch-lightning"
    ]
    
    for package in packages:
        try:
            print(f"安装 {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ])
        except subprocess.CalledProcessError as e:
            print(f"安装 {package} 失败: {e}")
    
    # 添加taming-transformers到Python路径
    sys.path.insert(0, str(taming_dir))
    
    # 验证能否导入taming模块
    try:
        from taming.models.vqgan import VQModel
        print("✅ taming.models.vqgan导入成功")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    # 创建工作目录
    work_dirs = [
        "checkpoints",
        "logs", 
        "results",
        "configs"
    ]
    
    for directory in work_dirs:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ 创建目录: {directory}")
    
    print("\n🎉 Kaggle taming-transformers环境设置完成!")
    return True

def create_kaggle_custom_dataset():
    """创建Kaggle自定义数据集类"""
    dataset_code = '''
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
import numpy as np
from pathlib import Path

class KaggleImageDataset(Dataset):
    """Kaggle环境自定义图像数据集 - 兼容taming-transformers"""
    
    def __init__(self, data_path="/kaggle/input/dataset", size=256, 
                 random_crop=False, interpolation="bicubic"):
        self.data_path = data_path
        self.size = size
        self.random_crop = random_crop
        self.interpolation = interpolation
        
        # 获取所有图像文件
        self.image_paths = self._collect_image_paths()
        print(f"📸 Kaggle数据集: {len(self.image_paths)} 张图像")
        
        # 设置图像变换
        self.transform = self._get_transform()
    
    def _collect_image_paths(self):
        """收集所有图像路径"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_paths = []
        
        data_path = Path(self.data_path)
        for ext in extensions:
            paths = list(data_path.rglob(ext))  # 递归搜索
            image_paths.extend([str(p) for p in paths])
        
        return sorted(image_paths)
    
    def _get_transform(self):
        """获取图像变换 - 与taming兼容"""
        if self.random_crop:
            # 训练时的变换
            transform_list = [
                transforms.Resize(self.size),
                transforms.RandomCrop(self.size),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        else:
            # 验证时的变换
            transform_list = [
                transforms.Resize(self.size),
                transforms.CenterCrop(self.size),
            ]
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 标准化到[-1,1]
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            
            # 返回taming期望的格式
            return {"image": image}
            
        except Exception as e:
            print(f"⚠️ 加载图像失败 {self.image_paths[idx]}: {e}")
            # 返回随机图像作为备用
            random_image = torch.randn(3, self.size, self.size)
            random_image = torch.tanh(random_image)  # 限制到[-1,1]
            return {"image": random_image}

class KaggleDataModule:
    """Kaggle数据模块 - 替代官方DataModuleFromConfig"""
    
    def __init__(self, data_path="/kaggle/input/dataset", batch_size=12, 
                 num_workers=2, image_size=256):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        # 创建数据集
        self.setup_datasets()
    
    def setup_datasets(self):
        """设置训练/验证/测试数据集"""
        # 创建完整数据集
        full_dataset = KaggleImageDataset(
            data_path=self.data_path,
            size=self.image_size,
            random_crop=True  # 训练时随机裁剪
        )
        
        # 分割数据集
        total_size = len(full_dataset)
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)
        test_size = total_size - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = \\
            torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
        
        print(f"📊 数据集分割:")
        print(f"  训练集: {train_size} 张")
        print(f"  验证集: {val_size} 张")
        print(f"  测试集: {test_size} 张")
    
    def train_dataloader(self):
        """训练数据加载器"""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        """验证数据加载器"""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """测试数据加载器"""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
'''
    
    with open("kaggle_dataset.py", "w", encoding='utf-8') as f:
        f.write(dataset_code)
    
    print("✅ Kaggle自定义数据集创建完成")

def create_p100_vqgan_config():
    """创建P100 GPU优化的VQ-GAN配置"""
    config = '''# Kaggle P100 16GB VQ-GAN配置
# 基于官方taming-transformers，针对P100优化

model:
  base_learning_rate: 4.5e-6   # 官方推荐学习率
  target: taming.models.vqgan.VQModel
  params:
    embed_dim: 256
    n_embed: 8192              # 适中码本大小，平衡质量和内存
    
    ddconfig:
      double_z: false
      z_channels: 256
      resolution: 256
      in_channels: 3
      out_ch: 3
      ch: 128                  # 基础通道数
      ch_mult: [1,1,2,4]      # P100内存优化：减少层数
      num_res_blocks: 2
      attn_resolutions: [32]   # 只在32x32分辨率使用注意力
      dropout: 0.0

    lossconfig:
      target: taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator
      params:
        disc_conditional: false
        disc_in_channels: 3
        disc_start: 30001       # 较早启动判别器
        disc_weight: 0.6        # 适中的判别器权重
        codebook_weight: 1.0
        pixelloss_weight: 1.0
        perceptual_weight: 1.0

# Lightning训练器配置
lightning:
  trainer:
    accelerator: "gpu"
    devices: 1
    precision: 16             # 混合精度节省内存
    max_epochs: 100
    check_val_every_n_epoch: 2
    accumulate_grad_batches: 2  # 梯度累积：等效batch_size=24
    gradient_clip_val: 1.0
    log_every_n_steps: 50
    enable_progress_bar: true
    
  callbacks:
    model_checkpoint:
      dirpath: "checkpoints"
      filename: "vqgan-p100-{epoch:02d}-{val_rec_loss:.4f}"
      monitor: "val/rec_loss"
      mode: "min"
      save_top_k: 3
      save_last: true
      every_n_epochs: 5
      
    image_logger:
      batch_frequency: 500     # 每500步记录一次图像
      max_images: 8

# P100性能预期
performance_expectations:
  memory_usage: "~14GB"
  time_per_epoch: "15-25分钟"
  recommended_batch_size: 12
  effective_batch_size: 24    # 通过梯度累积
  
# 训练提示
training_tips:
  - "使用mixed precision (16-bit) 节省内存"
  - "batch_size=12是P100的最优设置"
  - "accumulate_grad_batches=2增加有效批次大小"
  - "监控val/rec_loss作为主要指标"
  - "建议训练80-100个epoch"
'''
    
    with open("configs/kaggle_p100_vqgan.yaml", "w", encoding='utf-8') as f:
        f.write(config)
    
    print("✅ P100优化VQ-GAN配置创建完成")

def create_kaggle_training_script():
    """创建Kaggle训练脚本"""
    script = '''#!/usr/bin/env python3
"""
Kaggle VQ-GAN训练脚本
基于官方taming-transformers，P100 16GB优化
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from pathlib import Path

# 添加taming-transformers到路径
sys.path.insert(0, str(Path("../taming-transformers-master").resolve()))

def main():
    print("🚀 开始Kaggle VQ-GAN训练 (基于官方taming-transformers)")
    print("=" * 70)
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ 未检测到GPU")
        return
        
    print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # 导入taming模块
    try:
        from taming.models.vqgan import VQModel
        print("✅ 成功导入taming.models.vqgan.VQModel")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请检查taming-transformers-master路径")
        return
    
    # 导入自定义数据模块
    try:
        from kaggle_dataset import KaggleDataModule
        print("✅ 成功导入KaggleDataModule")
    except ImportError as e:
        print(f"❌ 导入kaggle_dataset失败: {e}")
        return
    
    # 加载配置
    config_path = "configs/kaggle_p100_vqgan.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
        
    config = OmegaConf.load(config_path)
    print("✅ 配置加载成功")
    
    # 创建数据模块
    try:
        data_module = KaggleDataModule(
            data_path="/kaggle/input/dataset",
            batch_size=12,  # P100优化批次大小
            num_workers=2,
            image_size=256
        )
        print("✅ 数据模块创建成功")
    except Exception as e:
        print(f"❌ 数据模块创建失败: {e}")
        return
    
    # 创建模型
    try:
        model = VQModel(**config.model.params)
        model.learning_rate = config.model.base_learning_rate
        print(f"✅ VQ-GAN模型创建成功")
        
        # 计算模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数: {total_params/1e6:.2f}M")
        print(f"可训练参数: {trainable_params/1e6:.2f}M")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return
    
    # 创建回调函数
    callbacks = []
    
    # 检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="vqgan-kaggle-{epoch:02d}-{val_rec_loss:.4f}",
        monitor="val/rec_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=5,
        save_weights_only=False
    )
    callbacks.append(checkpoint_callback)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # 创建日志记录器
    logger = TensorBoardLogger(
        save_dir="logs",
        name="vqgan_kaggle_p100",
        version=None
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=16,  # 混合精度
        max_epochs=100,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        check_val_every_n_epoch=2,
        accumulate_grad_batches=2,  # 梯度累积
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        benchmark=True
    )
    
    print("✅ 训练器创建成功")
    
    # 开始训练
    try:
        print("\\n🚀 开始训练...")
        print("📊 训练配置:")
        print(f"  - 批次大小: 12 (有效: 24)")
        print(f"  - 学习率: {config.model.base_learning_rate}")
        print(f"  - 码本大小: {config.model.params.n_embed}")
        print(f"  - 最大epochs: 100")
        print(f"  - 混合精度: 16-bit")
        
        trainer.fit(model, data_module)
        print("\\n✅ 训练完成!")
        
        # 保存最终模型
        final_path = "checkpoints/final_vqgan_model.ckpt"
        trainer.save_checkpoint(final_path)
        print(f"✅ 最终模型已保存: {final_path}")
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    
    with open("train_kaggle_vqgan.py", "w", encoding='utf-8') as f:
        f.write(script)
    
    os.chmod("train_kaggle_vqgan.py", 0o755)
    print("✅ Kaggle训练脚本创建完成")

def create_evaluation_script():
    """创建模型评估脚本"""
    eval_script = '''#!/usr/bin/env python3
"""
VQ-GAN模型评估脚本
用于检查训练结果和生成样本
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# 添加taming-transformers到路径
sys.path.insert(0, str(Path("../taming-transformers-master").resolve()))

def load_model(checkpoint_path):
    """加载训练好的VQ-GAN模型"""
    try:
        from taming.models.vqgan import VQModel
        
        # 从检查点加载
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = VQModel.load_from_checkpoint(checkpoint_path)
        model.eval()
        
        print(f"✅ 模型加载成功: {checkpoint_path}")
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

def evaluate_reconstruction(model, image_path, device='cuda'):
    """评估重建质量"""
    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    x = transform(image).unsqueeze(0).to(device)
    model = model.to(device)
    
    with torch.no_grad():
        # 编码
        quant, _, info = model.encode(x)
        
        # 解码
        xrec = model.decode(quant)
        
        # 计算重建损失
        rec_loss = torch.nn.functional.mse_loss(x, xrec)
        
        # 码本使用情况
        indices = info[2].flatten()
        unique_indices = torch.unique(indices)
        codebook_usage = len(unique_indices) / model.quantize.n_embed
        
    return {
        'original': x,
        'reconstructed': xrec,
        'rec_loss': rec_loss.item(),
        'codebook_usage': codebook_usage,
        'quantized_indices': indices
    }

def visualize_results(results, save_path="evaluation_results.png"):
    """可视化重建结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 反归一化函数
    def denorm(x):
        return (x + 1) / 2
    
    # 原始图像
    orig_img = denorm(results['original'].squeeze().cpu()).permute(1, 2, 0)
    axes[0].imshow(orig_img.numpy())
    axes[0].set_title("原始图像")
    axes[0].axis('off')
    
    # 重建图像
    rec_img = denorm(results['reconstructed'].squeeze().cpu()).permute(1, 2, 0)
    axes[1].imshow(rec_img.numpy())
    axes[1].set_title(f"重建图像\\nMSE Loss: {results['rec_loss']:.4f}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 结果可视化已保存: {save_path}")

def main():
    print("📊 VQ-GAN模型评估")
    print("=" * 40)
    
    # 查找最新的检查点
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print("❌ 检查点目录不存在")
        return
    
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        print("❌ 未找到检查点文件")
        return
    
    # 使用最新的检查点
    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
    print(f"🔍 使用检查点: {latest_checkpoint.name}")
    
    # 加载模型
    model = load_model(str(latest_checkpoint))
    if model is None:
        return
    
    # 查找测试图像
    test_images = list(Path("/kaggle/input/dataset").rglob("*.jpg"))[:5]
    if not test_images:
        test_images = list(Path("/kaggle/input/dataset").rglob("*.png"))[:5]
    
    if not test_images:
        print("❌ 未找到测试图像")
        return
    
    print(f"🖼️ 找到 {len(test_images)} 张测试图像")
    
    # 评估每张图像
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_losses = []
    all_codebook_usage = []
    
    for i, img_path in enumerate(test_images):
        print(f"评估图像 {i+1}/{len(test_images)}: {img_path.name}")
        
        try:
            results = evaluate_reconstruction(model, str(img_path), device)
            all_losses.append(results['rec_loss'])
            all_codebook_usage.append(results['codebook_usage'])
            
            # 保存可视化结果
            save_path = f"evaluation_image_{i+1}.png"
            visualize_results(results, save_path)
            
        except Exception as e:
            print(f"❌ 评估失败: {e}")
            continue
    
    # 打印总体统计
    if all_losses:
        print("\\n📈 评估总结:")
        print(f"平均重建损失: {np.mean(all_losses):.4f} ± {np.std(all_losses):.4f}")
        print(f"平均码本使用率: {np.mean(all_codebook_usage):.2%} ± {np.std(all_codebook_usage):.2%}")
        
        # 模型信息
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"模型参数: {total_params:.2f}M")
        print(f"码本大小: {model.quantize.n_embed}")
        print(f"嵌入维度: {model.quantize.embed_dim}")

if __name__ == "__main__":
    main()
'''
    
    with open("evaluate_vqgan.py", "w", encoding='utf-8') as f:
        f.write(eval_script)
    
    os.chmod("evaluate_vqgan.py", 0o755)
    print("✅ 评估脚本创建完成")

def main():
    """主函数"""
    print("🎯 基于官方taming-transformers的Kaggle VQ-VAE环境")
    print("=" * 60)
    
    # 1. 环境设置
    if not setup_kaggle_taming_environment():
        print("❌ 环境设置失败")
        return
    
    # 2. 创建自定义数据集
    create_kaggle_custom_dataset()
    
    # 3. 创建P100优化配置
    create_p100_vqgan_config()
    
    # 4. 创建训练脚本
    create_kaggle_training_script()
    
    # 5. 创建评估脚本
    create_evaluation_script()
    
    print("\n🎉 完整的Kaggle VQ-GAN训练环境设置完成!")
    print("\n📋 使用方法:")
    print("1. 确保taming-transformers-master在上级目录")
    print("2. 在Kaggle Notebook中运行:")
    print("   !cd VQ-VAE && python kaggle_vqvae_setup.py")
    print("3. 开始训练:")
    print("   !cd VQ-VAE && python train_kaggle_vqgan.py")
    print("4. 评估模型:")
    print("   !cd VQ-VAE && python evaluate_vqgan.py")
    
    print("\n💡 关键特性:")
    print("✅ 100%基于官方taming-transformers")
    print("✅ P100 16GB专门优化 (batch_size=12)")
    print("✅ 混合精度训练节省内存")
    print("✅ 梯度累积增加有效批次大小")
    print("✅ 自动适配 /kaggle/input/dataset")
    print("✅ 完整的训练监控和可视化")
    
    print("\n🔧 P100优化配置:")
    print(f"- 码本大小: 8192")
    print(f"- 批次大小: 12 (有效: 24)")
    print(f"- 内存使用: ~14GB")
    print(f"- 预计训练时间: 15-25分钟/epoch")

if __name__ == "__main__":
    main() 
