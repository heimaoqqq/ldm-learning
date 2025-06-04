#!/usr/bin/env python3
"""
VQ-VAE快速开始脚本
帮助您快速测试和使用VQ-VAE训练系统
"""

import os
import sys
import subprocess
import torch

def check_basic_environment():
    """检查基本环境"""
    print("🔍 检查基本环境...")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print()

def create_sample_data():
    """创建示例数据用于测试"""
    print("🎨 创建示例数据集...")
    
    import numpy as np
    from PIL import Image
    
    # 创建data目录
    os.makedirs("data", exist_ok=True)
    
    # 生成一些随机彩色图像作为示例
    for i in range(50):
        # 创建256x256的随机彩色图像
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # 添加一些几何形状使图像更有意义
        if i % 3 == 0:
            # 添加圆形
            center = (128, 128)
            for y in range(256):
                for x in range(256):
                    if (x - center[0])**2 + (y - center[1])**2 < 3000:
                        img_array[y, x] = [255, 100, 100]
        elif i % 3 == 1:
            # 添加矩形
            img_array[50:200, 50:200] = [100, 255, 100]
        else:
            # 添加条纹
            for y in range(256):
                if y % 20 < 10:
                    img_array[y, :] = [100, 100, 255]
        
        # 保存图像
        img = Image.fromarray(img_array)
        img.save(f"data/sample_image_{i:03d}.png")
    
    print(f"✅ 创建了50张示例图像到 data/ 目录")
    print()

def install_minimal_dependencies():
    """安装最小依赖"""
    print("📦 安装必要依赖...")
    
    packages = [
        "torch",
        "torchvision", 
        "pytorch-lightning",
        "omegaconf",
        "pillow",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "tqdm"
    ]
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"📦 安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print()

def test_dataset_loading():
    """测试数据加载"""
    print("🔄 测试数据加载...")
    
    try:
        from dataset import create_dataloaders
        
        train_loader, val_loader, test_loader = create_dataloaders(
            data_path="data",
            batch_size=4,
            num_workers=0,  # 避免多进程问题
            image_size=256
        )
        
        # 测试加载一个批次
        batch = next(iter(train_loader))
        print(f"✅ 数据加载成功! 批次形状: {batch.shape}")
        print(f"   训练集: {len(train_loader.dataset)} 张")
        print(f"   验证集: {len(val_loader.dataset)} 张")
        print(f"   测试集: {len(test_loader.dataset)} 张")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False
    
    print()
    return True

def test_model_creation():
    """测试模型创建"""
    print("🏗️ 测试模型创建...")
    
    try:
        # 使用简化配置避免taming-transformers依赖
        from vqvae_model import create_vqvae_model
        
        model = create_vqvae_model(
            n_embed=1024,      # 小码本用于测试
            embed_dim=64,      # 小嵌入维度
            ch=32,             # 小通道数
            resolution=256,
            learning_rate=1e-4
        )
        
        # 测试前向传播
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            recon, loss = model(x)
        
        print(f"✅ 模型创建成功!")
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {recon.shape}")
        print(f"   参数数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        print("可能需要安装taming-transformers，请运行: python setup_environment.py")
        return False
    
    print()
    return True

def run_quick_training():
    """运行快速训练测试"""
    print("🚀 运行快速训练测试 (5个epoch)...")
    
    # 创建简化的训练配置
    quick_config = """
# 快速测试配置
model:
  target: vqvae_model.VQVAEModel  
  params:
    embed_dim: 64
    n_embed: 1024
    ch: 32
    ch_mult: [1,2,2]
    num_res_blocks: 1
    attn_resolutions: []
    resolution: 256
    in_channels: 3
    out_ch: 3
    z_channels: 64
    beta: 0.25
    learning_rate: 1e-4
    disc_start: 10000
    disc_weight: 0.5

data:
  target: dataset.create_dataloaders
  params:
    data_path: "data"
    batch_size: 4
    num_workers: 0
    image_size: 256

trainer:
  accelerator: "auto"
  devices: 1
  precision: 32
  max_epochs: 5
  check_val_every_n_epoch: 1
  log_every_n_steps: 5
  enable_progress_bar: true
"""
    
    # 保存配置
    os.makedirs("configs", exist_ok=True)
    with open("configs/quick_test_config.yaml", "w") as f:
        f.write(quick_config)
    
    print("配置文件已创建: configs/quick_test_config.yaml")
    print("现在可以运行:")
    print("  python train_vqvae.py --config configs/quick_test_config.yaml")
    print()

def main():
    """主函数"""
    print("🎯 VQ-VAE快速开始向导")
    print("=" * 50)
    
    # 1. 检查环境
    check_basic_environment()
    
    # 2. 安装依赖
    install_minimal_dependencies()
    
    # 3. 创建示例数据
    if not os.path.exists("data") or len(os.listdir("data")) == 0:
        create_sample_data()
    else:
        print("✅ 数据目录已存在，跳过示例数据创建")
        print()
    
    # 4. 测试数据加载
    if not test_dataset_loading():
        print("❌ 数据加载测试失败，请检查环境")
        return
    
    # 5. 测试模型创建
    if not test_model_creation():
        print("❌ 模型创建测试失败")
        print("建议运行完整环境设置: python setup_environment.py")
        return
    
    # 6. 创建快速训练配置
    run_quick_training()
    
    print("🎉 快速开始配置完成!")
    print()
    print("📋 接下来的步骤:")
    print("1. 将您的256×256彩色图像放入 data/ 目录")
    print("2. 运行快速训练测试:")
    print("   python train_vqvae.py --config configs/quick_test_config.yaml")
    print("3. 如需完整功能，请运行:")
    print("   python setup_environment.py")
    print()
    print("💡 提示:")
    print("- 当前使用简化配置，适合快速测试")
    print("- 完整功能需要安装taming-transformers")
    print("- 查看README.md获取详细使用说明")

if __name__ == "__main__":
    main() 
