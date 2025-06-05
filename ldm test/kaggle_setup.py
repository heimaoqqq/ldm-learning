#!/usr/bin/env python3
"""
Kaggle环境设置脚本
在Kaggle Notebook中运行此脚本来设置训练环境
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def setup_kaggle_environment():
    """设置Kaggle训练环境"""
    
    print("🚀 开始设置Kaggle环境...")
    
    # 1. 检查数据集
    dataset_path = "/kaggle/input/the-oxfordiiit-pet-dataset"
    if os.path.exists(dataset_path):
        print(f"✅ 数据集已找到: {dataset_path}")
        files = os.listdir(dataset_path)
        print(f"   数据集包含 {len(files)} 个文件")
        
        # 检查图像文件
        image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"   其中图像文件: {len(image_files)} 个")
        
        if len(image_files) > 0:
            print(f"   示例文件: {image_files[:5]}")
    else:
        print(f"❌ 数据集未找到: {dataset_path}")
        print("请确保您已添加 'the-oxfordiiit-pet-dataset' 数据集到您的Kaggle Notebook")
        return False
    
    # 2. 检查并安装依赖
    print("\n📦 检查依赖...")
    
    required_packages = [
        'omegaconf',
        'pytorch-lightning==1.4.2',
        'torchmetrics==0.6.0',
        'torch-fidelity',
        'transformers==4.19.2',
        'kornia==0.6.0',
        'opencv-python==4.1.2.30',
        'imageio==2.9.0',
        'imageio-ffmpeg==0.4.2',
        'albumentations==0.4.3',
        'pudb==2019.2',
        'invisible-watermark',
        'streamlit>=0.73.1',
        'test-tube>=0.7.5',
        'streamlit==1.12.1',
        'einops==0.3.0',
        'torch-fidelity==0.3.0',
        'taming-transformers',
        'clip-by-openai',
        'torchtext==0.2.3',
    ]
    
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
            print(f"   ✅ {package}")
        except:
            print(f"   ⚠️ {package} 安装失败，可能已存在或不兼容")
    
    # 3. 检查latent-diffusion是否存在
    if not os.path.exists("/kaggle/working/latent-diffusion"):
        print("\n📥 克隆latent-diffusion仓库...")
        try:
            subprocess.check_call([
                "git", "clone", 
                "https://github.com/CompVis/latent-diffusion.git",
                "/kaggle/working/latent-diffusion"
            ])
            print("   ✅ latent-diffusion 克隆完成")
        except:
            print("   ❌ 克隆失败，请手动克隆")
            return False
    else:
        print("\n✅ latent-diffusion 已存在")
    
    # 4. 设置工作目录结构
    print("\n📁 创建目录结构...")
    directories = [
        "/kaggle/working/VAE",
        "/kaggle/working/VAE/logs",
        "/kaggle/working/VAE/logs/checkpoints",
        "/kaggle/working/LDM", 
        "/kaggle/working/LDM/logs",
        "/kaggle/working/LDM/logs/checkpoints",
        "/kaggle/working/generated_pets"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"   ✅ {dir_path}")
    
    # 5. 创建符号链接到数据集（如果需要）
    print("\n🔗 设置数据集访问...")
    print(f"   数据集位置: {dataset_path}")
    print(f"   可直接在配置文件中使用此路径")
    
    print("\n🎉 Kaggle环境设置完成!")
    print("\n📋 接下来的步骤:")
    print("1. 确保您的项目文件（VAE/, LDM/）已上传到 /kaggle/working/")
    print("2. 运行: python VAE/train_vae_kaggle.py")
    print("3. 训练完成后运行: python LDM/train_ldm_kaggle.py")
    print("4. 生成图像: python LDM/generate_pets.py --ldm_ckpt /kaggle/working/ldm_final.ckpt")
    
    return True

def check_gpu():
    """检查GPU可用性"""
    import torch
    print(f"\n🖥️ GPU信息:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   ✅ GPU: {gpu_name}")
        print(f"   💾 显存: {gpu_memory:.1f} GB")
    else:
        print("   ❌ 未检测到GPU")

def main():
    """主函数"""
    print("=" * 60)
    print("🐾 Oxford-IIIT Pet Dataset VAE+LDM 训练环境设置")
    print("=" * 60)
    
    check_gpu()
    
    if setup_kaggle_environment():
        print("\n🎯 环境设置成功！准备开始训练...")
    else:
        print("\n💥 环境设置失败，请检查错误信息")

if __name__ == "__main__":
    main() 
