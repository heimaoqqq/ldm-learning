#!/usr/bin/env python3
"""
VQ-VAE环境设置脚本
基于CompVis/taming-transformers实现
用于256*256彩色图像数据集训练
"""

import os
import subprocess
import sys
import torch

def check_environment():
    """检查Python和CUDA环境"""
    print("=== 检查环境 ===")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

def install_dependencies():
    """安装必要的依赖包"""
    print("=== 安装依赖包 ===")
    
    # 基础依赖
    packages = [
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "torchaudio>=0.8.0",
        "pytorch-lightning>=1.5.0",
        "omegaconf>=2.1.0",
        "einops>=0.3.0",
        "transformers>=4.0.0",
        "pillow>=8.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.3.0",
        "scipy>=1.7.0",
        "scikit-image>=0.18.0",
        "tensorboard>=2.7.0",
        "wandb>=0.12.0",
        "lpips>=0.1.4",
        "albumentations>=1.1.0",
        "opencv-python>=4.5.0",
        "tqdm>=4.62.0"
    ]
    
    for package in packages:
        try:
            print(f"安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"安装 {package} 失败: {e}")
            continue
    
    print("依赖安装完成!")
    print()

def clone_taming_transformers():
    """克隆taming-transformers仓库"""
    print("=== 克隆taming-transformers仓库 ===")
    
    if not os.path.exists("taming-transformers"):
        try:
            subprocess.check_call([
                "git", "clone", "https://github.com/CompVis/taming-transformers.git"
            ])
            print("成功克隆taming-transformers仓库")
        except subprocess.CalledProcessError:
            print("克隆失败，请手动下载：")
            print("https://github.com/CompVis/taming-transformers")
            return False
    else:
        print("taming-transformers仓库已存在")
    
    # 安装taming-transformers
    try:
        original_dir = os.getcwd()
        os.chdir("taming-transformers")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        os.chdir(original_dir)
        print("taming-transformers安装完成")
    except subprocess.CalledProcessError as e:
        print(f"安装taming-transformers失败: {e}")
        return False
    
    return True

def create_directories():
    """创建必要的目录结构"""
    print("=== 创建目录结构 ===")
    
    directories = [
        "data",
        "configs",
        "checkpoints",
        "logs",
        "results",
        "scripts"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")
    
    print("目录结构创建完成!")
    print()

def main():
    """主函数"""
    print("VQ-VAE训练环境设置")
    print("=" * 50)
    
    check_environment()
    install_dependencies()
    
    if clone_taming_transformers():
        create_directories()
        print("✅ 环境设置完成!")
        print("\n下一步:")
        print("1. 将您的256*256彩色图像数据集放入 data/ 文件夹")
        print("2. 运行 python train_vqvae.py 开始训练")
    else:
        print("❌ 环境设置失败，请检查网络连接")

if __name__ == "__main__":
    main() 
