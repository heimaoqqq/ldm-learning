#!/usr/bin/env python3
"""
在Kaggle环境中设置VAE训练环境
"""

import os
import sys
import shutil
import subprocess
import time
import glob

def print_step(step, desc):
    """打印彩色步骤信息"""
    print(f"\n\033[1;36m[{step}]\033[0m \033[1;33m{desc}\033[0m")
    time.sleep(0.5)

def run_cmd(cmd):
    """运行命令并打印输出"""
    print(f"\033[1;30m$ {cmd}\033[0m")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"\033[0;31m{result.stderr}\033[0m")
    return result.returncode == 0

def check_path(path):
    """检查路径是否存在"""
    if os.path.exists(path):
        print(f"\033[0;32m✓ 已找到: {path}\033[0m")
        return True
    else:
        print(f"\033[0;31m✗ 未找到: {path}\033[0m")
        return False

def find_pet_images(base_path):
    """查找宠物图像文件夹"""
    # 常见的图像目录名称
    potential_dirs = ["images", "photos", "pictures", "imgs", "data"]
    
    # 首先检查常见目录名
    for dirname in potential_dirs:
        path = os.path.join(base_path, dirname)
        if os.path.isdir(path):
            # 检查目录中是否有jpg图片
            jpg_files = glob.glob(os.path.join(path, "*.jpg"))
            if jpg_files:
                return path, len(jpg_files)
    
    # 直接检查根目录下的jpg文件
    jpg_files = glob.glob(os.path.join(base_path, "*.jpg"))
    if jpg_files:
        return base_path, len(jpg_files)
    
    # 递归搜索所有子目录
    best_dir = None
    max_images = 0
    
    for root, dirs, files in os.walk(base_path):
        # 跳过annotations目录，因为它通常包含掩码而不是实际图像
        if "annotations" in root:
            continue
            
        jpg_files = [f for f in files if f.lower().endswith(".jpg")]
        if len(jpg_files) > max_images:
            max_images = len(jpg_files)
            best_dir = root
    
    if best_dir:
        return best_dir, max_images
    
    return None, 0

def main():
    """主函数"""
    print("\n" + "="*50)
    print("\033[1;32mVAE训练环境设置 (Kaggle版)\033[0m")
    print("="*50)
    
    # 步骤1: 检查Kaggle环境
    print_step("1", "检查Kaggle环境")
    
    kaggle_dir = "/kaggle"
    if not check_path(kaggle_dir):
        print("\033[0;31m错误: 未检测到Kaggle环境。请在Kaggle Notebook中运行此脚本。\033[0m")
        return False
    
    # 查找数据集
    print("查找Oxford-IIIT Pet数据集...")
    dataset_paths = [
        "/kaggle/input/the-oxfordiiit-pet-dataset",
        "/kaggle/input/oxford-iiit-pet-dataset",
        "/kaggle/input/cats-and-dogs-breeds-classification-oxford-dataset"
    ]
    
    dataset_path = None
    for path in dataset_paths:
        if check_path(path):
            dataset_path = path
            print(f"\033[0;32m已找到数据集路径: {dataset_path}\033[0m")
            break
    
    if not dataset_path:
        print("\033[0;31m未找到数据集。请确保添加了Oxford-IIIT Pet数据集。\033[0m")
        return False
    
    # 步骤2: 安装依赖
    print_step("2", "安装依赖")
    
    # 使用预编译的包避免编译问题
    print("安装PyTorch和相关依赖...")
    run_cmd("pip install torch torchvision torchaudio -q")
    run_cmd("pip install omegaconf==2.1.1 pytorch-lightning==2.0.9 einops==0.4.1 -q")
    
    # 安装其他依赖
    print("安装其他依赖...")
    run_cmd("pip install kornia==0.6.5 torchmetrics>=0.6.0 lpips -q")
    run_cmd("pip install albumentations>=1.1.0 -q")
    
    # 步骤3: 设置latent-diffusion和taming-transformers
    print_step("3", "设置必要的库")
    
    # 克隆并安装taming-transformers（ldm依赖）
    print("克隆taming-transformers...")
    taming_dir = "/kaggle/working/taming-transformers"
    run_cmd(f"git clone --depth 1 https://github.com/CompVis/taming-transformers.git {taming_dir}")
    
    if os.path.exists(taming_dir):
        print("✓ 成功克隆taming-transformers")
        # 将taming-transformers添加到PYTHONPATH
        sys.path.append(taming_dir)
        # 创建空的setup.py以便安装
        with open(os.path.join(taming_dir, "setup.py"), "w") as f:
            f.write("""
from setuptools import setup, find_packages
setup(
    name="taming-transformers",
    version="0.0.1",
    packages=find_packages(),
)
""")
        # 安装taming-transformers
        print("安装taming-transformers...")
        run_cmd(f"pip install -e {taming_dir} -q")
    else:
        print("\033[0;31m克隆taming-transformers失败\033[0m")
    
    # 查找或克隆latent-diffusion
    ld_paths = [
        "/kaggle/input/latent-diffusion",
        "/kaggle/input/latentdiffusion",
        "/kaggle/input/compvis-latent-diffusion"
    ]
    
    ldm_found = False
    for ld_path in ld_paths:
        if check_path(ld_path):
            print(f"找到latent-diffusion源码: {ld_path}")
            ldm_found = True
            
            # 创建目标目录
            working_ldm = "/kaggle/working/latent-diffusion"
            os.makedirs(working_ldm, exist_ok=True)
            
            # 复制必要文件
            print("复制必要文件到工作目录...")
            dirs_to_copy = ["ldm"]
            for dir_name in dirs_to_copy:
                src = os.path.join(ld_path, dir_name)
                dst = os.path.join(working_ldm, dir_name)
                if os.path.exists(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                    print(f"✓ 已复制: {dir_name}")
            
            # 复制main.py
            main_py = os.path.join(ld_path, "main.py")
            if os.path.exists(main_py):
                shutil.copy2(main_py, working_ldm)
                print("✓ 已复制: main.py")
            break
    
    if not ldm_found:
        print("\033[0;31m未找到latent-diffusion。尝试从GitHub克隆...\033[0m")
        
        # 尝试从GitHub克隆
        print("尝试从GitHub克隆latent-diffusion...")
        os.makedirs("/kaggle/working/latent-diffusion", exist_ok=True)
        clone_cmd = "git clone --depth 1 https://github.com/CompVis/latent-diffusion.git /kaggle/working/latent-diffusion-temp"
        if run_cmd(clone_cmd):
            # 复制必要文件
            print("复制必要文件到工作目录...")
            if os.path.exists("/kaggle/working/latent-diffusion-temp/ldm"):
                if os.path.exists("/kaggle/working/latent-diffusion/ldm"):
                    shutil.rmtree("/kaggle/working/latent-diffusion/ldm")
                shutil.copytree("/kaggle/working/latent-diffusion-temp/ldm", "/kaggle/working/latent-diffusion/ldm")
            
            if os.path.exists("/kaggle/working/latent-diffusion-temp/main.py"):
                shutil.copy2("/kaggle/working/latent-diffusion-temp/main.py", "/kaggle/working/latent-diffusion/")
            
            # 清理临时目录
            shutil.rmtree("/kaggle/working/latent-diffusion-temp")
            print("✓ 成功从GitHub克隆并设置latent-diffusion")
            ldm_found = True
        else:
            print("\033[0;31m从GitHub克隆失败。请手动添加latent-diffusion。\033[0m")
    
    # 创建符号链接，确保taming可被导入
    print("创建taming符号链接...")
    taming_target = "/kaggle/working/latent-diffusion/taming"
    if os.path.exists(taming_dir) and not os.path.exists(taming_target):
        if os.path.islink(taming_target):
            os.unlink(taming_target)
        try:
            # 尝试使用符号链接
            os.symlink(os.path.join(taming_dir, "taming"), taming_target)
            print("✓ 已创建taming符号链接")
        except:
            # 如果符号链接失败，尝试复制
            if os.path.exists(os.path.join(taming_dir, "taming")):
                shutil.copytree(os.path.join(taming_dir, "taming"), taming_target)
                print("✓ 已复制taming目录")
    
    # 步骤4: 创建VAE目录结构
    print_step("4", "创建VAE目录结构")
    
    # 在working目录下创建VAE目录
    vae_dir = "/kaggle/working/VAE"
    os.makedirs(vae_dir, exist_ok=True)
    os.makedirs(os.path.join(vae_dir, "logs", "checkpoints"), exist_ok=True)
    print(f"✓ 已创建VAE目录: {vae_dir}")
    
    # 步骤5: 复制VAE相关文件
    print_step("5", "复制VAE代码文件")
    
    # 复制当前目录中的Python文件到Kaggle的VAE目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files_to_copy = [
        "train_vae_kaggle.py", 
        "pet_dataset.py", 
        "vae_config.yaml", 
        "__init__.py"
    ]
    
    for file in files_to_copy:
        src = os.path.join(current_dir, file)
        dst = os.path.join(vae_dir, file)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✓ 已复制: {file}")
        else:
            print(f"✗ 未找到文件: {file}")
    
    # 步骤6: 分析数据集
    print_step("6", "分析数据集")
    
    # 使用专门的宠物图像查找函数
    images_dir, image_count = find_pet_images(dataset_path)
    
    if images_dir and image_count > 0:
        print(f"\033[0;32m找到宠物图像目录: {images_dir} (包含 {image_count} 张图像)\033[0m")
    else:
        # 尝试一些已知的路径模式
        known_patterns = [
            os.path.join(dataset_path, "images", "*.jpg"),
            os.path.join(dataset_path, "images", "images", "*.jpg")
        ]
        
        for pattern in known_patterns:
            files = glob.glob(pattern)
            if files:
                images_dir = os.path.dirname(files[0])
                image_count = len(files)
                print(f"\033[0;32m通过模式匹配找到图像目录: {images_dir} (包含 {image_count} 张图像)\033[0m")
                break
    
    if not images_dir or image_count == 0:
        print("\033[0;31m无法找到宠物图像。请手动检查数据集结构。\033[0m")
        
        # 列出数据集目录以帮助用户定位
        print("\n数据集目录结构:")
        cmd = f"find {dataset_path} -type d | sort"
        run_cmd(cmd)
        
        print("\n图像文件示例:")
        cmd = f"find {dataset_path} -name '*.jpg' | head -5"
        run_cmd(cmd)
        return False
    
    # 更新VAE配置文件中的数据路径
    vae_config_path = os.path.join(vae_dir, "vae_config.yaml")
    if os.path.exists(vae_config_path):
        try:
            with open(vae_config_path, 'r') as f:
                config_content = f.read()
            
            # 替换数据路径 (处理Windows和Unix路径)
            config_content = config_content.replace(
                "data_root: G:\\Latent Diffusion\\data\\pet_dataset\\images", 
                f"data_root: {images_dir}"
            )
            config_content = config_content.replace(
                "data_root: /kaggle/input/the-oxfordiiit-pet-dataset/annotations/annotations/trimaps", 
                f"data_root: {images_dir}"
            )
            
            with open(vae_config_path, 'w') as f:
                f.write(config_content)
            
            print(f"✓ 已更新配置文件中的数据路径: {images_dir}")
        except Exception as e:
            print(f"\033[0;31m更新配置文件失败: {e}\033[0m")
    
    # 步骤7: 创建main.py文件
    print_step("7", "创建必要的辅助文件")
    
    # 确保main.py文件存在于latent-diffusion目录中
    main_py = "/kaggle/working/latent-diffusion/main.py"
    if not os.path.exists(main_py):
        print("创建基本的main.py文件...")
        main_py_content = """
import sys
import argparse
import pytorch_lightning as pl
from omegaconf import OmegaConf

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader

        self.wrap = wrap

    def prepare_data(self):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def setup(self, stage=None):
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def _val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

class ImageLogger:
    def __init__(self, batch_frequency=1000, max_images=8, increase_log_steps=True):
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [batch_frequency]
        self.increase_log_steps = increase_log_steps
        
    def __call__(self, pl_module, batch, batch_idx, split="train"):
        pass # 简化版实现
"""
        with open(main_py, 'w') as f:
            f.write(main_py_content)
        print("✓ 已创建main.py")
    
    # 步骤8: 创建修改版的autoencoder.py（移除taming依赖）
    print_step("8", "修改LDM代码")
    
    # 尝试修改autoencoder.py，移除taming依赖
    autoencoder_path = "/kaggle/working/latent-diffusion/ldm/models/autoencoder.py"
    if os.path.exists(autoencoder_path):
        print("修改autoencoder.py，减少taming依赖...")
        try:
            with open(autoencoder_path, 'r') as f:
                content = f.read()
            
            # 添加简单的VectorQuantizer实现
            if "from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer" in content:
                vq_impl = """
# 简化版的VectorQuantizer实现，替代taming依赖
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25, remap=None, unknown_index="random"):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.einsum('bd,dn->bn', z_flattened, self.embedding.weight.t())
        
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # Compute loss
        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
        
        # Straight through
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2)
        
        return z_q, loss, (None, None, min_encoding_indices)
"""
                # 替换taming导入
                content = content.replace(
                    "from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer", 
                    vq_impl
                )
                
                # 保存修改后的文件
                with open(autoencoder_path, 'w') as f:
                    f.write(content)
                print("✓ 已修改autoencoder.py，减少taming依赖")
        except Exception as e:
            print(f"\033[0;31m修改autoencoder.py失败: {e}\033[0m")
    
    # 步骤9: 总结和指导
    print_step("9", "环境设置完成")
    
    print("\n" + "="*50)
    print("\033[1;32mVAE训练环境已设置完成!\033[0m")
    print("="*50)
    
    print("\nVAE训练命令:")
    print(f"\033[1;33mpython /kaggle/working/VAE/train_vae_kaggle.py --data_root {images_dir}\033[0m")
    print(f"\n可选参数:")
    print(f"  --batch_size 32         # 批次大小，可根据GPU内存调整")
    print(f"  --max_epochs 100        # 最大训练轮数")
    print(f"  --precision 16-mixed    # 使用半精度训练")
    
    print("\n训练完成后，VAE模型将保存在:")
    print(f"\033[1;33m/kaggle/working/VAE/logs/checkpoints/\033[0m")
    
    print("\n如需查看训练日志，可使用TensorBoard:")
    print(f"\033[1;33m%load_ext tensorboard\033[0m")
    print(f"\033[1;33m%tensorboard --logdir /kaggle/working/VAE/logs\033[0m")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n\033[0;31m设置失败。请查看错误信息并手动解决问题。\033[0m")
        sys.exit(1) 
