#!/usr/bin/env python3
"""
在Kaggle环境中设置VAE训练环境
"""

import os
import sys
import shutil
import subprocess
import time

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
    
    # 安装VAE训练所需的主要依赖
    print("安装PyTorch和相关依赖...")
    deps = [
        "omegaconf==2.1.1",
        "pytorch-lightning==2.0.9",
        "einops==0.4.1",
        "torch-fidelity==0.3.0",
        "transformers==4.19.2",
        "kornia==0.6.5",
        "torchmetrics>=0.6.0",
        "lpips",
        "albumentations>=1.1.0"
    ]
    for dep in deps:
        run_cmd(f"pip install {dep} -q")
    
    # 步骤3: 设置latent-diffusion
    print_step("3", "设置latent-diffusion库")
    
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
    
    # 步骤6: 查找并分析数据集
    print_step("6", "分析数据集")
    
    # 查找图像目录
    images_paths = [
        os.path.join(dataset_path, "images"),
        os.path.join(dataset_path, "data", "images"),
        dataset_path
    ]
    
    images_dir = None
    for path in images_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # 检查是否包含图像文件
            image_files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                images_dir = path
                print(f"\033[0;32m找到图像目录: {path} (包含 {len(image_files)} 张图像)\033[0m")
                break
    
    if not images_dir:
        print("\033[0;31m未找到有效的图像目录。尝试递归搜索...\033[0m")
        # 递归搜索所有子目录中的jpg文件
        for root, dirs, files in os.walk(dataset_path):
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(image_files) > 0:
                images_dir = root
                print(f"\033[0;32m找到图像目录: {root} (包含 {len(image_files)} 张图像)\033[0m")
                break
    
    if not images_dir:
        print("\033[0;31m未找到任何图像文件。请检查数据集结构。\033[0m")
        return False
    
    # 更新VAE配置文件中的数据路径
    vae_config_path = os.path.join(vae_dir, "vae_config.yaml")
    if os.path.exists(vae_config_path):
        try:
            with open(vae_config_path, 'r') as f:
                config_content = f.read()
            
            # 替换数据路径
            config_content = config_content.replace(
                "data_root: G:\\Latent Diffusion\\data\\pet_dataset\\images", 
                f"data_root: {images_dir}"
            )
            
            with open(vae_config_path, 'w') as f:
                f.write(config_content)
            
            print(f"✓ 已更新配置文件中的数据路径: {images_dir}")
        except Exception as e:
            print(f"\033[0;31m更新配置文件失败: {e}\033[0m")
    
    # 步骤7: 总结和指导
    print_step("7", "环境设置完成")
    
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
