import torch
import torch.nn as nn
import os
import argparse
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from diffusers import AutoencoderKL
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import shutil
from tqdm import tqdm

# --- Helper Functions and Classes ---

def check_and_install_dependencies():
    """检查并安装必要的依赖，并强制使用与Kaggle环境兼容的PyTorch和NumPy版本。"""
    try:
        import torch
        import numpy as np
        import diffusers
        import torch_fidelity
        
        # 验证版本
        torch_ok = torch.__version__.startswith("2.1.2")
        # 确保NumPy版本是1.x
        numpy_ok = int(np.__version__.split('.')[0]) < 2
        
        if torch_ok and numpy_ok:
             print("✅ 所有依赖已安装且版本兼容。")
             return
        else:
            # 如果版本不对，触发重新安装流程
            error_msg = []
            if not torch_ok: error_msg.append(f"PyTorch (当前: {torch.__version__}, 需要: 2.1.2.x)")
            if not numpy_ok: error_msg.append(f"NumPy (当前: {np.__version__}, 需要: <2.0)")
            raise ImportError(f"版本不匹配: {', '.join(error_msg)}")

    except ImportError as e:
        print(f"📦 检测到依赖问题: {e}")
        print("   将开始修复流程...")
        
        # 定义安装命令
        # 强制安装 NumPy < 2.0 来解决 ABI 兼容性问题
        pytorch_install_cmd = "pip install --upgrade --force-reinstall torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 -q"
        other_deps_install_cmd = "pip install diffusers transformers accelerate scikit-image torch-fidelity tqdm numpy==1.26.4 -q"

        print("   第一步: 强制重装与Kaggle CUDA 12.1 兼容的PyTorch版本...")
        os.system(pytorch_install_cmd)
        
        print("   第二步: 安装其他必要的库并固定NumPy版本到1.26.4...")
        os.system(other_deps_install_cmd)
        
        print("\n" + "="*50)
        print("✅ 依赖修复完成。")
        print("🛑 请务必重新运行此脚本/单元格以使更改生效。")
        print("="*50)
        
        import sys
        sys.exit(0)

class KaggleOxfordPetDataset(torch.utils.data.Dataset):
    """
    适配Kaggle路径的Oxford-IIIT Pet数据集类
    (从训练脚本中复制，用于加载数据)
    """
    def __init__(self, images_dir, annotation_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        
        self.samples = []
        try:
            with open(annotation_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        image_id = parts[0]
                        class_id = int(parts[1]) - 1
                        self.samples.append((image_id, class_id))
            print(f"加载了 {len(self.samples)} 个样本")
        except FileNotFoundError:
            print(f"❌ 标注文件未找到: {annotation_file}")
            # 如果标注文件不存在，尝试直接扫描图像目录
            image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
            image_files.extend(glob.glob(os.path.join(images_dir, "*.png")))
            for img_path in image_files:
                image_id = os.path.splitext(os.path.basename(img_path))[0]
                self.samples.append((image_id, 0)) # 使用默认类别0
            print(f"⚠️  标注文件未找到，已直接从目录加载 {len(self.samples)} 张图片。")

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_id, class_id = self.samples[idx]
        img_path = os.path.join(self.images_dir, f"{image_id}.jpg")
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
            placeholder = torch.zeros((3, 256, 256))
            return placeholder, class_id

@torch.no_grad()
def test_vae(args):
    """主测试函数"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 使用设备: {device}")

    # 1. 加载模型
    print(f"📦 加载预训练的 Stable Diffusion VAE 结构...")
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        subfolder="vae"
    )
    
    print(f"💾 加载您微调过的VAE权重从: {args.checkpoint_path}")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if 'vae_state_dict' in checkpoint:
            vae.load_state_dict(checkpoint['vae_state_dict'])
        else:
            # 兼容直接保存state_dict的情况
            vae.load_state_dict(checkpoint)
        print("✅ 权重加载成功!")
    except Exception as e:
        print(f"🔥 加载权重失败: {e}")
        return

    vae = vae.to(device).eval()

    # 2. 准备数据
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 智能判断标注文件路径
    if os.path.basename(args.annotations_dir) == 'annotations':
        annotation_file = os.path.join(args.annotations_dir, "list.txt")
    else: # 假设 annotations_dir 就是文件本身
        annotation_file = args.annotations_dir

    print(f"📁 准备数据集...")
    dataset = KaggleOxfordPetDataset(
        images_dir=args.images_dir,
        annotation_file=annotation_file,
        transform=transform
    )
    
    if len(dataset) == 0:
        print("❌ 数据集为空，请检查路径。")
        return

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.num_samples, shuffle=True
    )

    # 3. 执行重建和评估
    try:
        original_images, _ = next(iter(dataloader))
    except StopIteration:
        print("❌ 数据加载器为空，无法获取图像。")
        return
        
    original_images = original_images.to(device)
    
    print("✨ 执行重建...")
    reconstructed_images = vae.decode(vae.encode(original_images).latent_dist.sample()).sample

    # 反归一化以便显示和计算指标
    def denormalize(tensor):
        return torch.clamp((tensor + 1) / 2, 0, 1)

    original_images_denorm = denormalize(original_images).cpu().permute(0, 2, 3, 1).numpy()
    reconstructed_images_denorm = denormalize(reconstructed_images).cpu().permute(0, 2, 3, 1).numpy()
    
    # 4. 计算指标
    print("\n--- 质量评估指标 ---")
    all_psnr = []
    all_ssim = []
    for i in range(args.num_samples):
        orig_img = original_images_denorm[i]
        recon_img = reconstructed_images_denorm[i]
        
        # data_range 是像素值的范围
        current_psnr = psnr(orig_img, recon_img, data_range=1.0)
        current_ssim = ssim(orig_img, recon_img, data_range=1.0, channel_axis=-1, win_size=7)
        
        all_psnr.append(current_psnr)
        all_ssim.append(current_ssim)
        print(f"  样本 {i+1}: PSNR = {current_psnr:.2f} dB, SSIM = {current_ssim:.4f}")
        
    print(f"------------------------")
    print(f"📊 平均值: PSNR = {np.mean(all_psnr):.2f} dB, SSIM = {np.mean(all_ssim):.4f}")
    print("------------------------")
    print("💡 PSNR越高越好 (通常 > 30dB 表示不错)。SSIM 越接近1越好。")


    # 5. 可视化对比
    print(f"🖼️ 生成对比图像: {args.output_path}")
    fig, axes = plt.subplots(2, args.num_samples, figsize=(args.num_samples * 3, 6))
    
    for i in range(args.num_samples):
        # 原图
        axes[0, i].imshow(original_images_denorm[i])
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # 重建图
        axes[1, i].imshow(reconstructed_images_denorm[i])
        axes[1, i].set_title(f'Recon {i+1}')
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. FID评估 (如果启用)
    if args.calc_fid:
        print("\n--- FID (Fréchet Inception Distance) 评估 ---")
        print(f"准备 {args.num_fid_samples} 个样本进行评估，这可能需要一些时间...")
        
        # 导入 torch_fidelity
        try:
            from torch_fidelity.metrics import calculate_metrics
        except ImportError:
            print("🔥 torch-fidelity 未安装，无法计算FID。请重新运行脚本以自动安装。")
            return

        # 创建临时目录
        real_dir = './fid_real_images'
        recon_dir = './fid_recon_images'
        if os.path.exists(real_dir): shutil.rmtree(real_dir)
        if os.path.exists(recon_dir): shutil.rmtree(recon_dir)
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)

        try:
            # 确保我们不会请求比数据集更多的样本
            num_fid_samples = min(args.num_fid_samples, len(dataset))
            if num_fid_samples < args.num_fid_samples:
                print(f"⚠️  请求的FID样本数 ({args.num_fid_samples}) 大于数据集中的样本数 ({len(dataset)})。")
                print(f"将使用 {num_fid_samples} 个样本进行计算。")

            # 我们需要对数据集进行子集采样
            fid_subset = torch.utils.data.Subset(dataset, list(range(num_fid_samples)))

            fid_loader = torch.utils.data.DataLoader(
                fid_subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2
            )

            # 保存图像到目录
            img_counter = 0
            pbar = tqdm(fid_loader, desc="生成FID评估样本")
            for batch_images, _ in pbar:
                batch_images = batch_images.to(device)
                
                # 重建
                recon_batch = vae.decode(vae.encode(batch_images).latent_dist.sample()).sample
                
                # 反归一化
                orig_denorm_batch = denormalize(batch_images).cpu()
                recon_denorm_batch = denormalize(recon_batch).cpu()

                for i in range(batch_images.size(0)):
                    orig_pil = transforms.ToPILImage()(orig_denorm_batch[i])
                    recon_pil = transforms.ToPILImage()(recon_denorm_batch[i])
                    
                    orig_pil.save(os.path.join(real_dir, f'img_{img_counter:05d}.png'))
                    recon_pil.save(os.path.join(recon_dir, f'img_{img_counter:05d}.png'))
                    
                    img_counter += 1
            
            print(f"✅ 已生成 {img_counter} 张真实图像和重建图像。")
            print("⏳ 开始计算FID分数...")
            
            metrics = calculate_metrics(
                input1=real_dir, 
                input2=recon_dir, 
                cuda=torch.cuda.is_available(), 
                fid=True,
                verbose=False
            )
            
            fid_score = metrics['frechet_inception_distance']
            print(f"------------------------")
            print(f"📊 FID 分数: {fid_score:.4f}")
            print("------------------------")
            print("💡 FID越低越好。表示重建图像的分布与真实图像的分布更接近。")

        finally:
            # 清理临时目录
            print("🧹 清理临时文件...")
            if os.path.exists(real_dir): shutil.rmtree(real_dir)
            if os.path.exists(recon_dir): shutil.rmtree(recon_dir)
            print("✅ 清理完成。")

    print("✅ 测试完成!")


if __name__ == '__main__':
    # 首先检查依赖
    check_and_install_dependencies()

    # --- 为Kaggle环境定制的批量测试配置 ---
    # 您可以在这里添加或修改要测试的模型
    test_configs = [
        {
            "checkpoint_path": "/kaggle/input/fid-test/vae_finetuned_epoch_1.pth",
            "output_path": "./epoch_1_quality_report.png",
            "description": "第1轮训练的模型"
        },
        {
            "checkpoint_path": "/kaggle/input/fid-test/vae_finetuned_epoch_19.pth",
            "output_path": "./epoch_19_quality_report.png",
            "description": "第19轮训练的模型"
        }
    ]

    # --- 通用参数 ---
    # 请确保这些路径指向您的数据集
    base_args = {
        "images_dir": "/kaggle/input/dataset-test/images/images",
        "annotations_dir": "/kaggle/input/dataset-test/annotations/annotations",
        "image_size": 256,
        "num_samples": 8,      # 可视化对比的样本数
        "calc_fid": True,      # 自动计算FID
        "num_fid_samples": 1000, # 用于FID计算的样本数
        "batch_size": 16,      # 处理FID样本时的批次大小
    }
    
    # --- 开始执行批量测试 ---
    for i, config in enumerate(test_configs):
        print("\n" + "="*60)
        print(f"🔬 开始测试配置 #{i+1}: {config['description']}")
        print(f"   模型路径: {config['checkpoint_path']}")
        print("="*60 + "\n")

        # 合并通用参数和当前测试配置
        current_args_dict = {**base_args, **config}
        
        # 将字典转换为 argparse.Namespace 对象，以兼容 test_vae 函数
        current_args = argparse.Namespace(**current_args_dict)
        
        try:
            test_vae(current_args)
        except Exception as e:
            print(f"🔥 测试配置 #{i+1} 时发生错误: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "*"*60)
    print("🎉 所有评估任务已完成!")
    print("请在输出目录中查看生成的报告图片。")
    print("*"*60) 
