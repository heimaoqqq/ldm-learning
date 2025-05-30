import os
import yaml
import torch
import torch.nn.functional as F
import sys
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from typing import Dict, Any, List, Optional

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VAE'))

from ldm import LatentDiffusionModel
# 直接从VAE目录导入dataset功能
from dataset import build_dataloader

def denormalize(x):
    """反归一化图像，从[-1,1]转换到[0,1]"""
    return (x * 0.5 + 0.5).clamp(0, 1)

def generate_class_samples(
    model: LatentDiffusionModel,
    device: torch.device,
    config: Dict[str, Any],
    class_id: int,
    num_samples: int = 16,
    save_path: Optional[str] = None,
) -> torch.Tensor:
    """
    为指定类别生成样本
    """
    model.eval()
    
    with torch.no_grad():
        class_labels = torch.tensor([class_id] * num_samples, device=device)
        
        generated_images = model.sample(
            batch_size=num_samples,
            class_labels=class_labels,
            num_inference_steps=config['inference']['num_inference_steps'],
            guidance_scale=config['inference']['guidance_scale'],
            eta=config['inference']['eta'],
        )
        
        # 反归一化
        generated_images = denormalize(generated_images)
        
        if save_path:
            # 创建图像网格
            grid = vutils.make_grid(generated_images, nrow=4, normalize=False, padding=2)
            vutils.save_image(grid, save_path)
            print(f"类别 {class_id} 的生成图像已保存: {save_path}")
        
        return generated_images

def generate_all_classes(
    model: LatentDiffusionModel,
    device: torch.device,
    config: Dict[str, Any],
    output_dir: str,
    num_samples_per_class: int = 8,
):
    """
    为所有类别生成样本
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_classes = config['unet']['num_classes']
    
    print(f"为 {num_classes} 个类别生成样本，每类 {num_samples_per_class} 个...")
    
    all_images = []
    for class_id in tqdm(range(num_classes), desc="生成类别样本"):
        save_path = os.path.join(output_dir, f"class_{class_id:03d}.png")
        
        images = generate_class_samples(
            model, device, config, class_id, 
            num_samples_per_class, save_path
        )
        
        # 只保留前4个样本用于总览
        all_images.append(images[:4])
    
    # 创建所有类别的总览
    if all_images:
        all_images = torch.cat(all_images, dim=0)
        overview_grid = vutils.make_grid(all_images, nrow=4, normalize=False, padding=2)
        overview_path = os.path.join(output_dir, "all_classes_overview.png")
        vutils.save_image(overview_grid, overview_path)
        print(f"所有类别总览已保存: {overview_path}")

def interpolation_demo(
    model: LatentDiffusionModel,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    output_dir: str,
    num_steps: int = 10,
):
    """
    潜在空间插值演示
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    # 获取两个真实图像进行插值
    with torch.no_grad():
        for images, labels in val_loader:
            if images.shape[0] >= 2:
                images = images[:2].to(device)
                labels = labels[:2].to(device)
                break
        
        # 进行插值
        interpolated_images = model.interpolate(
            images1=images[0:1],
            images2=images[1:2],
            num_steps=num_steps,
        )
        
        # 反归一化
        interpolated_images = denormalize(interpolated_images)
        original_images = denormalize(images)
        
        # 创建插值序列图像
        all_images = [original_images[0:1]]  # 起始图像
        all_images.extend([img for img in interpolated_images])
        all_images.append(original_images[1:2])  # 结束图像
        
        all_images = torch.cat(all_images, dim=0)
        
        # 保存插值序列
        grid = vutils.make_grid(all_images, nrow=len(all_images), normalize=False, padding=2)
        save_path = os.path.join(output_dir, "interpolation_demo.png")
        vutils.save_image(grid, save_path)
        print(f"插值演示已保存: {save_path}")

def guidance_scale_comparison(
    model: LatentDiffusionModel,
    device: torch.device,
    config: Dict[str, Any],
    output_dir: str,
    class_id: int = 0,
    guidance_scales: List[float] = [1.0, 3.0, 5.0, 7.5, 10.0, 15.0],
):
    """
    不同引导强度的比较
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    # 固定种子以确保可重现性
    generator = torch.Generator(device=device).manual_seed(42)
    
    # 使用相同的初始噪声
    latent_shape = model.get_latent_shape(config['dataset']['image_size'])
    initial_latents = torch.randn((1, *latent_shape), generator=generator, device=device)
    
    all_images = []
    labels = []
    
    for guidance_scale in guidance_scales:
        with torch.no_grad():
            class_labels = torch.tensor([class_id], device=device)
            
            generated_image = model.sample(
                batch_size=1,
                class_labels=class_labels,
                num_inference_steps=config['inference']['num_inference_steps'],
                guidance_scale=guidance_scale,
                eta=config['inference']['eta'],
            )
            
            # 反归一化
            generated_image = denormalize(generated_image)
            all_images.append(generated_image)
            labels.append(f"CFG={guidance_scale}")
    
    # 创建比较图像
    all_images = torch.cat(all_images, dim=0)
    grid = vutils.make_grid(all_images, nrow=len(guidance_scales), normalize=False, padding=2)
    
    save_path = os.path.join(output_dir, f"guidance_comparison_class_{class_id}.png")
    vutils.save_image(grid, save_path)
    print(f"引导强度比较已保存: {save_path}")

def sampling_steps_comparison(
    model: LatentDiffusionModel,
    device: torch.device,
    config: Dict[str, Any],
    output_dir: str,
    class_id: int = 0,
    step_counts: List[int] = [10, 20, 30, 50, 100],
):
    """
    不同采样步数的比较
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    # 固定种子
    generator = torch.Generator(device=device).manual_seed(42)
    latent_shape = model.get_latent_shape(config['dataset']['image_size'])
    initial_latents = torch.randn((1, *latent_shape), generator=generator, device=device)
    
    all_images = []
    
    for num_steps in step_counts:
        with torch.no_grad():
            class_labels = torch.tensor([class_id], device=device)
            
            generated_image = model.sample(
                batch_size=1,
                class_labels=class_labels,
                num_inference_steps=num_steps,
                guidance_scale=config['inference']['guidance_scale'],
                eta=config['inference']['eta'],
            )
            
            generated_image = denormalize(generated_image)
            all_images.append(generated_image)
    
    # 创建比较图像
    all_images = torch.cat(all_images, dim=0)
    grid = vutils.make_grid(all_images, nrow=len(step_counts), normalize=False, padding=2)
    
    save_path = os.path.join(output_dir, f"steps_comparison_class_{class_id}.png")
    vutils.save_image(grid, save_path)
    print(f"采样步数比较已保存: {save_path}")

def evaluate_reconstruction(
    model: LatentDiffusionModel,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    output_dir: str,
    num_samples: int = 16,
):
    """
    评估重建质量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        # 获取真实图像
        for images, labels in val_loader:
            if images.shape[0] >= num_samples:
                images = images[:num_samples].to(device)
                break
        
        # 通过VAE重建
        reconstructed = model.vae(images)[0]
        
        # 反归一化
        original_images = denormalize(images)
        reconstructed_images = denormalize(reconstructed)
        
        # 创建比较图像
        comparison = []
        for i in range(min(8, num_samples)):  # 最多显示8对
            comparison.extend([original_images[i], reconstructed_images[i]])
        
        comparison = torch.stack(comparison)
        grid = vutils.make_grid(comparison, nrow=2, normalize=False, padding=2)
        
        save_path = os.path.join(output_dir, "reconstruction_comparison.png")
        vutils.save_image(grid, save_path)
        print(f"重建质量评估已保存: {save_path}")
        
        # 计算重建损失
        mse_loss = F.mse_loss(reconstructed, images).item()
        print(f"重建MSE损失: {mse_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="LDM推理脚本")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--model_path", type=str, default="ldm_models/best_model", help="模型路径")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="输出目录")
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["all", "generate", "interpolate", "guidance", "steps", "reconstruct"],
                        help="推理模式")
    parser.add_argument("--class_id", type=int, default=0, help="指定类别ID")
    parser.add_argument("--num_samples", type=int, default=16, help="生成样本数量")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化模型
    print("加载LDM模型...")
    
    # 处理VAE路径
    vae_path = config['vae']['model_path']
    if vae_path.startswith('../'):
        vae_path = os.path.join(os.path.dirname(__file__), vae_path)
    
    model = LatentDiffusionModel(
        vae_config=config['vae'],
        unet_config=config['unet'],
        scheduler_config=config['diffusion'],
        vae_path=vae_path,
        freeze_vae=config['vae']['freeze'],
        use_cfg=config['training']['use_cfg'],
        cfg_dropout_prob=config['training']['cfg_dropout_prob'],
    ).to(device)
    
    # 加载训练好的模型
    if os.path.exists(args.model_path):
        model.load_pretrained(args.model_path)
    else:
        print(f"警告: 模型路径不存在: {args.model_path}")
        print("使用未训练的模型进行推理...")
    
    # 加载验证数据
    print("加载验证数据...")
    _, val_loader, _, _ = build_dataloader(
        config['dataset']['root_dir'],
        batch_size=config['inference']['sample_batch_size'],
        num_workers=0,
        shuffle_train=False,
        shuffle_val=False,
        val_split=config['dataset']['val_split']
    )
    
    # 执行推理
    if args.mode == "all" or args.mode == "generate":
        print("生成类别样本...")
        generate_all_classes(model, device, config, 
                            os.path.join(args.output_dir, "generated_samples"),
                            config['inference']['num_samples_per_class'])
    
    if args.mode == "all" or args.mode == "interpolate":
        print("执行插值演示...")
        interpolation_demo(model, device, val_loader, config,
                          os.path.join(args.output_dir, "interpolation"))
    
    if args.mode == "all" or args.mode == "guidance":
        print("比较不同引导强度...")
        guidance_scale_comparison(model, device, config,
                                 os.path.join(args.output_dir, "guidance_comparison"),
                                 args.class_id)
    
    if args.mode == "all" or args.mode == "steps":
        print("比较不同采样步数...")
        sampling_steps_comparison(model, device, config,
                                 os.path.join(args.output_dir, "steps_comparison"),
                                 args.class_id)
    
    if args.mode == "all" or args.mode == "reconstruct":
        print("评估重建质量...")
        evaluate_reconstruction(model, device, val_loader,
                               os.path.join(args.output_dir, "reconstruction"))
    
    print(f"推理完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 
