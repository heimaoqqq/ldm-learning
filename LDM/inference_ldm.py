#!/usr/bin/env python3
"""
轻量化StableLDM推理脚本
快速生成高质量图像
"""

import os
import yaml
import torch
import torchvision.utils as vutils
import sys
from typing import Optional

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VAE'))

from stable_ldm import create_stable_ldm

def load_model(config_path: str, checkpoint_path: str, device: str = 'auto') -> torch.nn.Module:
    """加载训练好的模型"""
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设备配置
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"🚀 使用设备: {device}")
    
    # 创建模型
    model = create_stable_ldm(config).to(device)
    
    # 加载权重
    unet_path = os.path.join(checkpoint_path, 'unet.pth')
    if os.path.exists(unet_path):
        state_dict = torch.load(unet_path, map_location=device)
        model.unet.load_state_dict(state_dict)
        print(f"✅ 已加载U-Net权重: {unet_path}")
    else:
        print(f"⚠️ 未找到U-Net权重文件: {unet_path}")
    
    # 加载EMA权重（如果有）
    ema_path = os.path.join(checkpoint_path, 'ema.pth')
    if os.path.exists(ema_path) and hasattr(model, 'ema') and model.ema is not None:
        ema_state = torch.load(ema_path, map_location=device)
        model.ema.shadow = ema_state
        print(f"✅ 已加载EMA权重: {ema_path}")
    
    model.eval()
    return model, config, device

def generate_samples(
    model,
    config,
    device,
    num_samples_per_class: int = 4,
    num_classes_to_show: int = 8,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    use_ema: bool = False,
    output_dir: str = 'generated_samples'
):
    """生成样本图像"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎨 开始生成样本...")
    print(f"   推理步数: {num_inference_steps}")
    print(f"   CFG强度: {guidance_scale}")
    print(f"   使用EMA: {'✓' if use_ema else '✗'}")
    
    all_images = []
    all_labels = []
    
    total_classes = config['unet']['num_classes']
    classes_to_generate = min(num_classes_to_show, total_classes)
    
    with torch.no_grad():
        for class_id in range(classes_to_generate):
            print(f"   生成类别 {class_id+1}/{classes_to_generate}...")
            
            class_labels = torch.tensor([class_id] * num_samples_per_class, device=device)
            
            generated_images = model.sample(
                batch_size=num_samples_per_class,
                class_labels=class_labels,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                use_ema=use_ema,
                verbose=False
            )
            
            all_images.append(generated_images)
            all_labels.extend([class_id] * num_samples_per_class)
    
    # 拼接所有图像
    if all_images:
        all_images = torch.cat(all_images, dim=0)
        
        # 保存图像网格
        grid = vutils.make_grid(
            all_images, 
            nrow=num_samples_per_class, 
            normalize=False, 
            padding=2
        )
        
        grid_path = os.path.join(output_dir, 'sample_grid.png')
        vutils.save_image(grid, grid_path)
        print(f"✅ 样本网格已保存: {grid_path}")
        
        # 保存单个图像
        for i, (image, label) in enumerate(zip(all_images, all_labels)):
            single_path = os.path.join(output_dir, f'sample_class_{label}_idx_{i%num_samples_per_class}.png')
            vutils.save_image(image, single_path)
        
        print(f"✅ 共生成 {len(all_images)} 张图像")
        print(f"📁 保存目录: {output_dir}")

def generate_interpolation(
    model,
    config, 
    device,
    class_a: int,
    class_b: int,
    num_steps: int = 8,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    use_ema: bool = False,
    output_dir: str = 'generated_samples'
):
    """生成类别间插值"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔄 生成类别插值: {class_a} → {class_b}")
    
    # 创建插值权重
    weights = torch.linspace(0, 1, num_steps)
    
    interpolated_images = []
    
    with torch.no_grad():
        for i, w in enumerate(weights):
            print(f"   插值步骤 {i+1}/{num_steps} (权重: {w:.2f})")
            
            # 混合类别嵌入（需要修改模型以支持连续类别嵌入）
            # 这里简化为选择离散类别
            current_class = class_a if w < 0.5 else class_b
            class_labels = torch.tensor([current_class], device=device)
            
            generated_image = model.sample(
                batch_size=1,
                class_labels=class_labels,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                use_ema=use_ema,
                verbose=False
            )
            
            interpolated_images.append(generated_image)
    
    # 保存插值序列
    if interpolated_images:
        all_interp = torch.cat(interpolated_images, dim=0)
        
        # 水平排列
        grid = vutils.make_grid(all_interp, nrow=num_steps, normalize=False, padding=2)
        
        interp_path = os.path.join(output_dir, f'interpolation_{class_a}_to_{class_b}.png')
        vutils.save_image(grid, interp_path)
        print(f"✅ 插值序列已保存: {interp_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='StableLDM推理脚本')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output', type=str, default='generated_samples', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=4, help='每个类别生成的样本数')
    parser.add_argument('--num_classes', type=int, default=8, help='显示的类别数')
    parser.add_argument('--steps', type=int, default=50, help='推理步数')
    parser.add_argument('--cfg', type=float, default=7.5, help='CFG引导强度')
    parser.add_argument('--use_ema', action='store_true', help='使用EMA权重')
    parser.add_argument('--interpolation', action='store_true', help='生成插值样本')
    parser.add_argument('--class_a', type=int, default=0, help='插值起始类别')
    parser.add_argument('--class_b', type=int, default=5, help='插值结束类别')
    
    args = parser.parse_args()
    
    print("🎯 轻量化StableLDM推理")
    print("=" * 50)
    
    # 加载模型
    model, config, device = load_model(args.config, args.checkpoint)
    
    if args.interpolation:
        # 生成插值
        generate_interpolation(
            model, config, device,
            class_a=args.class_a,
            class_b=args.class_b,
            num_steps=8,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            use_ema=args.use_ema,
            output_dir=args.output
        )
    else:
        # 生成常规样本
        generate_samples(
            model, config, device,
            num_samples_per_class=args.num_samples,
            num_classes_to_show=args.num_classes,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            use_ema=args.use_ema,
            output_dir=args.output
        )
    
    print("=" * 50)
    print("🎉 推理完成！")

if __name__ == "__main__":
    main() 
