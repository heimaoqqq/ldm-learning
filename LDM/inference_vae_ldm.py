"""
VAE-LDM 推理脚本
用于生成微多普勒时频图样本
"""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from typing import List, Optional

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VAE'))

from vae_ldm import create_vae_ldm

class VAELDMInference:
    """VAE-LDM推理器"""
    
    def __init__(
        self,
        checkpoint_path: str,
        vae_checkpoint_path: str,
        device: str = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        print("🚀 初始化VAE-LDM模型...")
        self.model = create_vae_ldm(
            vae_checkpoint_path=vae_checkpoint_path,
            device=self.device
        )
        
        # 加载检查点
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📂 加载检查点: {checkpoint_path}")
            self.model.load_checkpoint(checkpoint_path)
        else:
            print(f"⚠️  检查点未找到: {checkpoint_path}")
        
        self.model.eval()
        
        print("✅ VAE-LDM推理器初始化完成!")
    
    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int,
        class_labels: Optional[List[int]] = None,
        use_ddim: bool = True,
        num_inference_steps: Optional[int] = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        生成样本
        
        Args:
            num_samples: 生成样本数量
            class_labels: 类别标签列表
            use_ddim: 是否使用DDIM采样
            num_inference_steps: 推理步数
            eta: DDIM随机性参数
            guidance_scale: 分类器自由引导强度
            seed: 随机种子
            show_progress: 是否显示进度
            
        Returns:
            generated_images: [num_samples, 3, H, W] 生成的图像
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 处理类别标签
        if class_labels is not None:
            if len(class_labels) != num_samples:
                raise ValueError(f"class_labels长度 ({len(class_labels)}) 与 num_samples ({num_samples}) 不匹配")
            class_labels = torch.tensor(class_labels, dtype=torch.long, device=self.device)
        else:
            # 随机选择类别
            class_labels = torch.randint(0, self.model.unet.num_classes, (num_samples,), device=self.device)
        
        print(f"🎨 开始生成 {num_samples} 个样本...")
        if class_labels is not None:
            print(f"   类别分布: {torch.bincount(class_labels).tolist()}")
        print(f"   采样方法: {'DDIM' if use_ddim else 'DDPM'}")
        print(f"   推理步数: {num_inference_steps}")
        print(f"   随机性参数: {eta}")
        
        # 生成样本
        generated_images, _ = self.model.sample(
            num_samples=num_samples,
            class_labels=class_labels,
            use_ddim=use_ddim,
            eta=eta,
            return_intermediates=False
        )
        
        print("✅ 样本生成完成!")
        
        return generated_images
    
    def save_samples(
        self,
        images: torch.Tensor,
        save_dir: str,
        prefix: str = "generated",
        format: str = "png"
    ):
        """
        保存生成的样本
        
        Args:
            images: [N, 3, H, W] 图像张量
            save_dir: 保存目录
            prefix: 文件名前缀
            format: 图像格式
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 转换到CPU并归一化到[0,1]
        images = images.cpu()
        images = (images + 1) / 2  # 从[-1,1]转换到[0,1]
        images = torch.clamp(images, 0, 1)
        
        print(f"💾 保存 {len(images)} 个样本到 {save_dir}")
        
        for i, img in enumerate(images):
            # 转换为PIL图像
            img_np = img.permute(1, 2, 0).numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
            # 保存
            filename = f"{prefix}_{i:04d}.{format}"
            filepath = os.path.join(save_dir, filename)
            img_pil.save(filepath)
        
        print(f"✅ 样本保存完成: {save_dir}")
    
    def create_class_grid(
        self,
        num_samples_per_class: int = 4,
        max_classes: int = 16,
        use_ddim: bool = True,
        eta: float = 0.0,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        创建类别网格图像
        
        Args:
            num_samples_per_class: 每个类别的样本数
            max_classes: 最大类别数
            use_ddim: 是否使用DDIM
            eta: DDIM随机性参数
            seed: 随机种子
            
        Returns:
            grid_image: 网格图像
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        num_classes = min(max_classes, self.model.unet.num_classes)
        total_samples = num_classes * num_samples_per_class
        
        # 创建类别标签
        class_labels = []
        for class_id in range(num_classes):
            class_labels.extend([class_id] * num_samples_per_class)
        
        # 生成样本
        generated_images = self.generate_samples(
            num_samples=total_samples,
            class_labels=class_labels,
            use_ddim=use_ddim,
            eta=eta,
            show_progress=True
        )
        
        # 创建网格
        return self._make_grid(generated_images, nrow=num_samples_per_class)
    
    def _make_grid(self, images: torch.Tensor, nrow: int = 8, padding: int = 2) -> torch.Tensor:
        """创建图像网格"""
        # 归一化到[0,1]
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        
        B, C, H, W = images.shape
        ncol = (B + nrow - 1) // nrow
        
        # 创建网格画布
        grid_h = ncol * H + (ncol + 1) * padding
        grid_w = nrow * W + (nrow + 1) * padding
        grid = torch.ones((C, grid_h, grid_w))
        
        # 填充图像
        for idx in range(B):
            row = idx // nrow
            col = idx % nrow
            
            y_start = row * H + (row + 1) * padding
            y_end = y_start + H
            x_start = col * W + (col + 1) * padding
            x_end = x_start + W
            
            grid[:, y_start:y_end, x_start:x_end] = images[idx]
        
        return grid
    
    def interpolate_between_classes(
        self,
        class_a: int,
        class_b: int,
        num_steps: int = 8,
        use_ddim: bool = True,
        eta: float = 0.0,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        在两个类别之间进行插值
        
        Args:
            class_a: 起始类别
            class_b: 结束类别
            num_steps: 插值步数
            use_ddim: 是否使用DDIM
            eta: DDIM随机性参数
            seed: 随机种子
            
        Returns:
            interpolated_images: 插值图像序列
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        print(f"🔄 在类别 {class_a} 和 {class_b} 之间进行插值 ({num_steps} 步)")
        
        # 生成端点样本
        end_samples = self.generate_samples(
            num_samples=2,
            class_labels=[class_a, class_b],
            use_ddim=use_ddim,
            eta=eta,
            show_progress=False
        )
        
        # 简单线性插值（在图像空间）
        interpolated = []
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            interpolated_img = (1 - alpha) * end_samples[0] + alpha * end_samples[1]
            interpolated.append(interpolated_img)
        
        return torch.stack(interpolated)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='VAE-LDM推理')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                       help='VAE检查点路径')
    parser.add_argument('--output_dir', type=str, default='generated_samples',
                       help='输出目录')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='生成样本数量')
    parser.add_argument('--class_labels', type=int, nargs='*', default=None,
                       help='指定类别标签')
    parser.add_argument('--ddim_steps', type=int, default=50,
                       help='DDIM采样步数')
    parser.add_argument('--eta', type=float, default=0.0,
                       help='DDIM随机性参数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--create_grid', action='store_true',
                       help='创建类别网格')
    parser.add_argument('--samples_per_class', type=int, default=4,
                       help='每个类别的样本数（网格模式）')
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = VAELDMInference(
        checkpoint_path=args.checkpoint,
        vae_checkpoint_path=args.vae_checkpoint
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.create_grid:
        # 创建类别网格
        print("🎨 创建类别网格...")
        grid_image = inference.create_class_grid(
            num_samples_per_class=args.samples_per_class,
            use_ddim=True,
            eta=args.eta,
            seed=args.seed
        )
        
        # 保存网格
        grid_pil = Image.fromarray((grid_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        grid_path = os.path.join(args.output_dir, 'class_grid.png')
        grid_pil.save(grid_path)
        print(f"✅ 类别网格保存至: {grid_path}")
    
    else:
        # 生成普通样本
        generated_images = inference.generate_samples(
            num_samples=args.num_samples,
            class_labels=args.class_labels,
            use_ddim=True,
            num_inference_steps=args.ddim_steps,
            eta=args.eta,
            seed=args.seed
        )
        
        # 保存样本
        inference.save_samples(
            images=generated_images,
            save_dir=args.output_dir,
            prefix="sample"
        )
        
        # 创建预览网格
        grid_image = inference._make_grid(generated_images, nrow=4)
        grid_pil = Image.fromarray((grid_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        grid_path = os.path.join(args.output_dir, 'preview_grid.png')
        grid_pil.save(grid_path)
        print(f"✅ 预览网格保存至: {grid_path}")

if __name__ == "__main__":
    main() 
