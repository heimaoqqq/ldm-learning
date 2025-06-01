"""
🔮 增强版LDM推理脚本
使用训练好的模型生成高质量图像
支持批量生成、质量评估和可视化
"""

import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
from tqdm import tqdm
import torchvision.transforms as transforms
from typing import List, Dict, Optional, Tuple
import argparse
from datetime import datetime

# 添加VAE模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VAE'))

from stable_ldm import create_stable_ldm
from metrics import DiffusionMetrics, denormalize_for_metrics
from dataset_adapter import build_dataloader  # 🔧 使用适配器

class EnhancedInference:
    """增强版LDM推理器"""
    
    def __init__(self, config_path: str, model_path: str):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔮 推理设备: {self.device}")
        
        # 创建模型
        self.model = create_stable_ldm(self.config).to(self.device)
        
        # 加载预训练权重
        self.load_model(model_path)
        
        # 创建指标计算器
        self.metrics_calculator = DiffusionMetrics(self.device)
        
        # 类别标签映射（如果有的话）
        self.class_names = self._get_class_names()
        
        print(f"✅ 模型加载完成，支持{self.config['unet']['num_classes']}个类别")
    
    def load_model(self, model_path: str):
        """加载模型权重"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理不同的保存格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            print(f"✅ 模型权重加载成功: {model_path}")
            
            # 打印额外信息（如果有的话）
            if 'epoch' in checkpoint:
                print(f"📊 训练轮数: {checkpoint['epoch']}")
            if 'best_fid' in checkpoint:
                print(f"🏆 最佳FID: {checkpoint['best_fid']:.2f}")
                
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    def _get_class_names(self) -> List[str]:
        """获取类别名称"""
        # 这里可以根据数据集定义类别名称
        # 暂时使用通用名称
        return [f"Class_{i}" for i in range(self.config['unet']['num_classes'])]
    
    def generate_single_class(
        self, 
        class_id: int, 
        num_samples: int = 16,
        **sampling_kwargs
    ) -> torch.Tensor:
        """为单个类别生成图像"""
        
        if class_id < 0 or class_id >= self.config['unet']['num_classes']:
            raise ValueError(f"类别ID {class_id} 超出范围 [0, {self.config['unet']['num_classes']-1}]")
        
        class_labels = torch.full((num_samples,), class_id, dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            images = self.model.sample(
                batch_size=num_samples,
                class_labels=class_labels,
                num_inference_steps=sampling_kwargs.get('num_inference_steps', 
                                                       self.config['inference']['num_inference_steps']),
                guidance_scale=sampling_kwargs.get('guidance_scale', 
                                                 self.config['inference']['guidance_scale']),
                eta=sampling_kwargs.get('eta', self.config['inference']['eta']),
                use_ema=sampling_kwargs.get('use_ema', True),
                verbose=sampling_kwargs.get('verbose', True)
            )
        
        return images
    
    def generate_mixed_classes(
        self, 
        num_samples: int = 16,
        class_distribution: Optional[List[int]] = None,
        **sampling_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成混合类别的图像"""
        
        if class_distribution is None:
            # 均匀分布
            class_labels = torch.randint(0, self.config['unet']['num_classes'], 
                                       (num_samples,), device=self.device)
        else:
            # 按指定分布生成
            if len(class_distribution) != self.config['unet']['num_classes']:
                raise ValueError("class_distribution长度必须等于类别数")
            
            # 按权重随机采样
            weights = torch.tensor(class_distribution, dtype=torch.float)
            class_labels = torch.multinomial(weights, num_samples, replacement=True).to(self.device)
        
        with torch.no_grad():
            images = self.model.sample(
                batch_size=num_samples,
                class_labels=class_labels,
                num_inference_steps=sampling_kwargs.get('num_inference_steps', 
                                                       self.config['inference']['num_inference_steps']),
                guidance_scale=sampling_kwargs.get('guidance_scale', 
                                                 self.config['inference']['guidance_scale']),
                eta=sampling_kwargs.get('eta', self.config['inference']['eta']),
                use_ema=sampling_kwargs.get('use_ema', True),
                verbose=sampling_kwargs.get('verbose', True)
            )
        
        return images, class_labels
    
    def generate_unconditional(self, num_samples: int = 16, **sampling_kwargs) -> torch.Tensor:
        """生成无条件图像"""
        
        with torch.no_grad():
            images = self.model.sample(
                batch_size=num_samples,
                class_labels=None,  # 无条件生成
                num_inference_steps=sampling_kwargs.get('num_inference_steps', 
                                                       self.config['inference']['num_inference_steps']),
                guidance_scale=sampling_kwargs.get('guidance_scale', 1.0),  # 无条件时关闭CFG
                eta=sampling_kwargs.get('eta', self.config['inference']['eta']),
                use_ema=sampling_kwargs.get('use_ema', True),
                verbose=sampling_kwargs.get('verbose', True)
            )
        
        return images
    
    def evaluate_generation_quality(
        self, 
        num_samples: int = 1000,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """评估生成质量"""
        
        print(f"🔄 评估生成质量，共生成{num_samples}个样本...")
        
        all_images = []
        all_is_scores = []
        
        # 分批生成
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="生成样本"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # 生成混合类别图像
            images, _ = self.generate_mixed_classes(
                num_samples=current_batch_size,
                verbose=False
            )
            
            all_images.append(images.cpu())
        
        # 合并所有图像
        all_images = torch.cat(all_images, dim=0)
        
        # 计算IS分数
        try:
            is_mean, is_std = self.metrics_calculator.inception_calculator.calculate_inception_score(
                all_images
            )
            print(f"📊 IS分数: {is_mean:.2f} ± {is_std:.2f}")
        except Exception as e:
            print(f"⚠️ IS分数计算失败: {e}")
            is_mean, is_std = None, None
        
        # 计算图像统计信息
        img_stats = {
            'mean': all_images.mean().item(),
            'std': all_images.std().item(),
            'min': all_images.min().item(),
            'max': all_images.max().item()
        }
        
        # 检查图像质量指标
        quality_metrics = {
            'inception_score_mean': is_mean,
            'inception_score_std': is_std,
            'image_mean': img_stats['mean'],
            'image_std': img_stats['std'],
            'pixel_range_min': img_stats['min'],
            'pixel_range_max': img_stats['max'],
            'total_samples': len(all_images)
        }
        
        return quality_metrics
    
    def save_images(
        self, 
        images: torch.Tensor, 
        output_dir: str,
        prefix: str = "generated",
        class_labels: Optional[torch.Tensor] = None
    ):
        """保存生成的图像"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存网格图像
        import torchvision.utils as vutils
        grid_path = os.path.join(output_dir, f"{prefix}_grid.png")
        vutils.save_image(
            images, grid_path,
            nrow=int(np.sqrt(len(images))),
            normalize=True, value_range=(0, 1)
        )
        print(f"📸 网格图像已保存: {grid_path}")
        
        # 保存单独图像
        to_pil = transforms.ToPILImage()
        
        for i, img in enumerate(images):
            # 确保图像在[0,1]范围内
            img = torch.clamp(img, 0, 1)
            pil_img = to_pil(img)
            
            if class_labels is not None:
                class_name = self.class_names[class_labels[i].item()]
                filename = f"{prefix}_{i:04d}_{class_name}.png"
            else:
                filename = f"{prefix}_{i:04d}.png"
            
            img_path = os.path.join(output_dir, filename)
            pil_img.save(img_path)
        
        print(f"💾 {len(images)}张单独图像已保存到: {output_dir}")
    
    def compare_sampling_parameters(
        self, 
        class_id: int = 0,
        num_samples: int = 4
    ):
        """比较不同采样参数的效果"""
        
        print(f"🔬 比较采样参数效果 (类别: {self.class_names[class_id]})")
        
        # 不同的参数组合
        param_configs = [
            {'num_inference_steps': 25, 'guidance_scale': 1.0, 'eta': 0.0, 'name': 'Fast_NoGuidance'},
            {'num_inference_steps': 50, 'guidance_scale': 7.5, 'eta': 0.0, 'name': 'Standard_DDIM'},
            {'num_inference_steps': 75, 'guidance_scale': 8.5, 'eta': 0.3, 'name': 'High_Quality'},
            {'num_inference_steps': 100, 'guidance_scale': 10.0, 'eta': 0.5, 'name': 'Maximum_Quality'},
        ]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_dir = f"sampling_comparison_{timestamp}"
        os.makedirs(comparison_dir, exist_ok=True)
        
        for config in param_configs:
            print(f"🔄 测试配置: {config['name']}")
            
            # 生成图像
            images = self.generate_single_class(
                class_id=class_id,
                num_samples=num_samples,
                **{k: v for k, v in config.items() if k != 'name'}
            )
            
            # 保存结果
            self.save_images(
                images, 
                comparison_dir,
                prefix=config['name']
            )
        
        print(f"✅ 采样参数比较完成，结果保存在: {comparison_dir}")
    
    def interpolate_between_classes(
        self, 
        class_a: int, 
        class_b: int,
        num_steps: int = 8,
        num_samples: int = 1
    ):
        """类别间插值生成"""
        
        print(f"🌈 类别插值: {self.class_names[class_a]} → {self.class_names[class_b]}")
        
        # 创建插值权重
        alphas = torch.linspace(0, 1, num_steps)
        
        all_images = []
        
        for alpha in alphas:
            # 混合类别embedding（这里简化处理）
            if alpha < 0.5:
                current_class = class_a
            else:
                current_class = class_b
            
            images = self.generate_single_class(
                class_id=current_class,
                num_samples=num_samples,
                verbose=False
            )
            
            all_images.append(images)
        
        # 合并所有图像
        interpolation_images = torch.cat(all_images, dim=0)
        
        # 保存插值结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interp_dir = f"interpolation_{timestamp}"
        self.save_images(
            interpolation_images, 
            interp_dir,
            prefix=f"interp_{self.class_names[class_a]}_to_{self.class_names[class_b]}"
        )
        
        print(f"✅ 类别插值完成，结果保存在: {interp_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强版LDM推理')
    parser.add_argument('--config', type=str, default='config_enhanced_unet.yaml', 
                       help='配置文件路径')
    parser.add_argument('--model', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--mode', type=str, default='generate',
                       choices=['generate', 'evaluate', 'compare', 'interpolate'],
                       help='推理模式')
    parser.add_argument('--class_id', type=int, default=0,
                       help='类别ID')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='生成样本数')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='输出目录')
    parser.add_argument('--steps', type=int, default=None,
                       help='推理步数')
    parser.add_argument('--guidance', type=float, default=None,
                       help='CFG引导强度')
    parser.add_argument('--eta', type=float, default=None,
                       help='DDIM随机性参数')
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = EnhancedInference(args.config, args.model)
    
    # 准备采样参数
    sampling_kwargs = {}
    if args.steps is not None:
        sampling_kwargs['num_inference_steps'] = args.steps
    if args.guidance is not None:
        sampling_kwargs['guidance_scale'] = args.guidance
    if args.eta is not None:
        sampling_kwargs['eta'] = args.eta
    
    # 执行推理
    if args.mode == 'generate':
        print(f"🎨 生成模式: 类别{args.class_id}, {args.num_samples}个样本")
        images = inferencer.generate_single_class(
            class_id=args.class_id,
            num_samples=args.num_samples,
            **sampling_kwargs
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{args.output_dir}_{timestamp}"
        
        inferencer.save_images(images, output_dir, f"class_{args.class_id}")
        
    elif args.mode == 'evaluate':
        print(f"📊 评估模式: {args.num_samples}个样本")
        quality_metrics = inferencer.evaluate_generation_quality(args.num_samples)
        
        print("\n📊 生成质量评估结果:")
        for key, value in quality_metrics.items():
            if value is not None:
                print(f"  {key}: {value:.4f}")
    
    elif args.mode == 'compare':
        print("🔬 参数比较模式")
        inferencer.compare_sampling_parameters(args.class_id, args.num_samples)
    
    elif args.mode == 'interpolate':
        if args.class_id + 1 < inferencer.config['unet']['num_classes']:
            print("🌈 插值模式")
            inferencer.interpolate_between_classes(
                args.class_id, 
                args.class_id + 1,
                num_samples=args.num_samples
            )
        else:
            print("⚠️ 类别ID过大，无法进行插值")

if __name__ == "__main__":
    main() 
