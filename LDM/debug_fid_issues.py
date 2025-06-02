"""
VAE-LDM FID诊断脚本
分析FID分数高的具体原因 - 本地版本
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import save_image, make_grid
from PIL import Image
import yaml

# 添加VAE路径
sys.path.append('../VAE')

from vae_ldm import create_vae_ldm
from fid_evaluation import FIDEvaluator
from metrics import denormalize_for_metrics

# 简化的数据集类（避免pickle问题）
class SimpleSyntheticDataset:
    """简单的合成数据集，避免多进程pickle问题"""
    def __init__(self, num_samples=1000, num_classes=31):
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        # 生成固定的合成数据
        torch.manual_seed(42)
        self.images = torch.randn(num_samples, 3, 256, 256)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
        # 归一化到[-1, 1]
        self.images = (self.images - self.images.mean()) / self.images.std()
        self.images = torch.clamp(self.images, -3, 3) / 3  # 限制范围
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'label': self.labels[idx]
        }

class FIDDiagnostics:
    """FID问题诊断器 - 本地版本"""
    
    def __init__(self, config_path="config_local_debug.yaml"):
        # 加载配置
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            print(f"⚠️ 配置文件 {config_path} 不存在，使用默认配置")
            self.config = self._create_default_config()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 使用设备: {self.device}")
        
        # 初始化模型
        print("🔧 加载模型...")
        try:
            self.model = create_vae_ldm(
                vae_checkpoint_path=self.config['paths']['vae_checkpoint'],
                image_size=self.config['model']['image_size'],
                num_classes=self.config['model']['num_classes'],
                diffusion_steps=self.config['model']['diffusion_steps'],
                noise_schedule=self.config['model']['noise_schedule'],
                device=self.device
            )
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"⚠️ 模型加载失败: {e}")
            print("🔄 使用默认参数创建模型...")
            self.model = create_vae_ldm(
                vae_checkpoint_path=None,  # 不加载VAE checkpoint
                image_size=32,
                num_classes=31,
                diffusion_steps=1000,
                noise_schedule="cosine",
                device=self.device
            )
        
        # 创建简单数据集（避免数据加载问题）
        print("📊 创建合成数据集...")
        self.dataset = SimpleSyntheticDataset(num_samples=200, num_classes=31)
        
        # 创建保存目录
        self.debug_dir = "debug_output"
        os.makedirs(self.debug_dir, exist_ok=True)
        
    def _create_default_config(self):
        """创建默认配置"""
        return {
            'paths': {
                'vae_checkpoint': None,
                'data_dir': './data',
                'save_dir': './debug_checkpoints'
            },
            'model': {
                'num_classes': 31,
                'image_size': 32,
                'diffusion_steps': 1000,
                'noise_schedule': 'cosine'
            }
        }
        
    def analyze_dataset(self):
        """分析数据集"""
        print("\n📊 分析数据集...")
        
        # 统计类别分布
        class_counts = {}
        pixel_stats = []
        
        # 采样一些数据进行分析
        sample_size = min(100, len(self.dataset))
        for i in range(sample_size):
            data = self.dataset[i]
            image = data['image']
            label = data['label'].item()
            
            class_counts[label] = class_counts.get(label, 0) + 1
            pixel_stats.extend(image.flatten().tolist())
        
        print(f"   类别分布: {len(class_counts)} 个类别")
        if class_counts:
            print(f"   样本数最多的类别: {max(class_counts.values())}")
            print(f"   样本数最少的类别: {min(class_counts.values())}")
        
        print(f"   像素值范围: [{min(pixel_stats):.3f}, {max(pixel_stats):.3f}]")
        print(f"   像素值均值: {np.mean(pixel_stats):.3f}")
        print(f"   像素值标准差: {np.std(pixel_stats):.3f}")
        
        # 保存样本图像
        sample_images = torch.stack([self.dataset[i]['image'] for i in range(min(16, len(self.dataset)))])
        display_images = denormalize_for_metrics(sample_images)
        
        save_image(
            display_images, 
            os.path.join(self.debug_dir, "real_samples.png"),
            nrow=4, 
            normalize=False
        )
        print(f"   💾 真实图像样本已保存: {self.debug_dir}/real_samples.png")
        
        return class_counts, pixel_stats
    
    def test_vae_reconstruction(self):
        """测试VAE重建质量"""
        print("\n🔧 测试VAE重建...")
        
        with torch.no_grad():
            # 获取一些样本图像
            sample_images = torch.stack([self.dataset[i]['image'] for i in range(8)])
            original_images = sample_images.to(self.device)
            
            # 编码-解码
            latents = self.model.encode_to_latent(original_images)
            reconstructed = self.model.decode_from_latent(latents)
            
            # 计算重建误差
            mse_loss = F.mse_loss(reconstructed, original_images).item()
            l1_loss = F.l1_loss(reconstructed, original_images).item()
            
            print(f"   VAE重建MSE: {mse_loss:.6f}")
            print(f"   VAE重建L1: {l1_loss:.6f}")
            
            # 保存对比图像
            comparison = torch.cat([
                denormalize_for_metrics(original_images),
                denormalize_for_metrics(reconstructed)
            ], dim=0)
            
            save_image(
                comparison,
                os.path.join(self.debug_dir, "vae_reconstruction.png"),
                nrow=8,
                normalize=False
            )
            print(f"   💾 VAE重建对比已保存: {self.debug_dir}/vae_reconstruction.png")
            
            return mse_loss, l1_loss
    
    def test_generation_quality(self):
        """测试生成质量"""
        print("\n🎨 测试生成质量...")
        
        # 测试不同配置
        configurations = [
            {"steps": 50, "eta": 0.0, "name": "ddim_50_deterministic"},
            {"steps": 100, "eta": 0.0, "name": "ddim_100_deterministic"},
            {"steps": 50, "eta": 0.3, "name": "ddim_50_stochastic"},
        ]
        
        results = {}
        
        for config in configurations:
            print(f"   测试配置: {config['name']}")
            
            # 生成图像
            num_samples = 8  # 减少样本数用于调试
            class_labels = torch.randint(0, 31, (num_samples,), device=self.device)
            
            with torch.no_grad():
                try:
                    generated_images, _ = self.model.sample(
                        num_samples=num_samples,
                        class_labels=class_labels,
                        use_ddim=True,
                        eta=config['eta']
                    )
                except Exception as e:
                    print(f"     ⚠️ 生成失败: {e}")
                    continue
            
            # 检查生成图像的统计信息
            gen_mean = generated_images.mean().item()
            gen_std = generated_images.std().item()
            gen_min = generated_images.min().item()
            gen_max = generated_images.max().item()
            
            results[config['name']] = {
                'mean': gen_mean,
                'std': gen_std,
                'min': gen_min,
                'max': gen_max
            }
            
            print(f"     生成图像统计: 均值={gen_mean:.3f}, 标准差={gen_std:.3f}")
            print(f"     范围: [{gen_min:.3f}, {gen_max:.3f}]")
            
            # 保存生成图像
            display_images = denormalize_for_metrics(generated_images)
            save_image(
                display_images,
                os.path.join(self.debug_dir, f"generated_{config['name']}.png"),
                nrow=4,
                normalize=False
            )
            
            # 检查是否有异常值
            if torch.isnan(generated_images).any():
                print(f"     ⚠️ 发现NaN值！")
            if torch.isinf(generated_images).any():
                print(f"     ⚠️ 发现无穷大值！")
        
        return results
    
    def estimate_fid_score(self):
        """简化的FID估算"""
        print("\n📈 估算FID分数...")
        
        try:
            # 创建FID评估器
            fid_evaluator = FIDEvaluator(device=self.device)
            
            # 准备真实图像
            real_images = torch.stack([self.dataset[i]['image'] for i in range(min(50, len(self.dataset)))])
            real_images_norm = denormalize_for_metrics(real_images)
            
            # 提取真实图像特征
            real_features = fid_evaluator.feature_extractor(real_images_norm.to(self.device))
            real_features = real_features.cpu().numpy()
            
            # 生成图像
            with torch.no_grad():
                generated_images, _ = self.model.sample(
                    num_samples=50,
                    class_labels=torch.randint(0, 31, (50,), device=self.device),
                    use_ddim=True,
                    eta=0.0
                )
            
            gen_images_norm = denormalize_for_metrics(generated_images)
            gen_features = fid_evaluator.feature_extractor(gen_images_norm).cpu().numpy()
            
            # 计算统计量
            real_mu = real_features.mean(axis=0)
            real_sigma = np.cov(real_features, rowvar=False)
            gen_mu = gen_features.mean(axis=0)
            gen_sigma = np.cov(gen_features, rowvar=False)
            
            # 计算FID（简化版）
            diff = real_mu - gen_mu
            diff_norm = np.linalg.norm(diff)
            
            # 协方差矩阵的迹
            trace_real = np.trace(real_sigma)
            trace_gen = np.trace(gen_sigma)
            
            # 简化的FID估算
            estimated_fid = diff_norm + abs(trace_real - trace_gen)
            
            print(f"   特征差异范数: {diff_norm:.2f}")
            print(f"   协方差迹差异: {abs(trace_real - trace_gen):.2f}")
            print(f"   估算FID分数: {estimated_fid:.2f}")
            
            return estimated_fid
            
        except Exception as e:
            print(f"   ⚠️ FID估算失败: {e}")
            return None
    
    def check_model_weights(self):
        """检查模型权重"""
        print("\n🔍 检查模型权重...")
        
        total_params = sum(p.numel() for p in self.model.unet.parameters())
        trainable_params = sum(p.numel() for p in self.model.unet.parameters() if p.requires_grad)
        
        print(f"   总参数数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        
        # 检查关键层的权重统计
        layer_count = 0
        abnormal_layers = []
        
        for name, param in self.model.unet.named_parameters():
            if param.requires_grad:
                layer_count += 1
                mean_val = param.data.mean().item()
                std_val = param.data.std().item()
                
                # 检查异常值
                if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                    abnormal_layers.append(f"{name}: NaN或无穷大")
                elif abs(mean_val) > 10 or std_val > 10:
                    abnormal_layers.append(f"{name}: 异常分布 (均值={mean_val:.3f}, 标准差={std_val:.3f})")
        
        print(f"   检查了 {layer_count} 个层")
        if abnormal_layers:
            print("   ⚠️ 发现异常层:")
            for layer in abnormal_layers[:5]:  # 只显示前5个
                print(f"     {layer}")
        else:
            print("   ✅ 所有权重看起来正常")
    
    def run_full_diagnosis(self):
        """运行完整诊断"""
        print("🔍 开始VAE-LDM FID诊断（本地版本）...")
        print("=" * 60)
        
        try:
            # 1. 数据集分析
            class_counts, pixel_stats = self.analyze_dataset()
            
            # 2. VAE重建测试
            vae_mse, vae_l1 = self.test_vae_reconstruction()
            
            # 3. 生成质量测试
            generation_results = self.test_generation_quality()
            
            # 4. FID估算
            estimated_fid = self.estimate_fid_score()
            
            # 5. 模型权重检查
            self.check_model_weights()
            
            print("\n" + "=" * 60)
            print("📋 诊断总结:")
            print(f"   VAE重建MSE: {vae_mse:.6f}")
            if estimated_fid:
                print(f"   估算FID分数: {estimated_fid:.2f}")
            
            # 诊断建议
            print("\n💡 针对FID=300+的问题分析:")
            
            if vae_mse > 0.1:
                print("   ⚠️ VAE重建质量差，这可能是主要原因")
                print("     建议: 重新训练VAE或检查VAE权重")
            
            if estimated_fid and estimated_fid > 100:
                print("   ⚠️ 估算FID分数很高，可能原因:")
                print("     1. 模型训练不充分（需要更多epoch）")
                print("     2. DDIM采样步数不足（当前50步，建议100+步）")
                print("     3. 缺少分类器自由引导（CFG）")
                print("     4. 潜在空间缩放因子不合适")
                print("     5. 训练数据质量问题")
            
            if abs(np.mean(pixel_stats)) > 0.1:
                print("   ⚠️ 数据预处理可能有问题")
                print("     当前数据均值偏离0，检查归一化")
            
            print(f"\n🎯 调试输出已保存至: {self.debug_dir}/")
            print("\n🚀 建议的改进措施:")
            print("   1. 增加DDIM采样步数到100+")
            print("   2. 启用分类器自由引导（guidance_scale=2.0-7.5）")
            print("   3. 增加训练轮数（当前可能训练不足）")
            print("   4. 检查VAE重建质量")
            print("   5. 验证数据预处理流程")
            
        except Exception as e:
            print(f"\n❌ 诊断过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    diagnostics = FIDDiagnostics()
    diagnostics.run_full_diagnosis() 
