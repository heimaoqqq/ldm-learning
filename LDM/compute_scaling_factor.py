"""
计算微调后VAE的最佳缩放因子
单独运行此脚本来分析VAE的潜变量分布并计算最佳缩放因子
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

# 导入VAE模型
sys.path.append('../VAE')
try:
    from vae_model import AutoencoderKL
    from dataset import build_dataloader as vae_build_dataloader
    print("✅ 成功导入VAE相关模块")
except ImportError as e:
    print(f"❌ 导入VAE模块失败: {e}")
    sys.exit(1)

def load_vae_model(checkpoint_path: str, device: str = 'cuda') -> AutoencoderKL:
    """加载VAE模型"""
    print(f"🔧 加载VAE模型: {checkpoint_path}")
    
    # 创建VAE模型
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        down_block_types=['DownEncoderBlock2D'] * 4,
        up_block_types=['UpDecoderBlock2D'] * 4,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        norm_num_groups=32,
        sample_size=256
    )
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 处理不同的保存格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    vae.load_state_dict(state_dict)
    vae.eval()
    vae.to(device)
    
    print(f"✅ VAE模型加载成功")
    return vae

def compute_scaling_factor(
    vae_model,
    data_loader,
    device: str = 'cuda',
    num_samples: int = 1000,
    plot_distribution: bool = True
) -> dict:
    """
    计算VAE的最佳缩放因子
    
    Returns:
        dict: 包含统计信息和建议缩放因子的字典
    """
    print(f"🔍 计算VAE缩放因子 (分析 {num_samples} 个样本)...")
    
    vae_model.eval()
    latent_samples = []
    sample_count = 0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="收集潜变量样本")
        
        for batch in progress_bar:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images = batch[0]
            else:
                images = batch
            
            images = images.to(device)
            
            # VAE编码（获取原始潜变量）
            with torch.amp.autocast(device_type='cuda', enabled=False):
                images = images.float()
                latent_dist = vae_model.encode(images).latent_dist
                latents = latent_dist.sample()  # [B, 4, H, W]
            
            latent_samples.append(latents.cpu().flatten())  # 展平所有维度
            sample_count += latents.numel()
            
            progress_bar.set_postfix({'samples': sample_count})
            
            if sample_count >= num_samples * 4 * 32 * 32:  # 考虑4通道和空间维度
                break
    
    # 合并所有样本
    all_latents = torch.cat(latent_samples, dim=0)[:num_samples * 4 * 32 * 32]
    
    # 计算统计信息
    mean = all_latents.mean().item()
    std = all_latents.std().item()
    min_val = all_latents.min().item()
    max_val = all_latents.max().item()
    
    # 计算不同百分位数
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    percentile_values = [np.percentile(all_latents.numpy(), p) for p in percentiles]
    
    # 最佳缩放因子 = 1 / std
    optimal_scaling_factor = 1.0 / std
    
    # 分析结果
    results = {
        'num_samples': len(all_latents),
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'percentiles': dict(zip(percentiles, percentile_values)),
        'optimal_scaling_factor': optimal_scaling_factor,
        'stable_diffusion_factor': 0.18215,  # 参考值
        'factor_ratio': optimal_scaling_factor / 0.18215
    }
    
    # 打印详细统计
    print(f"\n📊 VAE潜变量分布统计:")
    print(f"   样本数量: {results['num_samples']:,}")
    print(f"   均值: {mean:.6f}")
    print(f"   标准差: {std:.6f}")
    print(f"   最小值: {min_val:.6f}")
    print(f"   最大值: {max_val:.6f}")
    print(f"\n📈 百分位数分布:")
    for p, v in zip(percentiles, percentile_values):
        print(f"   {p:2d}%: {v:8.4f}")
    
    print(f"\n🎯 缩放因子分析:")
    print(f"   建议缩放因子: {optimal_scaling_factor:.6f}")
    print(f"   Stable Diffusion因子: {0.18215:.6f}")
    print(f"   相对比例: {results['factor_ratio']:.2f}x")
    
    # 评估缩放效果
    scaled_std = std * optimal_scaling_factor
    print(f"   缩放后标准差: {scaled_std:.6f} (目标: 1.0)")
    
    if abs(scaled_std - 1.0) < 0.1:
        print(f"   ✅ 缩放效果良好")
    else:
        print(f"   ⚠️  缩放效果需要验证")
    
    # 绘制分布图
    if plot_distribution:
        plot_latent_distribution(all_latents, results, optimal_scaling_factor)
    
    return results

def plot_latent_distribution(latents, stats, scaling_factor):
    """绘制潜变量分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始分布
    axes[0, 0].hist(latents.numpy(), bins=100, alpha=0.7, density=True)
    axes[0, 0].axvline(stats['mean'], color='red', linestyle='--', label=f'均值: {stats["mean"]:.3f}')
    axes[0, 0].axvline(stats['mean'] + stats['std'], color='orange', linestyle='--', label=f'+1σ: {stats["mean"] + stats["std"]:.3f}')
    axes[0, 0].axvline(stats['mean'] - stats['std'], color='orange', linestyle='--', label=f'-1σ: {stats["mean"] - stats["std"]:.3f}')
    axes[0, 0].set_title('原始潜变量分布')
    axes[0, 0].set_xlabel('值')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 缩放后分布
    scaled_latents = latents * scaling_factor
    axes[0, 1].hist(scaled_latents.numpy(), bins=100, alpha=0.7, density=True, color='green')
    scaled_mean = scaled_latents.mean().item()
    scaled_std = scaled_latents.std().item()
    axes[0, 1].axvline(scaled_mean, color='red', linestyle='--', label=f'均值: {scaled_mean:.3f}')
    axes[0, 1].axvline(scaled_mean + scaled_std, color='orange', linestyle='--', label=f'+1σ: {scaled_mean + scaled_std:.3f}')
    axes[0, 1].axvline(scaled_mean - scaled_std, color='orange', linestyle='--', label=f'-1σ: {scaled_mean - scaled_std:.3f}')
    axes[0, 1].set_title(f'缩放后分布 (×{scaling_factor:.4f})')
    axes[0, 1].set_xlabel('值')
    axes[0, 1].set_ylabel('密度')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q图：检验是否接近正态分布
    from scipy import stats as scipy_stats
    scipy_stats.probplot(scaled_latents.numpy(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q图 (缩放后 vs 标准正态)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 百分位数对比
    percentiles = list(stats['percentiles'].keys())
    values = list(stats['percentiles'].values())
    axes[1, 1].plot(percentiles, values, 'o-', label='原始', linewidth=2)
    axes[1, 1].plot(percentiles, [v * scaling_factor for v in values], 's-', label='缩放后', linewidth=2)
    axes[1, 1].set_title('百分位数对比')
    axes[1, 1].set_xlabel('百分位数 (%)')
    axes[1, 1].set_ylabel('值')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vae_scaling_factor_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📊 分布分析图已保存: vae_scaling_factor_analysis.png")

def main():
    """主函数"""
    print("🔍 VAE缩放因子计算工具")
    print("=" * 50)
    
    # VAE检查点路径
    vae_checkpoint_paths = [
        "/kaggle/input/vae-model/vae_finetuned_epoch_23.pth",
        "VAE/vae_model_best.pth",
        "VAE/vae_model_final.pth",
        "../VAE/vae_model_best.pth"
    ]
    
    vae_checkpoint_path = None
    for path in vae_checkpoint_paths:
        if os.path.exists(path):
            vae_checkpoint_path = path
            break
    
    if vae_checkpoint_path is None:
        print("❌ 未找到VAE检查点文件")
        print("请检查以下路径是否存在:")
        for path in vae_checkpoint_paths:
            print(f"   - {path}")
        return
    
    print(f"✅ 找到VAE检查点: {vae_checkpoint_path}")
    
    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用设备: {device}")
    
    # 加载VAE模型
    try:
        vae_model = load_vae_model(vae_checkpoint_path, device)
    except Exception as e:
        print(f"❌ VAE模型加载失败: {e}")
        return
    
    # 创建数据加载器
    try:
        data_dir = "/kaggle/input/dataset/dataset"
        if not os.path.exists(data_dir):
            data_dir = "dataset"  # 备用路径
        
        train_loader, val_loader, train_size, val_size = vae_build_dataloader(
            root_dir=data_dir,
            batch_size=16,  # 适中的批次大小
            num_workers=2,
            shuffle_train=False,  # 不需要打乱，只是统计
            shuffle_val=False,
            val_split=0.2,
            random_state=42
        )
        
        print(f"✅ 数据加载器创建成功")
        print(f"   训练集: {train_size} 样本")
        print(f"   验证集: {val_size} 样本")
        
        # 使用训练集计算缩放因子
        data_loader = train_loader
        
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        return
    
    # 计算缩放因子
    try:
        results = compute_scaling_factor(
            vae_model=vae_model,
            data_loader=data_loader,
            device=device,
            num_samples=2000,  # 更多样本以获得更准确的统计
            plot_distribution=True
        )
        
        # 保存结果
        import json
        with open('vae_scaling_factor_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 结果已保存到: vae_scaling_factor_results.json")
        
        # 给出使用建议
        print(f"\n📝 使用建议:")
        print(f"   1. 在 ldm_config.yaml 中更新缩放因子")
        print(f"   2. 或者在训练时设置 auto_update_scaling_factor: true")
        print(f"   3. 建议缩放因子: {results['optimal_scaling_factor']:.6f}")
        
        if results['factor_ratio'] > 2.0 or results['factor_ratio'] < 0.5:
            print(f"   ⚠️  缩放因子与标准值差异较大，请仔细验证")
        else:
            print(f"   ✅ 缩放因子在合理范围内")
            
    except Exception as e:
        print(f"❌ 缩放因子计算失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
