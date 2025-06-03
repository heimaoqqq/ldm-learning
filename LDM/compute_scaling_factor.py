"""
计算微调后VAE的最佳缩放因子
单独运行此脚本来分析VAE的潜变量分布并计算最佳缩放因子
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import List, Tuple, Optional, Union

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  使用设备: {device}")

# ==================== VAE模型定义 ====================
# 直接在脚本中定义VAE模型，避免导入问题

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0, groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = None

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = torch.nn.functional.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = torch.nn.functional.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)

        return x + h

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)

class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, add_downsample=True):
        super().__init__()
        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock(in_ch, out_channels))
        
        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample(out_channels)])

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        
        return hidden_states

class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, add_upsample=True):
        super().__init__()
        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock(in_ch, out_channels))
        
        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample(out_channels)])

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        
        return hidden_states

class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        if self.deterministic:
            return self.mean
        else:
            return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var + 
                    self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def mode(self):
        return self.mean

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=4,
        down_block_types=["DownEncoderBlock2D"],
        block_out_channels=[128],
        layers_per_block=2,
        norm_num_groups=32,
        double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DownEncoderBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block,
                add_downsample=not is_final_block,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = ResnetBlock(block_out_channels[-1], block_out_channels[-1])

        # end
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[-1])
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

    def forward(self, x):
        sample = x
        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        sample = self.mid_block(sample)

        sample = self.conv_norm_out(sample)
        sample = torch.nn.functional.silu(sample)
        sample = self.conv_out(sample)

        return sample

class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        up_block_types=["UpDecoderBlock2D"],
        block_out_channels=[128],
        layers_per_block=2,
        norm_num_groups=32,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)

        self.mid_block = ResnetBlock(block_out_channels[-1], block_out_channels[-1])

        self.up_blocks = nn.ModuleList([])

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            up_block = UpDecoderBlock2D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block + 1,
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.conv_norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[0])
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, z):
        sample = z
        sample = self.conv_in(sample)

        sample = self.mid_block(sample)

        for up_block in self.up_blocks:
            sample = up_block(sample)

        sample = self.conv_norm_out(sample)
        sample = torch.nn.functional.silu(sample)
        sample = self.conv_out(sample)

        return sample

class AutoencoderKL(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        norm_num_groups=32,
        sample_size=256,
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
        )

        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, sample):
        x = sample
        posterior = self.encode(x)
        z = posterior.sample()
        dec = self.decode(z)
        return dec, posterior

# ==================== 数据加载器 ====================

class ImageDataset(Dataset):
    """简单的图像数据集类"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # 收集所有图像文件
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
        
        print(f"找到 {len(self.image_paths)} 张图像")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def create_dataloader(data_dir, batch_size=16, num_workers=2):
    """创建数据加载器"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    dataset = ImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, len(dataset)

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
                latent_dist = vae_model.encode(images)  # 直接返回DiagonalGaussianDistribution
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
    
    # 加载VAE模型
    try:
        print(f"🔧 加载VAE模型: {vae_checkpoint_path}")
        
        # 创建VAE模型
        vae_model = AutoencoderKL(
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
        checkpoint = torch.load(vae_checkpoint_path, map_location='cpu')
        
        # 处理不同的保存格式
        if 'vae_state_dict' in checkpoint:
            state_dict = checkpoint['vae_state_dict']
            print("✅ 检测到 'vae_state_dict' 格式")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("✅ 检测到 'model_state_dict' 格式")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("✅ 检测到 'state_dict' 格式")
        else:
            state_dict = checkpoint
            print("✅ 检测到直接状态字典格式")
        
        # 清理可能的键名前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('vae.'):
                new_key = key[4:]  # 移除 'vae.' 前缀
            elif key.startswith('model.'):
                new_key = key[6:]  # 移除 'model.' 前缀
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        # 加载状态字典
        missing_keys, unexpected_keys = vae_model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  缺失的键: {missing_keys[:5]}...")  # 只显示前5个
        if unexpected_keys:
            print(f"⚠️  意外的键: {unexpected_keys[:5]}...")  # 只显示前5个
        
        vae_model.eval()
        vae_model.to(device)
        
        print(f"✅ VAE模型加载成功")
        
    except Exception as e:
        print(f"❌ VAE模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建数据加载器
    try:
        # Kaggle环境的可能数据路径
        possible_data_dirs = [
            "/kaggle/input/dataset/dataset",
            "/kaggle/input/dataset",
            "/kaggle/input/data/dataset", 
            "dataset",
            "data",
            "../dataset"
        ]
        
        data_dir = None
        for path in possible_data_dirs:
            if os.path.exists(path) and os.path.isdir(path):
                # 检查是否包含图像文件
                has_images = False
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            has_images = True
                            break
                    if has_images:
                        break
                
                if has_images:
                    data_dir = path
                    break
        
        if data_dir is None:
            print("❌ 未找到有效的数据集目录")
            print("请检查以下路径是否存在且包含图像文件:")
            for path in possible_data_dirs:
                exists = "✅" if os.path.exists(path) else "❌"
                print(f"   {exists} {path}")
            return
        
        print(f"✅ 找到数据集目录: {data_dir}")
        
        train_loader, train_size = create_dataloader(data_dir)
        
        print(f"✅ 数据加载器创建成功")
        print(f"   训练集: {train_size} 样本")
        
        # 使用训练集计算缩放因子
        data_loader = train_loader
        
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        import traceback
        traceback.print_exc()
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
