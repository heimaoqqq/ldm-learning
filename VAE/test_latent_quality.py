import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from adv_vq_vae import AdvVQVAE
from dataset import build_dataloader
import torchvision.utils as vutils

def test_latent_space_quality():
    """测试VQ-VAE潜在空间的质量"""
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    _, val_loader, _, _ = build_dataloader(
        config['dataset']['root_dir'],
        batch_size=8,
        num_workers=0,
        shuffle_train=False,
        shuffle_val=False,
        val_split=0.3
    )
    
    # 加载模型（使用最佳FID模型）
    model = AdvVQVAE(
        in_channels=config['model']['in_channels'],
        latent_dim=config['model']['latent_dim'],
        num_embeddings=config['model']['num_embeddings'],
        beta=config['model']['beta'],
        decay=config['model'].get('vq_ema_decay', 0.99),
        groups=config['model']['groups'],
        disc_ndf=64
    ).to(device)
    
    # save_dir = config['training']['save_dir'] # 不再使用配置文件中的save_dir
    # model_path = os.path.join(save_dir, 'adv_vqvae_best_fid.pth') # 不再拼接路径
    model_path = '/kaggle/input/vae-best-fid2/adv_vqvae_best_fid2.pth' # 直接使用你的路径
    
    if not os.path.exists(model_path):
        print(f"未找到模型: {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"已加载模型: {model_path}")
    
    def denormalize(x):
        return (x * 0.5 + 0.5).clamp(0, 1)
    
    with torch.no_grad():
        # 获取一批测试图像
        images, _ = next(iter(val_loader))
        images = images[:4].to(device)  # 只用前4张图像
        
        print("\n=== 1. 重建质量测试 ===")
        
        # 编码和重建
        z = model.encoder(images)
        z_q, _, _ = model.vq(z)
        recons = model.decoder(z_q)
        
        # 计算重建指标
        mse_loss = torch.nn.MSELoss()(recons, images)
        print(f"重建MSE损失: {mse_loss:.6f}")
        
        # 保存重建对比图
        os.makedirs("latent_quality_test", exist_ok=True)
        
        comparison = torch.cat([
            denormalize(images),
            denormalize(recons)
        ], dim=0)
        
        vutils.save_image(
            comparison, 
            "latent_quality_test/reconstruction_quality.png",
            nrow=4, 
            padding=2
        )
        print("✅ 重建对比图已保存: latent_quality_test/reconstruction_quality.png")
        
        print("\n=== 2. 潜在空间插值测试 ===")
        
        # 选择两张图像进行插值
        img1, img2 = images[0:1], images[1:2]
        
        # 编码
        z1 = model.encoder(img1)
        z2 = model.encoder(img2)
        
        # 在编码空间插值（注意：这是在VQ之前的连续空间）
        interpolation_steps = 8
        interpolated_images = []
        
        for i in range(interpolation_steps):
            alpha = i / (interpolation_steps - 1)
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # VQ量化
            z_q_interp, _, _ = model.vq(z_interp)
            
            # 解码
            recon_interp = model.decoder(z_q_interp)
            interpolated_images.append(recon_interp)
        
        # 保存插值结果
        interpolated_tensor = torch.cat(interpolated_images, dim=0)
        vutils.save_image(
            denormalize(interpolated_tensor),
            "latent_quality_test/latent_interpolation.png",
            nrow=interpolation_steps,
            padding=2
        )
        print("✅ 潜在空间插值图已保存: latent_quality_test/latent_interpolation.png")
        
        print("\n=== 3. 码本使用分析 ===")
        
        # 编码更多图像来分析码本使用
        all_indices = []
        sample_count = 0
        max_samples = 500  # 分析500个样本
        
        for batch_images, _ in val_loader:
            if sample_count >= max_samples:
                break
                
            batch_images = batch_images.to(device)
            z = model.encoder(batch_images)
            _, _, info = model.vq(z)
            
            # 获取量化索引
            if hasattr(info, 'indices'):
                indices = info.indices
            elif isinstance(info, dict) and 'indices' in info:
                indices = info['indices']
            else:
                # 如果无法直接获取索引，通过距离计算
                z_flattened = z.view(-1, z.size(-1))
                codebook = model.vq.embedding.weight.data
                distances = torch.cdist(z_flattened, codebook)
                indices = torch.argmin(distances, dim=1)
            
            all_indices.extend(indices.cpu().numpy().flatten())
            sample_count += batch_images.size(0)
        
        # 分析码本使用统计
        unique_indices = set(all_indices)
        total_codes = config['model']['num_embeddings']
        usage_rate = len(unique_indices) / total_codes
        
        print(f"码本使用统计:")
        print(f"  总码字数: {total_codes}")
        print(f"  使用的码字数: {len(unique_indices)}")
        print(f"  使用率: {usage_rate:.2%}")
        
        # 分析使用频率分布
        from collections import Counter
        usage_counts = Counter(all_indices)
        most_common = usage_counts.most_common(10)
        
        print(f"  最常用的10个码字:")
        for idx, count in most_common:
            print(f"    码字 {idx}: 使用 {count} 次")
        
        print("\n=== 4. 潜在空间维度分析 ===")
        
        # 收集潜在编码的统计信息
        all_z = []
        sample_count = 0
        
        for batch_images, _ in val_loader:
            if sample_count >= 200:  # 分析200个样本
                break
                
            batch_images = batch_images.to(device)
            z = model.encoder(batch_images)
            all_z.append(z.cpu())
            sample_count += batch_images.size(0)
        
        all_z = torch.cat(all_z, dim=0)
        z_flat = all_z.view(all_z.size(0), -1)  # 展平为 [N, H*W*C]
        
        # 计算统计量
        z_mean = torch.mean(z_flat, dim=0)
        z_std = torch.std(z_flat, dim=0)
        
        print(f"潜在编码统计:")
        print(f"  形状: {all_z.shape}")
        print(f"  均值范围: [{z_mean.min():.4f}, {z_mean.max():.4f}]")
        print(f"  标准差范围: [{z_std.min():.4f}, {z_std.max():.4f}]")
        print(f"  均值的标准差: {torch.std(z_mean):.4f}")
        
        # 检查是否有异常值
        z_abs_max = torch.abs(z_flat).max()
        print(f"  绝对值最大值: {z_abs_max:.4f}")
        
        if z_abs_max > 10:
            print("  ⚠️ 警告: 潜在编码值过大，可能存在梯度爆炸")
        elif z_abs_max < 0.1:
            print("  ⚠️ 警告: 潜在编码值过小，可能存在梯度消失")
        else:
            print("  ✅ 潜在编码值在合理范围内")

if __name__ == "__main__":
    test_latent_space_quality() 
