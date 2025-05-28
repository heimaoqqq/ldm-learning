#!/usr/bin/env python3
"""
测试VQ-VAE模型的压缩和重建能力
验证上游模型是否正确工作
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加路径
sys.path.append('VAE')
from adv_vq_vae import AdvVQVAE
from dataset import build_dataloader

def denormalize(x):
    """反归一化 [-1,1] -> [0,1]"""
    return (x + 1) / 2

def calculate_mse(x, y):
    """计算MSE"""
    return F.mse_loss(x, y).item()

def calculate_psnr(x, y):
    """计算PSNR"""
    mse = F.mse_loss(x, y)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def test_vqvae_reconstruction():
    """测试VQ-VAE重建质量"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径配置
    dataset_path = "/kaggle/input/dataset"  # Kaggle路径
    model_path = "/kaggle/input/adv-vqvae-best-fid-pth/adv_vqvae_best_fid.pth"
    
    # 检查路径是否存在
    if not os.path.exists(dataset_path):
        print(f"数据集路径不存在: {dataset_path}")
        return False
    
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        return False
    
    # 加载数据
    print("加载数据...")
    try:
        train_loader, val_loader, train_size, val_size = build_dataloader(
            dataset_path,
            batch_size=4,
            num_workers=2,
            shuffle_train=False,
            shuffle_val=False,
            val_split=0.2
        )
        print(f"数据加载成功: 训练集 {train_size}, 验证集 {val_size}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return False
    
    # 创建VQ-VAE模型
    print("创建VQ-VAE模型...")
    vqvae = AdvVQVAE(
        in_channels=3,
        latent_dim=256,
        num_embeddings=512,
        beta=0.25,
        decay=0.99,
        groups=32
    ).to(device)
    
    # 加载模型权重
    print("加载模型权重...")
    try:
        state_dict = torch.load(model_path, map_location=device)
        vqvae.load_state_dict(state_dict)
        print("模型权重加载成功")
    except Exception as e:
        print(f"模型权重加载失败: {e}")
        return False
    
    vqvae.eval()
    
    # 模型信息
    total_params = sum(p.numel() for p in vqvae.parameters())
    print(f"VQ-VAE参数量: {total_params:,}")
    
    # 测试重建质量
    print("\n测试重建质量...")
    all_mse = []
    all_psnr = []
    test_samples = 100
    
    with torch.no_grad():
        sample_count = 0
        for images, labels in tqdm(val_loader, desc="测试重建"):
            if sample_count >= test_samples:
                break
                
            images = images.to(device)
            batch_size = images.size(0)
            
            # VQ-VAE前向传播
            try:
                # 编码
                z_e = vqvae.encoder(images)
                print(f"编码器输出形状: {z_e.shape}")
                
                # 量化
                z_q, commitment_loss, codebook_loss = vqvae.vq(z_e)
                print(f"量化输出形状: {z_q.shape}")
                
                # 解码
                reconstructed = vqvae.decoder(z_q)
                print(f"重建输出形状: {reconstructed.shape}")
                
                # 计算重建质量
                for i in range(batch_size):
                    if sample_count >= test_samples:
                        break
                    
                    orig = images[i:i+1]
                    recon = reconstructed[i:i+1]
                    
                    mse = calculate_mse(orig, recon)
                    psnr = calculate_psnr(orig, recon)
                    
                    all_mse.append(mse)
                    all_psnr.append(psnr)
                    sample_count += 1
                
                # 保存前几个样本的对比
                if sample_count <= 8:
                    # 保存原始图像和重建图像的对比
                    orig_denorm = denormalize(images[:4])
                    recon_denorm = denormalize(reconstructed[:4])
                    
                    # 创建对比图像
                    comparison = torch.cat([orig_denorm, recon_denorm], dim=0)
                    grid = vutils.make_grid(comparison, nrow=4, normalize=False, padding=2)
                    
                    os.makedirs('vqvae_test_results', exist_ok=True)
                    vutils.save_image(grid, f'vqvae_test_results/reconstruction_batch_{len(all_mse)//4}.png')
                
                break  # 只处理第一个batch来打印形状信息
                
            except Exception as e:
                print(f"前向传播出错: {e}")
                return False
    
    # 统计结果
    avg_mse = np.mean(all_mse)
    avg_psnr = np.mean(all_psnr)
    std_mse = np.std(all_mse)
    std_psnr = np.std(all_psnr)
    
    print(f"\n重建质量统计 (n={len(all_mse)}):")
    print(f"  平均MSE: {avg_mse:.6f} ± {std_mse:.6f}")
    print(f"  平均PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"  MSE范围: {min(all_mse):.6f} - {max(all_mse):.6f}")
    print(f"  PSNR范围: {min(all_psnr):.2f} - {max(all_psnr):.2f} dB")
    
    # 评估重建质量
    print(f"\n质量评估:")
    if avg_psnr > 20:
        print("✓ VQ-VAE重建质量良好 (PSNR > 20 dB)")
        quality_good = True
    elif avg_psnr > 15:
        print("⚠ VQ-VAE重建质量一般 (15 < PSNR < 20 dB)")
        quality_good = True
    else:
        print("✗ VQ-VAE重建质量差 (PSNR < 15 dB)")
        quality_good = False
    
    # 测试潜在空间形状
    print(f"\n潜在空间测试:")
    with torch.no_grad():
        test_input = torch.randn(1, 3, 128, 128).to(device)
        z_e = vqvae.encoder(test_input)
        z_q, _, _ = vqvae.vq(z_e)
        output = vqvae.decoder(z_q)
        
        print(f"  输入形状: {test_input.shape}")
        print(f"  潜在形状: {z_q.shape}")
        print(f"  输出形状: {output.shape}")
        
        # 验证维度匹配
        expected_latent_shape = (1, 256, 16, 16)
        if z_q.shape == expected_latent_shape:
            print("✓ 潜在空间形状正确")
            shape_correct = True
        else:
            print(f"✗ 潜在空间形状错误，期望 {expected_latent_shape}，实际 {z_q.shape}")
            shape_correct = False
    
    # 总结
    print(f"\n=== VQ-VAE模型测试总结 ===")
    print(f"重建质量: {'✓ 良好' if quality_good else '✗ 差'}")
    print(f"潜在形状: {'✓ 正确' if shape_correct else '✗ 错误'}")
    
    success = quality_good and shape_correct
    if success:
        print("✓ VQ-VAE模型工作正常，可以用于CLDM训练")
    else:
        print("✗ VQ-VAE模型存在问题，需要检查")
    
    return success

if __name__ == "__main__":
    test_vqvae_reconstruction() 