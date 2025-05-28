#!/usr/bin/env python3
"""
诊断FID问题的脚本
分析VQ-VAE重建质量和CLDM生成质量
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
sys.path.append('LDM')
from adv_vq_vae import AdvVQVAE
from dataset import build_dataloader
from cldm_optimized import ImprovedCLDM

def denormalize(x):
    """反归一化 [-1,1] -> [0,1]"""
    return (x + 1) / 2

def calculate_mse(x, y):
    """计算MSE"""
    return F.mse_loss(x, y).item()

def analyze_vqvae_quality():
    """分析VQ-VAE质量"""
    print("=== 分析VQ-VAE质量 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 配置路径
    dataset_path = "/kaggle/input/dataset"
    model_path = "/kaggle/input/adv-vqvae-best-fid-pth/adv_vqvae_best_fid.pth"
    
    if not os.path.exists(dataset_path) or not os.path.exists(model_path):
        print("路径不存在，使用本地路径测试")
        dataset_path = "../dataset"
        model_path = "../VAE/checkpoints/adv_vqvae_best_fid.pth"
    
    # 加载数据
    try:
        train_loader, val_loader, train_size, val_size = build_dataloader(
            dataset_path, batch_size=8, num_workers=2, val_split=0.2
        )
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None
    
    # 加载VQ-VAE
    vqvae = AdvVQVAE(
        in_channels=3, latent_dim=256, num_embeddings=512,
        beta=0.25, decay=0.99, groups=32
    ).to(device)
    
    try:
        vqvae.load_state_dict(torch.load(model_path, map_location=device))
        print("VQ-VAE模型加载成功")
    except Exception as e:
        print(f"VQ-VAE模型加载失败: {e}")
        return None
    
    vqvae.eval()
    
    # 测试重建质量
    reconstruction_errors = []
    codebook_usage = set()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if i >= 10:  # 只测试前10个batch
                break
                
            images = images.to(device)
            
            # VQ-VAE重建
            z_e = vqvae.encoder(images)
            z_q, _, codebook_indices = vqvae.vq(z_e)
            reconstructed = vqvae.decoder(z_q)
            
            # 计算重建误差
            mse = calculate_mse(images, reconstructed)
            reconstruction_errors.append(mse)
            
            # 记录codebook使用情况
            unique_indices = torch.unique(codebook_indices).cpu().numpy()
            codebook_usage.update(unique_indices)
            
            # 保存第一个batch的对比图
            if i == 0:
                orig_grid = vutils.make_grid(denormalize(images[:4]), nrow=2, normalize=False)
                recon_grid = vutils.make_grid(denormalize(reconstructed[:4]), nrow=2, normalize=False)
                
                os.makedirs('diagnosis_results', exist_ok=True)
                vutils.save_image(orig_grid, 'diagnosis_results/vqvae_original.png')
                vutils.save_image(recon_grid, 'diagnosis_results/vqvae_reconstructed.png')
    
    avg_recon_error = np.mean(reconstruction_errors)
    codebook_utilization = len(codebook_usage) / 512  # 总共512个embedding
    
    print(f"VQ-VAE重建MSE: {avg_recon_error:.6f}")
    print(f"Codebook利用率: {codebook_utilization:.2%} ({len(codebook_usage)}/512)")
    
    # 判断VQ-VAE质量
    vqvae_quality = "好" if avg_recon_error < 0.01 and codebook_utilization > 0.1 else "差"
    print(f"VQ-VAE整体质量: {vqvae_quality}")
    
    return {
        'vqvae': vqvae,
        'val_loader': val_loader,
        'reconstruction_error': avg_recon_error,
        'codebook_utilization': codebook_utilization,
        'quality': vqvae_quality
    }

def test_cldm_sampling(vqvae_result):
    """测试CLDM采样质量"""
    print("\n=== 测试CLDM采样质量 ===")
    
    if vqvae_result is None:
        print("跳过CLDM测试：VQ-VAE加载失败")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vqvae = vqvae_result['vqvae']
    
    # 创建CLDM模型
    cldm = ImprovedCLDM(
        latent_dim=256,
        num_classes=31,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        noise_schedule="cosine",
        time_emb_dim=128,
        class_emb_dim=64,
        base_channels=64,
        dropout=0.1,
        groups=32
    ).to(device)
    
    # 检查是否有预训练模型
    model_paths = [
        "/kaggle/working/cldm_optimized/cldm_best_fid.pth",
        "LDM/cldm_best_fid.pth",
        "/kaggle/working/cldm_optimized/cldm_best_loss.pth",
        "LDM/cldm_best_loss.pth"
    ]
    
    model_loaded = False
    for path in model_paths:
        if os.path.exists(path):
            try:
                cldm.load_state_dict(torch.load(path, map_location=device))
                print(f"CLDM模型加载成功: {path}")
                model_loaded = True
                break
            except Exception as e:
                continue
    
    if not model_loaded:
        print("未找到预训练CLDM模型，使用随机初始化")
        print("注意：随机初始化的模型生成质量会很差")
    
    cldm.eval()
    
    # 测试不同采样步数
    sampling_steps_list = [20, 50, 100, 200]
    
    for steps in sampling_steps_list:
        print(f"\n测试采样步数: {steps}")
        
        # 生成几个类别的样本
        test_classes = [0, 1, 2, 3]  # 测试前4个类别
        batch_size = len(test_classes)
        class_labels = torch.tensor(test_classes, device=device)
        
        with torch.no_grad():
            # CLDM采样
            generated_z = cldm.sample(
                class_labels, device, 
                sampling_steps=steps, 
                eta=0.0  # 确定性采样
            )
            
            # VQ-VAE解码
            generated_images = vqvae.decoder(generated_z)
            
            # 计算生成图像的统计信息
            gen_mean = generated_images.mean().item()
            gen_std = generated_images.std().item()
            gen_min = generated_images.min().item()
            gen_max = generated_images.max().item()
            
            print(f"  生成图像统计: 均值={gen_mean:.3f}, 标准差={gen_std:.3f}")
            print(f"  生成图像范围: [{gen_min:.3f}, {gen_max:.3f}]")
            
            # 保存生成图像
            gen_grid = vutils.make_grid(
                denormalize(generated_images), 
                nrow=2, normalize=False, padding=2
            )
            vutils.save_image(
                gen_grid, 
                f'diagnosis_results/cldm_generated_steps_{steps}.png'
            )
            
            # 检查生成图像是否合理
            if abs(gen_mean) > 2 or gen_std > 3:
                print(f"  ⚠ 生成图像统计异常，可能存在梯度爆炸或模式崩塌")
            else:
                print(f"  ✓ 生成图像统计正常")

def analyze_training_dynamics():
    """分析训练动态"""
    print("\n=== 分析训练动态建议 ===")
    
    print("基于常见FID问题的建议:")
    print("1. 采样步数: 建议使用100-200步，当前20-50步可能不足")
    print("2. 噪声调度: 尝试linear调度而非cosine")
    print("3. 模型容量: 当前base_channels=64可能过小，尝试128或256")
    print("4. 训练轮数: 确保充分训练，至少100-200轮")
    print("5. 学习率: 当前1e-4可能过大，尝试5e-5或1e-5")
    print("6. VQ-VAE质量: 确保VQ-VAE重建质量足够好")

def suggest_improvements():
    """提出改进建议"""
    print("\n=== 改进建议 ===")
    
    print("基于诊断结果的具体改进方案:")
    print("1. 增加采样步数到100以上")
    print("2. 使用更保守的学习率(5e-5)")
    print("3. 增加模型容量(base_channels=128)")
    print("4. 使用EMA (Exponential Moving Average)") 
    print("5. 增加训练时间")
    print("6. 检查数据预处理是否正确")
    print("7. 考虑添加classifier-free guidance")

def main():
    """主函数"""
    print("开始诊断FID问题...")
    
    # 创建结果目录
    os.makedirs('diagnosis_results', exist_ok=True)
    
    # 1. 分析VQ-VAE质量
    vqvae_result = analyze_vqvae_quality()
    
    # 2. 测试CLDM采样
    test_cldm_sampling(vqvae_result)
    
    # 3. 分析训练动态
    analyze_training_dynamics()
    
    # 4. 提出改进建议
    suggest_improvements()
    
    print(f"\n诊断完成，结果保存在 diagnosis_results/ 目录")

if __name__ == "__main__":
    main() 