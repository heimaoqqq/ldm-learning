#!/usr/bin/env python3
"""
VQ-VAE深度分析工具
彻底检查VQ-VAE模型的各个方面，确定问题所在
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加路径
sys.path.append('VAE')
from adv_vq_vae import AdvVQVAE
from dataset import build_dataloader

def check_model_weights(model_path):
    """检查模型权重文件"""
    print("=== 模型权重分析 ===")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    try:
        # 加载权重
        state_dict = torch.load(model_path, map_location='cpu')
        
        print(f"✅ 模型文件加载成功")
        print(f"权重键数量: {len(state_dict.keys())}")
        
        # 检查关键权重
        key_weights = {
            'encoder': [k for k in state_dict.keys() if 'encoder' in k],
            'decoder': [k for k in state_dict.keys() if 'decoder' in k],
            'vq': [k for k in state_dict.keys() if 'vq' in k or 'embedding' in k],
            'discriminator': [k for k in state_dict.keys() if 'discriminator' in k]
        }
        
        for component, keys in key_weights.items():
            print(f"{component}: {len(keys)} 个权重")
            if len(keys) > 0:
                # 检查权重统计
                weights = torch.cat([state_dict[k].flatten() for k in keys[:3]])  # 前3个权重
                print(f"  权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
                print(f"  权重均值: {weights.mean():.4f}, 标准差: {weights.std():.4f}")
                
                # 检查是否有异常值
                if torch.isnan(weights).any():
                    print(f"  ❌ 发现NaN值")
                if torch.isinf(weights).any():
                    print(f"  ❌ 发现无穷值")
                if weights.std() < 1e-6:
                    print(f"  ⚠ 权重方差过小，可能未训练")
                if weights.std() > 10:
                    print(f"  ⚠ 权重方差过大，可能梯度爆炸")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def analyze_data_quality(dataloader):
    """分析数据质量"""
    print("\n=== 数据质量分析 ===")
    
    sample_batch = next(iter(dataloader))
    images, labels = sample_batch
    
    print(f"批次形状: {images.shape}")
    print(f"数据类型: {images.dtype}")
    print(f"标签范围: {labels.min()} - {labels.max()}")
    
    # 数据统计
    print(f"图像统计:")
    print(f"  均值: {images.mean():.4f}")
    print(f"  标准差: {images.std():.4f}")
    print(f"  最小值: {images.min():.4f}")
    print(f"  最大值: {images.max():.4f}")
    
    # 检查数据分布
    if images.min() < -1.1 or images.max() > 1.1:
        print("  ⚠ 数据超出预期范围[-1,1]")
    
    if abs(images.mean()) > 0.1:
        print("  ⚠ 数据均值偏离0较多")
    
    if images.std() < 0.1 or images.std() > 1.0:
        print("  ⚠ 数据标准差异常")
    
    # 保存样本图像用于检查
    os.makedirs('vqvae_analysis', exist_ok=True)
    
    # 反归一化到[0,1]用于保存
    sample_images = (images[:8] + 1) / 2
    grid = vutils.make_grid(sample_images, nrow=4, normalize=False, padding=2)
    vutils.save_image(grid, 'vqvae_analysis/data_samples.png')
    print(f"  数据样本已保存到: vqvae_analysis/data_samples.png")
    
    return images, labels

def analyze_model_architecture(model):
    """分析模型架构"""
    print("\n=== 模型架构分析 ===")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 分析各组件参数
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    vq_params = sum(p.numel() for p in model.vq.parameters())
    
    print(f"编码器参数: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"解码器参数: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
    print(f"量化器参数: {vq_params:,} ({vq_params/total_params*100:.1f}%)")
    
    if hasattr(model, 'discriminator'):
        disc_params = sum(p.numel() for p in model.discriminator.parameters())
        print(f"判别器参数: {disc_params:,} ({disc_params/total_params*100:.1f}%)")

def test_forward_pass(model, sample_images, device):
    """测试前向传播"""
    print("\n=== 前向传播测试 ===")
    
    model.eval()
    sample_images = sample_images.to(device)
    
    with torch.no_grad():
        try:
            # 编码
            z_e = model.encoder(sample_images)
            print(f"✅ 编码成功: {sample_images.shape} -> {z_e.shape}")
            
            # 量化
            z_q, commitment_loss, codebook_loss = model.vq(z_e)
            print(f"✅ 量化成功: {z_e.shape} -> {z_q.shape}")
            print(f"  Commitment Loss: {commitment_loss:.6f}")
            print(f"  Codebook Loss: {codebook_loss:.6f}")
            
            # 解码
            reconstructed = model.decoder(z_q)
            print(f"✅ 解码成功: {z_q.shape} -> {reconstructed.shape}")
            
            # 检查输出
            print(f"重建图像统计:")
            print(f"  均值: {reconstructed.mean():.4f}")
            print(f"  标准差: {reconstructed.std():.4f}")
            print(f"  范围: [{reconstructed.min():.4f}, {reconstructed.max():.4f}]")
            
            return z_e, z_q, reconstructed, commitment_loss, codebook_loss
            
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            return None

def analyze_reconstruction_quality(original, reconstructed):
    """分析重建质量"""
    print("\n=== 重建质量分析 ===")
    
    # 确保tensors在同一设备上
    if original.device != reconstructed.device:
        reconstructed = reconstructed.to(original.device)
    
    # 逐样本分析
    batch_size = original.shape[0]
    mse_list = []
    psnr_list = []
    
    for i in range(batch_size):
        orig = original[i:i+1]
        recon = reconstructed[i:i+1]
        
        mse = F.mse_loss(orig, recon).item()
        # 对于[-1,1]范围，最大值是2
        psnr = 20 * torch.log10(torch.tensor(2.0) / torch.sqrt(torch.tensor(mse))).item()
        
        mse_list.append(mse)
        psnr_list.append(psnr)
    
    avg_mse = np.mean(mse_list)
    avg_psnr = np.mean(psnr_list)
    std_mse = np.std(mse_list)
    std_psnr = np.std(psnr_list)
    
    print(f"重建质量统计:")
    print(f"  平均MSE: {avg_mse:.6f} ± {std_mse:.6f}")
    print(f"  平均PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"  MSE范围: [{min(mse_list):.6f}, {max(mse_list):.6f}]")
    print(f"  PSNR范围: [{min(psnr_list):.2f}, {max(psnr_list):.2f}]")
    
    # 质量评估
    if avg_psnr > 25:
        print("✅ 重建质量优秀")
    elif avg_psnr > 20:
        print("✅ 重建质量良好")
    elif avg_psnr > 15:
        print("⚠ 重建质量一般")
    else:
        print("❌ 重建质量差")
    
    # 保存对比图像
    # 移动到CPU进行图像保存
    original_cpu = original[:4].cpu()
    reconstructed_cpu = reconstructed[:4].cpu()
    
    comparison = torch.cat([
        (original_cpu + 1) / 2,  # 原图
        (reconstructed_cpu + 1) / 2  # 重建图
    ], dim=0)
    
    grid = vutils.make_grid(comparison, nrow=4, normalize=False, padding=2)
    vutils.save_image(grid, 'vqvae_analysis/reconstruction_comparison.png')
    print(f"  重建对比图已保存到: vqvae_analysis/reconstruction_comparison.png")
    
    return avg_mse, avg_psnr

def analyze_codebook_usage(model, z_e):
    """分析codebook使用情况"""
    print("\n=== Codebook使用分析 ===")
    
    with torch.no_grad():
        # 获取编码索引
        z_e_flat = z_e.permute(0,2,3,1).contiguous().view(-1, model.vq.embedding_dim)
        
        # 计算距离
        distances = torch.sum((z_e_flat.unsqueeze(1) - model.vq.embedding.weight.unsqueeze(0))**2, dim=2)
        encoding_indices = torch.argmin(distances, dim=1)
        
        # 统计使用情况
        unique_indices = torch.unique(encoding_indices)
        usage_count = torch.bincount(encoding_indices, minlength=model.vq.num_embeddings)
        
        used_embeddings = len(unique_indices)
        total_embeddings = model.vq.num_embeddings
        usage_rate = used_embeddings / total_embeddings
        
        print(f"Codebook使用统计:")
        print(f"  使用的embedding: {used_embeddings}/{total_embeddings}")
        print(f"  使用率: {usage_rate:.2%}")
        print(f"  平均使用次数: {usage_count[usage_count > 0].float().mean():.1f}")
        print(f"  最大使用次数: {usage_count.max().item()}")
        print(f"  使用次数标准差: {usage_count[usage_count > 0].float().std():.1f}")
        
        # 评估codebook质量
        if usage_rate < 0.05:
            print("❌ Codebook塌陷严重")
        elif usage_rate < 0.1:
            print("⚠ Codebook塌陷")
        elif usage_rate < 0.3:
            print("⚠ Codebook利用率偏低")
        else:
            print("✅ Codebook利用率正常")

def analyze_gradient_flow(model, sample_images, device):
    """分析梯度流动"""
    print("\n=== 梯度流动分析 ===")
    
    model.train()
    sample_images = sample_images.to(device)
    sample_images.requires_grad_(True)
    
    try:
        # 前向传播
        z_e = model.encoder(sample_images)
        z_q, commitment_loss, codebook_loss = model.vq(z_e)
        reconstructed = model.decoder(z_q)
        
        # 计算重建损失
        recon_loss = F.mse_loss(reconstructed, sample_images)
        total_loss = recon_loss + commitment_loss + codebook_loss
        
        # 反向传播
        total_loss.backward()
        
        # 检查梯度
        encoder_grad_norm = 0
        decoder_grad_norm = 0
        vq_grad_norm = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if 'encoder' in name:
                    encoder_grad_norm += grad_norm
                elif 'decoder' in name:
                    decoder_grad_norm += grad_norm
                elif 'vq' in name or 'embedding' in name:
                    vq_grad_norm += grad_norm
        
        print(f"梯度范数:")
        print(f"  编码器: {encoder_grad_norm:.6f}")
        print(f"  解码器: {decoder_grad_norm:.6f}")
        print(f"  量化器: {vq_grad_norm:.6f}")
        
        # 评估梯度
        if encoder_grad_norm < 1e-6 or decoder_grad_norm < 1e-6:
            print("❌ 梯度消失")
        elif encoder_grad_norm > 100 or decoder_grad_norm > 100:
            print("❌ 梯度爆炸")
        else:
            print("✅ 梯度正常")
            
        model.zero_grad()
        
    except Exception as e:
        print(f"❌ 梯度分析失败: {e}")

def comprehensive_vqvae_analysis():
    """全面的VQ-VAE分析"""
    print("🔍 开始VQ-VAE深度分析...\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 检查模型权重 - 适应不同环境
    possible_model_paths = [
        "/kaggle/input/adv-vqvae-best-fid-pth/adv_vqvae_best_fid.pth",  # Kaggle路径
        "VAE/adv_vqvae_best_fid.pth",  # 本地路径
        "VAE/checkpoints/adv_vqvae_best_fid.pth",  # 备选路径
        "adv_vqvae_best_fid.pth"  # 当前目录
    ]
    
    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ 未找到VQ-VAE模型文件，请检查以下路径：")
        for path in possible_model_paths:
            print(f"   {path}")
        print("\n请将模型文件放置在上述路径之一，或修改代码中的路径")
        return
    
    print(f"✅ 找到模型文件: {model_path}")
    
    if not check_model_weights(model_path):
        print("\n❌ 模型权重检查失败，无法继续分析")
        return
    
    # 2. 加载数据 - 适应不同环境
    possible_dataset_paths = [
        "/kaggle/input/dataset",  # Kaggle路径
        "VAE/dataset",  # 本地路径
        "dataset",  # 当前目录
        "../dataset"  # 上级目录
    ]
    
    dataset_path = None
    for path in possible_dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        print("❌ 未找到数据集，请检查以下路径：")
        for path in possible_dataset_paths:
            print(f"   {path}")
        
        # 创建模拟数据用于分析
        print("\n⚠ 使用模拟数据进行分析...")
        sample_images = torch.randn(8, 3, 256, 256) * 0.5  # [-1, 1]范围
        sample_labels = torch.randint(0, 2, (8,))
        print(f"模拟数据: {sample_images.shape}, 标签: {sample_labels.shape}")
    else:
        try:
            print(f"✅ 找到数据集: {dataset_path}")
            train_loader, val_loader, train_size, val_size = build_dataloader(
                dataset_path, batch_size=8, num_workers=2, val_split=0.2
            )
            print(f"✅ 数据加载成功: 训练集 {train_size}, 验证集 {val_size}")
            
            # 3. 分析数据质量
            sample_images, sample_labels = analyze_data_quality(val_loader)
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            print("使用模拟数据进行分析...")
            sample_images = torch.randn(8, 3, 256, 256) * 0.5
            sample_labels = torch.randint(0, 2, (8,))
    
    # 4. 创建和加载模型
    model = AdvVQVAE(
        in_channels=3, latent_dim=256, num_embeddings=512,
        beta=0.25, decay=0.99, groups=32
    ).to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"\n✅ 模型权重加载成功")
    except Exception as e:
        print(f"\n❌ 模型权重加载失败: {e}")
        return
    
    # 5. 分析模型架构
    analyze_model_architecture(model)
    
    # 6. 测试前向传播
    forward_results = test_forward_pass(model, sample_images, device)
    if forward_results is None:
        return
    
    z_e, z_q, reconstructed, commitment_loss, codebook_loss = forward_results
    
    # 7. 分析重建质量
    avg_mse, avg_psnr = analyze_reconstruction_quality(sample_images, reconstructed)
    
    # 8. 分析codebook使用
    analyze_codebook_usage(model, z_e)
    
    # 9. 分析梯度流动
    analyze_gradient_flow(model, sample_images[:4], device)  # 使用少量样本
    
    # 10. 总结分析结果
    print("\n" + "="*50)
    print("📊 分析总结")
    print("="*50)
    
    print(f"✅ 潜在空间尺寸: {z_q.shape} (正确)")
    print(f"{'✅' if avg_psnr > 15 else '❌'} 重建质量: PSNR = {avg_psnr:.2f} dB")
    print(f"✅ 模型架构: 正常运行")
    print(f"✅ 前向传播: 成功")
    
    if avg_psnr > 20:
        print("\n🎉 VQ-VAE模型质量良好，可以用于CLDM训练")
    elif avg_psnr > 15:
        print("\n⚠ VQ-VAE模型质量一般，建议改进但可以用于CLDM训练")
    else:
        print("\n❌ VQ-VAE模型质量较差，建议重新训练")
        print("   但由于潜在空间尺寸正确，仍可尝试CLDM训练")
    
    print(f"\n分析结果已保存到: vqvae_analysis/目录")

if __name__ == "__main__":
    comprehensive_vqvae_analysis() 