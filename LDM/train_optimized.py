#!/usr/bin/env python3
"""
优化版本CLDM训练脚本
专门用于提升FID分数，基于修复版本的模型架构
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import math
import shutil
import numpy as np
import yaml
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
sys.path.append('/kaggle/input/advanced-vqvae')
from ae_debug.adv_vq_vae import AdvVQVAE
from build_dataloader import build_dataloader

# 导入优化版本的CLDM
from cldm_optimized import ImprovedCLDM

# 检查依赖
HAS_AMP = True
try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    HAS_AMP = False
    print("⚠️ 无法导入AMP，使用常规精度训练")

HAS_FIDELITY = True
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    import torchmetrics
except ImportError:
    HAS_FIDELITY = False
    print("⚠️ 无法导入torchmetrics，将跳过FID计算")

def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        sys.exit(1)

def set_seed(seed):
    """设置随机种子确保可重现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 设置随机种子: {seed}")

def denormalize(x):
    """反归一化 [-1,1] -> [0,1]"""
    return (x + 1) / 2

def extract_latent_features(vqvae_model, dataloader, device, max_samples=None):
    """从VQ-VAE提取潜在特征"""
    vqvae_model.eval()
    all_latents = []
    all_labels = []
    
    sample_count = 0
    print("📊 提取VQ-VAE潜在特征...")
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="提取特征"):
            if max_samples and sample_count >= max_samples:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            # VQ-VAE编码
            z_e = vqvae_model.encoder(images)
            z_q, _, _ = vqvae_model.quantize(z_e)
            
            all_latents.append(z_q.cpu())
            all_labels.append(labels.cpu())
            
            sample_count += images.size(0)
    
    latents = torch.cat(all_latents, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    print(f"✅ 提取完成: {latents.shape[0]} 个潜在特征，形状: {latents.shape}")
    return latents, labels

def calculate_fid_score(cldm_model, vqvae_model, val_loader, config, device):
    """计算高质量FID分数"""
    if not HAS_FIDELITY:
        return float('inf')
    
    try:
        print("🧮 计算FID分数...")
        
        # 初始化FID度量
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)
        fid_metric = fid_metric.to(device)
        
        cldm_model.eval()
        vqvae_model.eval()
        
        sample_size = config['training']['num_sample_images']
        sampling_steps = config['training']['fid_sampling_steps']
        eta = config['training']['eta']
        
        real_images = []
        generated_images = []
        sample_count = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="FID采样"):
                if sample_count >= sample_size:
                    break
                    
                images = images.to(device)
                labels = labels.to(device)
                batch_size = images.size(0)
                
                # 生成对应图像
                generated_z = cldm_model.sample(
                    labels, device, sampling_steps=sampling_steps, eta=eta
                )
                gen_images = vqvae_model.decoder(generated_z)
                
                # 转换为[0,255]格式
                real_uint8 = ((denormalize(images) * 255).clamp(0, 255).to(torch.uint8))
                gen_uint8 = ((denormalize(gen_images) * 255).clamp(0, 255).to(torch.uint8))
                
                real_images.append(real_uint8.cpu())
                generated_images.append(gen_uint8.cpu())
                
                sample_count += batch_size
                
            # 合并所有图像
            real_all = torch.cat(real_images, dim=0)[:sample_size]
            gen_all = torch.cat(generated_images, dim=0)[:sample_size]
            
            print(f"FID评估: 真实图像 {real_all.shape[0]}, 生成图像 {gen_all.shape[0]}")
            
            # 计算FID
            fid_metric.update(real_all.to(device), real=True)
            fid_metric.update(gen_all.to(device), real=False)
            fid_score = fid_metric.compute().item()
        
        cldm_model.train()
        return fid_score
        
    except Exception as e:
        print(f"❌ FID计算出错: {e}")
        return float('inf')

def sample_and_save(cldm_model, vqvae_model, config, device, epoch, save_dir):
    """生成并保存采样图像"""
    cldm_model.eval()
    vqvae_model.eval()
    
    num_classes = config['dataset']['num_classes']
    samples_per_class = 2
    
    print(f"🎨 生成第{epoch}轮采样图像...")
    
    all_samples = []
    
    with torch.no_grad():
        for class_id in range(min(16, num_classes)):  # 限制显示前16个类别
            class_labels = torch.full((samples_per_class,), class_id, device=device)
            
            # 高质量采样
            generated_z = cldm_model.sample(
                class_labels, 
                device, 
                sampling_steps=config['training']['sampling_steps'],
                eta=config['training']['eta']
            )
            
            # 解码为图像
            generated_images = vqvae_model.decoder(generated_z)
            generated_images = denormalize(generated_images)
            
            all_samples.append(generated_images)
    
    # 保存图像网格
    all_samples = torch.cat(all_samples, dim=0)
    grid = vutils.make_grid(all_samples, nrow=samples_per_class, normalize=False, padding=2)
    
    os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
    vutils.save_image(grid, os.path.join(save_dir, 'samples', f'epoch_{epoch:03d}.png'))
    
    cldm_model.train()

def main():
    parser = argparse.ArgumentParser(description="优化版本CLDM训练")
    parser.add_argument('--config', default='config_optimized.yaml', help='配置文件路径')
    parser.add_argument('--resume', default=None, help='继续训练的checkpoint路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置环境
    set_seed(config['system']['seed'])
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建保存目录
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    
    # 保存配置副本
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # 构建数据加载器
    print("📚 加载数据集...")
    train_loader, val_loader, train_size, val_size = build_dataloader(
        config['dataset']['root_dir'],
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        shuffle_train=True,
        shuffle_val=False,
        val_split=0.3
    )
    
    print(f"📊 数据集统计:")
    print(f"  训练集: {train_size} 样本")
    print(f"  验证集: {val_size} 样本")
    print(f"  类别数: {config['dataset']['num_classes']}")
    print(f"  批次大小: {config['dataset']['batch_size']}")
    
    # 加载VQ-VAE模型
    print("🔧 加载VQ-VAE模型...")
    vqvae = AdvVQVAE(
        in_channels=config['vqvae']['in_channels'],
        latent_dim=config['vqvae']['latent_dim'],
        num_embeddings=config['vqvae']['num_embeddings'],
        beta=config['vqvae']['beta'],
        decay=config['vqvae']['decay'],
        groups=config['vqvae']['groups']
    ).to(device)
    
    vqvae_path = config['vqvae']['model_path']
    if os.path.exists(vqvae_path):
        vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
        print(f"✅ VQ-VAE权重加载成功")
    else:
        print(f"❌ VQ-VAE权重文件不存在: {vqvae_path}")
        return
    
    # 冻结VQ-VAE
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False
    
    # 创建优化版本CLDM模型
    print("🚀 创建优化版CLDM模型...")
    cldm = ImprovedCLDM(
        latent_dim=config['cldm']['latent_dim'],
        num_classes=config['cldm']['num_classes'],
        num_timesteps=config['cldm']['num_timesteps'],
        beta_start=config['cldm']['beta_start'],
        beta_end=config['cldm']['beta_end'],
        noise_schedule=config['cldm']['noise_schedule'],
        time_emb_dim=config['cldm']['time_emb_dim'],
        class_emb_dim=config['cldm']['class_emb_dim'],
        base_channels=config['cldm']['base_channels'],
        dropout=config['cldm']['dropout'],
        groups=config['cldm']['groups']
    ).to(device)
    
    # 模型统计
    total_params = sum(p.numel() for p in cldm.parameters())
    trainable_params = sum(p.numel() for p in cldm.parameters() if p.requires_grad)
    print(f"📈 模型统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # 优化器
    optimizer = optim.AdamW(
        cldm.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    if config['training']['use_scheduler']:
        def lr_lambda(epoch):
            warmup_epochs = config['training']['warmup_epochs']
            total_epochs = config['training']['epochs']
            min_lr_ratio = config['training']['min_lr'] / config['training']['lr']
            
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            else:
                progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print("📚 使用Cosine学习率调度器")
    else:
        scheduler = None
    
    # 混合精度训练
    use_amp = config['system']['mixed_precision'] and HAS_AMP
    if use_amp:
        scaler = GradScaler()
        print("⚡ 启用混合精度训练")
    else:
        scaler = None
    
    # 恢复训练
    start_epoch = 0
    best_fid = float('inf')
    best_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"🔄 从checkpoint恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        cldm.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint.get('best_fid', float('inf'))
        best_loss = checkpoint.get('best_loss', float('inf'))
    
    # 提取训练数据的潜在特征
    train_latents, train_labels = extract_latent_features(vqvae, train_loader, device)
    
    # 创建潜在特征数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    latent_dataset = TensorDataset(train_latents, train_labels)
    latent_loader = DataLoader(
        latent_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # 训练循环
    print(f"🎯 开始训练优化版CLDM...")
    print(f"训练轮数: {config['training']['epochs']}")
    print(f"从第 {start_epoch + 1} 轮开始")
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # 训练阶段
        cldm.train()
        running_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(latent_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        for z_0, class_labels in pbar:
            z_0 = z_0.to(device)
            class_labels = class_labels.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    loss = cldm(z_0, class_labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(cldm.parameters(), config['training']['grad_clip_norm'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = cldm(z_0, class_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cldm.parameters(), config['training']['grad_clip_norm'])
                optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        avg_train_loss = running_loss / num_batches
        
        # 学习率调度
        if scheduler:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 验证阶段
        if (epoch + 1) % config['training']['val_interval'] == 0:
            cldm.eval()
            val_loss = 0.0
            val_batches = 0
            
            # 验证集潜在特征
            val_latents, val_labels = extract_latent_features(vqvae, val_loader, device, max_samples=1000)
            val_dataset = TensorDataset(val_latents, val_labels)
            val_latent_loader = DataLoader(val_dataset, batch_size=config['dataset']['batch_size'], shuffle=False)
            
            with torch.no_grad():
                for z_0, class_labels in val_latent_loader:
                    z_0 = z_0.to(device)
                    class_labels = class_labels.to(device)
                    
                    if use_amp:
                        with autocast():
                            loss = cldm(z_0, class_labels)
                    else:
                        loss = cldm(z_0, class_labels)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            
            print(f"\n📊 Epoch {epoch+1} 结果:")
            print(f"  训练损失: {avg_train_loss:.6f}")
            print(f"  验证损失: {avg_val_loss:.6f}")
            print(f"  学习率: {current_lr:.8f}")
            
            # 保存最佳损失模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(cldm.state_dict(), os.path.join(save_dir, 'cldm_best_loss.pth'))
                print(f"  ✅ 保存最佳损失模型: {best_loss:.6f}")
        else:
            print(f"\nEpoch {epoch+1}: 训练损失 {avg_train_loss:.6f}, 学习率 {current_lr:.8f}")
        
        # FID评估
        if (epoch + 1) % config['training']['fid_eval_freq'] == 0 and HAS_FIDELITY:
            fid_score = calculate_fid_score(cldm, vqvae, val_loader, config, device)
            print(f"  🎯 FID分数: {fid_score:.4f}")
            
            if fid_score < best_fid:
                best_fid = fid_score
                torch.save(cldm.state_dict(), os.path.join(save_dir, 'cldm_best_fid.pth'))
                print(f"  🏆 保存最佳FID模型: {best_fid:.4f}")
        
        # 生成采样
        if (epoch + 1) % config['training']['sample_interval'] == 0:
            sample_and_save(cldm, vqvae, config, device, epoch + 1, save_dir)
        
        # 保存checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': cldm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_fid': best_fid,
                'best_loss': best_loss,
                'config': config
            }
            if scheduler:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint, os.path.join(save_dir, 'checkpoints', f'epoch_{epoch+1:03d}.pth'))
            print(f"  💾 保存checkpoint")
    
    # 训练完成
    print(f"\n🎉 训练完成！")
    print(f"最佳验证损失: {best_loss:.6f}")
    if HAS_FIDELITY:
        print(f"最佳FID分数: {best_fid:.4f}")
    
    # 最终采样
    sample_and_save(cldm, vqvae, config, device, "final", save_dir)
    print(f"🎨 最终样本已保存")

if __name__ == "__main__":
    main() 
