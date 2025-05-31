#!/usr/bin/env python3
"""
基于Stable Diffusion的轻量化LDM训练脚本
高效训练，快速收敛，专门适配您的VAE
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import time
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from typing import Dict, Any
import json

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VAE'))

from stable_ldm import StableLDM, create_stable_ldm
from dataset import build_dataloader
from fid_evaluation import FIDEvaluator
from metrics import DiffusionMetrics, denormalize_for_metrics

def denormalize(x):
    """反归一化图像，从[-1,1]转换到[0,1]"""
    return (x * 0.5 + 0.5).clamp(0, 1)

def save_sample_images(model, device, config, epoch, save_dir):
    """生成并保存样本图像"""
    model.eval()
    with torch.no_grad():
        num_classes = config['unet']['num_classes']
        num_samples_per_class = min(2, config['inference']['num_samples_per_class'])
        
        all_images = []
        for class_id in range(min(8, num_classes)):  # 最多显示8个类别
            class_labels = torch.tensor([class_id] * num_samples_per_class, device=device)
            
            generated_images = model.sample(
                batch_size=num_samples_per_class,
                class_labels=class_labels,
                num_inference_steps=config['inference']['num_inference_steps'],
                guidance_scale=config['inference']['guidance_scale'],
                eta=config['inference']['eta'],
                use_ema=config.get('use_ema', False),
                verbose=True
            )
            
            all_images.append(generated_images)
        
        # 拼接所有图像
        if all_images:
            all_images = torch.cat(all_images, dim=0)
            
            # 保存图像网格
            grid = vutils.make_grid(all_images, nrow=num_samples_per_class, normalize=False, padding=2)
            save_path = os.path.join(save_dir, f"samples_epoch_{epoch:04d}.png")
            vutils.save_image(grid, save_path)
            print(f"  样本图像已保存: {save_path}")

def save_model_checkpoint(model, save_path, epoch=None, extra_info=None):
    """保存模型检查点"""
    os.makedirs(save_path, exist_ok=True)
    
    # 保存U-Net权重
    model_path = os.path.join(save_path, 'unet.pth')
    torch.save(model.unet.state_dict(), model_path)
    
    # 保存EMA权重（如果有）
    if hasattr(model, 'ema') and model.ema is not None:
        ema_path = os.path.join(save_path, 'ema.pth')
        torch.save(model.ema.shadow, ema_path)
    
    # 保存检查点信息
    checkpoint_info = {
        'model_type': 'StableLDM',
        'epoch': epoch,
        'unet_path': model_path,
        'has_ema': hasattr(model, 'ema') and model.ema is not None,
        'timestamp': time.time()
    }
    
    if extra_info:
        checkpoint_info.update(extra_info)
    
    info_path = os.path.join(save_path, 'checkpoint_info.json')
    with open(info_path, 'w') as f:
        json.dump(checkpoint_info, f, indent=2)
    
    return model_path

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """余弦学习率调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_stable_ldm():
    """轻量化LDM训练主函数"""
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设备配置
    if config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])
    print(f"🚀 使用设备: {device}")
    
    # 混合精度训练
    use_amp = config.get('mixed_precision', True) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"⚡ 混合精度训练: {'启用' if use_amp else '禁用'}")
    
    # 创建保存目录
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 构建数据加载器
    print("📁 加载数据集...")
    train_loader, val_loader, train_dataset_len, val_dataset_len = build_dataloader(
        config['dataset']['root_dir'],
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        shuffle_train=True,
        shuffle_val=False,
        val_split=config['dataset']['val_split']
    )
    
    print(f"✅ 数据集加载完成: 训练集 {train_dataset_len}, 验证集 {val_dataset_len}")
    
    # 初始化轻量化LDM模型
    print("🔥 初始化轻量化LDM模型...")
    
    # 处理VAE路径
    vae_path = config['vae']['model_path']
    if vae_path.startswith('../'):
        vae_path = os.path.join(os.path.dirname(__file__), vae_path)
    
    # 检查VAE文件
    if not os.path.exists(vae_path):
        print(f"❌ 错误: VAE权重文件不存在: {vae_path}")
        print(f"💡 请确保VAE模型已训练完成")
        return
    
    print(f"✓ 发现VAE权重文件: {vae_path}")
    config['vae']['model_path'] = vae_path
    
    model = create_stable_ldm(config).to(device)
    
    # 初始化评估器
    print("📊 初始化评估器...")
    diffusion_metrics = DiffusionMetrics(device=device)
    
    fid_evaluator = None
    if config['fid_evaluation']['enabled']:
        print("📈 初始化FID评估器...")
        fid_evaluator = FIDEvaluator(device=device)
        print("🔄 计算真实数据特征...")
        fid_evaluator.compute_real_features(val_loader, max_samples=1000)
    
    # 优化器配置
    learning_rate = config['training']['lr']
    optimizer = optim.AdamW(
        model.unet.parameters(),
        lr=learning_rate,
        weight_decay=config['training']['weight_decay'],
        betas=(config['training']['beta1'], config['training']['beta2']),
        eps=config['training']['eps']
    )
    
    # 学习率调度器
    scheduler = None
    if config['training']['use_scheduler']:
        total_steps = len(train_loader) * config['training']['epochs']
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['training']['warmup_steps'],
            num_training_steps=total_steps
        )
        print(f"📈 使用余弦学习率调度器")
    
    # 训练状态
    start_epoch = 0
    best_loss = float('inf')
    best_fid = float('inf')
    global_step = 0
    
    # 训练日志
    training_log = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'fid_scores': [],
        'learning_rates': []
    }
    
    print("=" * 80)
    print(f"🚀 开始轻量化LDM训练，共 {config['training']['epochs']} 个epoch")
    print("=" * 80)
    
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images, class_labels=labels)
                    loss = outputs['loss']
                
                # 混合精度反向传播
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                if config['training']['grad_clip_norm'] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.unet.parameters(), config['training']['grad_clip_norm'])
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # 标准训练
                outputs = model(images, class_labels=labels)
                loss = outputs['loss']
                
                loss.backward()
                
                if config['training']['grad_clip_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.unet.parameters(), config['training']['grad_clip_norm'])
                
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # 更新进度条
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}',
                'step': global_step
            })
            
            # 检查损失健康状况
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                print(f"❌ 检测到异常损失，跳过批次")
                continue
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # 验证
        avg_val_loss = None
        if val_loader:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(images, class_labels=labels)
                            loss = outputs['loss']
                    else:
                        outputs = model(images, class_labels=labels)
                        loss = outputs['loss']
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
        else:
            avg_val_loss = avg_epoch_loss
        
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']} - "
              f"Train Loss: {avg_epoch_loss:.4f} - "
              f"Val Loss: {avg_val_loss:.4f} - "
              f"Time: {epoch_time:.1f}s - "
              f"LR: {current_lr:.2e}")
        
        # FID评估
        fid_score = None
        eval_interval = config['fid_evaluation'].get('eval_interval', 5)
        if fid_evaluator and (epoch + 1) % eval_interval == 0:
            print(f"📊 计算FID分数...")
            try:
                fid_start_time = time.time()
                
                num_samples = config['fid_evaluation'].get('num_samples', 1000)
                batch_size = config['fid_evaluation'].get('batch_size', 16)
                num_inference_steps = config['inference'].get('num_inference_steps', 50)
                guidance_scale = config['inference'].get('guidance_scale', 7.5)
                
                fid_score = fid_evaluator.evaluate_model(
                    model, 
                    num_samples=num_samples,
                    batch_size=batch_size,
                    num_classes=config['unet']['num_classes'],
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                )
                
                fid_time = time.time() - fid_start_time
                print(f"✅ FID: {fid_score:.2f} | 时间: {fid_time:.1f}s")
                
                # 更新最佳FID
                if fid_score < best_fid:
                    best_fid = fid_score
                    best_fid_model_path = os.path.join(save_dir, 'best_fid_model')
                    save_model_checkpoint(
                        model, 
                        best_fid_model_path, 
                        epoch=epoch+1, 
                        extra_info={'fid_score': fid_score, 'best_fid': True}
                    )
                    print(f"🎉 新FID纪录: {best_fid:.2f}")
                
            except Exception as e:
                print(f"⚠️ FID计算失败: {e}")
        
        # 记录训练日志
        training_log['epochs'].append(epoch + 1)
        training_log['train_loss'].append(avg_epoch_loss)
        training_log['val_loss'].append(avg_val_loss)
        training_log['fid_scores'].append(fid_score)
        training_log['learning_rates'].append(current_lr)
        
        # 保存训练日志
        with open(os.path.join(log_dir, 'training_log.json'), 'w') as f:
            json.dump(training_log, f, indent=2)
        
        # 保存最佳损失模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, 'best_loss_model')
            save_model_checkpoint(
                model, 
                best_model_path, 
                epoch=epoch+1, 
                extra_info={'val_loss': best_loss, 'best_loss': True}
            )
            print(f"🎯 新最佳损失: {best_loss:.4f}")
        
        # 定期保存checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}')
            save_model_checkpoint(
                model, 
                checkpoint_path, 
                epoch=epoch+1, 
                extra_info={'train_loss': avg_epoch_loss, 'val_loss': avg_val_loss}
            )
            print(f"💾 保存checkpoint: epoch_{epoch+1}")
        
        # 生成样本图像
        if (epoch + 1) % config['training']['sample_interval'] == 0:
            print("🎨 生成样本图像...")
            save_sample_images(model, device, config, epoch + 1, sample_dir)
        
        print("-" * 60)
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'final_model')
    save_model_checkpoint(
        model, 
        final_model_path, 
        epoch=config['training']['epochs'], 
        extra_info={
            'final_train_loss': avg_epoch_loss,
            'final_val_loss': avg_val_loss,
            'best_fid': best_fid if best_fid != float('inf') else None,
            'training_completed': True
        }
    )
    
    # 创建训练报告
    final_report = {
        'model_type': 'StableLDM_Lightweight',
        'training_completed': True,
        'total_epochs': config['training']['epochs'],
        'best_train_loss': best_loss,
        'best_fid_score': best_fid if best_fid != float('inf') else None,
        'final_train_loss': avg_epoch_loss,
        'final_val_loss': avg_val_loss,
        'unet_parameters': sum(p.numel() for p in model.unet.parameters()),
        'device': str(device),
        'mixed_precision': use_amp,
        'has_ema': hasattr(model, 'ema') and model.ema is not None,
        'config': config
    }
    
    with open(os.path.join(save_dir, 'training_report.json'), 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print("=" * 80)
    print(f"🎉 轻量化LDM训练完成！")
    print(f"📁 最终模型: {final_model_path}")
    print(f"📉 最佳损失: {best_loss:.4f}")
    if best_fid != float('inf'):
        print(f"📊 最佳FID: {best_fid:.2f}")
    print(f"🔥 U-Net参数量: {sum(p.numel() for p in model.unet.parameters()):,}")
    print(f"📈 训练日志: {os.path.join(log_dir, 'training_log.json')}")
    print("=" * 80)

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 开始训练
    train_stable_ldm() 
