#!/usr/bin/env python3
"""
FID优化专用训练脚本
目标：将FID降到20以内
特点：大模型、高质量训练、精细化参数调优
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

# 添加路径
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
    
    # 每个类别生成2个样本
    num_classes = config['unet']['num_classes']
    samples_per_class = 2
    
    all_samples = []
    
    with torch.no_grad():
        for class_id in range(min(8, num_classes)):  # 只显示前8个类别
            class_labels = torch.tensor([class_id] * samples_per_class, device=device)
            
            # 高质量采样
            samples = model.sample(
                batch_size=samples_per_class,
                class_labels=class_labels,
                num_inference_steps=config['inference']['num_inference_steps'],
                guidance_scale=config['inference']['guidance_scale'],
                verbose=False
            )
            
            all_samples.append(samples)
    
    # 合并样本
    all_samples = torch.cat(all_samples, dim=0)
    all_samples = denormalize(all_samples)
    
    # 保存样本网格
    grid = vutils.make_grid(all_samples, nrow=samples_per_class, padding=2, normalize=False)
    
    save_path = os.path.join(save_dir, f"samples_epoch_{epoch:03d}.png")
    vutils.save_image(grid, save_path)
    
    print(f"💾 样本图像已保存: {save_path}")

def save_model_checkpoint(model, save_path, epoch=None, extra_info=None):
    """保存模型检查点"""
    os.makedirs(save_path, exist_ok=True)
    
    # 保存模型权重
    checkpoint = {
        'unet_state_dict': model.unet.state_dict(),
        'epoch': epoch,
        'extra_info': extra_info or {}
    }
    
    # 保存EMA权重（如果存在）
    if model.ema is not None:
        checkpoint['ema_shadow'] = model.ema.shadow
    
    torch.save(checkpoint, os.path.join(save_path, 'model.pth'))
    
    # 保存配置
    if hasattr(model, 'config'):
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(model.config, f, default_flow_style=False)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """余弦学习率调度器"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_fid_optimized():
    """FID优化专用训练主函数"""
    
    # 加载配置
    config_path = 'config_fid_optimized.yaml'
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("🎯 FID优化训练启动 - 目标FID < 20")
    
    # 设备配置
    if config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])
    print(f"🚀 使用设备: {device}")
    
    # 混合精度训练
    use_amp = config.get('mixed_precision', False) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"⚡ 混合精度训练: {'启用' if use_amp else '禁用 (追求质量)'}")
    
    # 创建保存目录
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    log_dir = config['logging']['log_dir']
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
    
    # 初始化强化LDM模型
    print("🔥 初始化强化LDM模型...")
    
    # 处理VAE路径
    vae_path = config['vae']['model_path']
    if vae_path.startswith('../'):
        vae_path = os.path.join(os.path.dirname(__file__), vae_path)
    
    # 检查VAE文件
    if not os.path.exists(vae_path):
        print(f"❌ 错误: VAE权重文件不存在: {vae_path}")
        return
    
    print(f"✓ 发现VAE权重文件: {vae_path}")
    config['vae']['model_path'] = vae_path
    
    model = create_stable_ldm(config).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.unet.parameters())
    print(f"🧠 U-Net参数量: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # 初始化评估器
    print("📊 初始化评估器...")
    diffusion_metrics = DiffusionMetrics(device=device)
    
    fid_evaluator = None
    if config['fid_evaluation']['enabled']:
        print("📈 初始化高精度FID评估器...")
        fid_evaluator = FIDEvaluator(device=device)
        print("🔄 计算真实数据特征...")
        max_real_samples = config['fid_evaluation']['max_real_samples']
        fid_evaluator.compute_real_features(val_loader, max_samples=max_real_samples)
    
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
        total_steps = len(train_loader) * config['training']['epochs'] // config['training']['gradient_accumulation_steps']
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['training']['warmup_steps'],
            num_training_steps=total_steps
        )
        print(f"📈 使用余弦学习率调度器 (总步数: {total_steps})")
    
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
        'learning_rates': [],
        'best_fid_epoch': None
    }
    
    print("=" * 80)
    print(f"🎯 开始FID优化训练，目标FID < 20，共 {config['training']['epochs']} 个epoch")
    print(f"📊 当前配置: 大模型 ({total_params/1e6:.1f}M参数) + 高质量采样")
    print("=" * 80)
    
    accumulation_steps = config['training']['gradient_accumulation_steps']
    
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            # 混合精度前向传播
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images, class_labels=labels)
                    loss = outputs['loss'] / accumulation_steps  # 梯度累积
                
                # 混合精度反向传播
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    # 梯度裁剪
                    if config['training']['grad_clip_norm'] > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.unet.parameters(), config['training']['grad_clip_norm'])
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    if scheduler is not None:
                        scheduler.step()
                    
                    global_step += 1
                    
            else:
                # 标准训练
                outputs = model(images, class_labels=labels)
                loss = outputs['loss'] / accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    if config['training']['grad_clip_norm'] > 0:
                        torch.nn.utils.clip_grad_norm_(model.unet.parameters(), config['training']['grad_clip_norm'])
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if scheduler is not None:
                        scheduler.step()
                    
                    global_step += 1
            
            epoch_loss += loss.item() * accumulation_steps
            
            # 更新进度条
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
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
        
        # 显示验证损失趋势
        if best_loss != float('inf'):
            loss_diff = avg_val_loss - best_loss
            if loss_diff <= 0:
                print(f"📉 验证损失: {avg_val_loss:.4f} (新纪录! ⬇{abs(loss_diff):.4f})")
            else:
                print(f"📊 验证损失: {avg_val_loss:.4f} (距最佳: +{loss_diff:.4f})")
        
        # FID评估
        fid_score = None
        eval_interval = config['fid_evaluation'].get('eval_interval', 25)
        if fid_evaluator and (epoch + 1) % eval_interval == 0:
            print(f"📊 计算高精度FID分数...")
            try:
                fid_start_time = time.time()
                
                num_samples = config['fid_evaluation'].get('num_samples', 5000)
                batch_size = config['fid_evaluation'].get('batch_size', 20)
                num_inference_steps = config['inference'].get('num_inference_steps', 250)
                guidance_scale = config['inference'].get('guidance_scale', 12.0)
                
                fid_score = fid_evaluator.evaluate_model(
                    model, 
                    num_samples=num_samples,
                    batch_size=batch_size,
                    num_classes=config['unet']['num_classes'],
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                )
                
                fid_time = time.time() - fid_start_time
                
                # 显示FID结果和趋势
                if best_fid != float('inf'):
                    fid_diff = fid_score - best_fid
                    if fid_score < best_fid:
                        print(f"✅ FID: {fid_score:.2f} (新纪录! ⬇{abs(fid_diff):.2f}) | 时间: {fid_time:.1f}s")
                    else:
                        print(f"📊 FID: {fid_score:.2f} (距最佳: +{fid_diff:.2f}) | 时间: {fid_time:.1f}s")
                else:
                    print(f"✅ FID: {fid_score:.2f} (首次评估) | 时间: {fid_time:.1f}s")
                
                # 更新最佳FID
                if fid_score < best_fid:
                    best_fid = fid_score
                    training_log['best_fid_epoch'] = epoch + 1
                    best_fid_model_path = os.path.join(save_dir, 'best_fid_model')
                    save_model_checkpoint(
                        model, 
                        best_fid_model_path, 
                        epoch=epoch+1, 
                        extra_info={'fid_score': fid_score, 'best_fid': True}
                    )
                    print(f"🎉 新FID最佳模型已保存!")
                    
                    # 如果达到目标FID
                    if fid_score < 20:
                        print(f"🎯 达到目标FID！FID = {fid_score:.2f} < 20")
                
            except Exception as e:
                print(f"⚠️ FID计算失败: {e}")
        
        # 记录训练日志
        training_log['epochs'].append(epoch + 1)
        training_log['train_loss'].append(avg_epoch_loss)
        training_log['val_loss'].append(avg_val_loss)
        training_log['fid_scores'].append(fid_score)
        training_log['learning_rates'].append(current_lr)
        
        # 保存训练日志
        with open(os.path.join(log_dir, 'fid_training_log.json'), 'w') as f:
            json.dump(training_log, f, indent=2)
        
        # 保存最佳损失模型（只在创建新纪录时输出）
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, 'best_loss_model')
            save_model_checkpoint(
                model, 
                best_model_path, 
                epoch=epoch+1, 
                extra_info={'val_loss': best_loss, 'best_loss': True}
            )
        
        # 定期保存checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}')
            save_model_checkpoint(
                model, 
                checkpoint_path, 
                epoch=epoch+1, 
                extra_info={'train_loss': avg_epoch_loss, 'val_loss': avg_val_loss}
            )
            print(f"💾 定期checkpoint已保存: epoch_{epoch+1}")
        
        # 生成样本图像
        if (epoch + 1) % config['training']['sample_interval'] == 0:
            print("🎨 生成高质量样本图像...")
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
    
    # 创建最终报告
    final_report = {
        'model_type': 'StableLDM_FID_Optimized',
        'training_completed': True,
        'total_epochs': config['training']['epochs'],
        'best_train_loss': best_loss,
        'best_fid_score': best_fid if best_fid != float('inf') else None,
        'best_fid_epoch': training_log['best_fid_epoch'],
        'final_train_loss': avg_epoch_loss,
        'final_val_loss': avg_val_loss,
        'unet_parameters': sum(p.numel() for p in model.unet.parameters()),
        'device': str(device),
        'mixed_precision': use_amp,
        'target_achieved': best_fid < 20 if best_fid != float('inf') else False,
        'config': config
    }
    
    with open(os.path.join(save_dir, 'final_report.json'), 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print("=" * 80)
    print(f"🎯 FID优化训练完成！")
    print(f"📁 最终模型: {final_model_path}")
    print(f"📉 最佳损失: {best_loss:.4f}")
    if best_fid != float('inf'):
        print(f"📊 最佳FID: {best_fid:.2f} (Epoch {training_log['best_fid_epoch']})")
        if best_fid < 20:
            print(f"🎉 成功达到目标FID < 20！")
        else:
            print(f"⚠️ 未达到目标FID，还需继续优化")
    print(f"🔥 U-Net参数量: {sum(p.numel() for p in model.unet.parameters()):,}")
    print("=" * 80)

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 开始FID优化训练
    train_fid_optimized() 
