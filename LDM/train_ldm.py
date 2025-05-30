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

from ldm import LatentDiffusionModel
# 直接从VAE目录导入dataset功能
from dataset import build_dataloader
from fid_evaluation import FIDEvaluator
from metrics import DiffusionMetrics, denormalize_for_metrics

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    创建带预热的余弦学习率调度器
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def denormalize(x):
    """反归一化图像，从[-1,1]转换到[0,1]"""
    return (x * 0.5 + 0.5).clamp(0, 1)

def save_sample_images(model, device, config, epoch, save_dir):
    """生成并保存样本图像"""
    model.eval()
    with torch.no_grad():
        # 为每个类别生成样本
        num_classes = config['unet']['num_classes']
        num_samples_per_class = min(4, config['inference']['num_samples_per_class'])
        
        all_images = []
        for class_id in range(min(8, num_classes)):  # 最多显示8个类别
            class_labels = torch.tensor([class_id] * num_samples_per_class, device=device)
            
            generated_images = model.generate(
                batch_size=num_samples_per_class,
                class_labels=class_labels,
                num_inference_steps=config['inference']['num_inference_steps'],
                guidance_scale=config['inference']['guidance_scale'],
                eta=config['inference']['eta'],
            )
            
            # 反归一化
            generated_images = denormalize(generated_images)
            all_images.append(generated_images)
        
        # 拼接所有图像
        if all_images:
            all_images = torch.cat(all_images, dim=0)
            
            # 保存图像网格
            grid = vutils.make_grid(all_images, nrow=num_samples_per_class, normalize=False, padding=2)
            save_path = os.path.join(save_dir, f"samples_epoch_{epoch:04d}.png")
            vutils.save_image(grid, save_path)
            print(f"样本图像已保存: {save_path}")

def train_ldm():
    """LDM训练主函数"""
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设备配置
    if config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])
    print(f"使用设备: {device}")
    
    # 创建保存目录
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 构建数据加载器
    print("加载数据集...")
    train_loader, val_loader, train_dataset_len, val_dataset_len = build_dataloader(
        config['dataset']['root_dir'],
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        shuffle_train=True,
        shuffle_val=False,
        val_split=config['dataset']['val_split']
    )
    
    print(f"数据集加载完成: 训练集 {train_dataset_len}, 验证集 {val_dataset_len}")
    
    # 初始化LDM模型
    print("初始化LDM模型...")
    
    # 处理VAE路径
    vae_path = config['vae']['model_path']
    if vae_path.startswith('../'):
        vae_path = os.path.join(os.path.dirname(__file__), vae_path)
    
    # 检查VAE文件是否存在
    if not os.path.exists(vae_path):
        print(f"❌ 错误: VAE权重文件不存在: {vae_path}")
        print(f"💡 请先训练VAE模型，运行以下命令:")
        print(f"   cd VAE")
        print(f"   python train_adv_vqvae.py")
        print(f"训练完成后，VAE模型将保存为 vae_models/adv_vqvae_best_fid.pth")
        print(f"然后再运行LDM训练")
        return
    
    print(f"✓ 发现VAE权重文件: {vae_path}")
    
    model = LatentDiffusionModel(
        vae_config=config['vae'],
        unet_config=config['unet'],
        scheduler_config=config['diffusion'],
        vae_path=vae_path,
        freeze_vae=config['vae']['freeze'],
        use_cfg=config['training']['use_cfg'],
        cfg_dropout_prob=config['training']['cfg_dropout_prob'],
    ).to(device)
    
    # 初始化FID评估器
    print("初始化FID评估器...")
    fid_evaluator = FIDEvaluator(device=device)
    
    # 初始化扩散指标计算器
    print("初始化扩散指标计算器...")
    diffusion_metrics = DiffusionMetrics(device=device)
    
    # 计算真实数据特征（用于FID计算）
    # 使用验证集来计算真实数据特征，减少计算开销
    print("计算真实数据特征用于FID评估...")
    fid_evaluator.compute_real_features(val_loader, max_samples=1000)
    
    # 优化器
    optimizer = optim.AdamW(
        model.unet.parameters(),  # 只优化U-Net参数
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # 学习率调度器
    if config['training']['use_scheduler']:
        total_steps = len(train_loader) * config['training']['epochs']
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['training']['warmup_steps'],
            num_training_steps=total_steps,
        )
    else:
        scheduler = None
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if config.get('mixed_precision', False) and device.type == 'cuda' else None
    
    # 训练状态和日志
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
        'inception_scores': [],
        'noise_metrics': [],
        'learning_rates': []
    }
    
    # 训练循环
    print(f"开始训练，共 {config['training']['epochs']} 个epoch...")
    
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images, class_labels=labels)
                    loss = outputs['loss']
                
                # 反向传播
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                if config['training']['grad_clip_norm'] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.unet.parameters(), config['training']['grad_clip_norm'])
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images, class_labels=labels)
                loss = outputs['loss']
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                if config['training']['grad_clip_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.unet.parameters(), config['training']['grad_clip_norm'])
                
                optimizer.step()
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
            
            # 记录损失
            epoch_loss += loss.item()
            global_step += 1
            
            # 更新进度条
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # 定期打印日志
            if global_step % config['training']['log_interval'] == 0:
                print(f"Step {global_step}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
        
        # 计算平均损失
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} 完成:")
        print(f"  平均训练损失: {avg_epoch_loss:.4f}")
        print(f"  用时: {epoch_time:.1f}s")
        print(f"  学习率: {current_lr:.2e}")
        
        # 验证
        avg_val_loss = None
        noise_metrics_epoch = {}
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            
            # 用于噪声指标计算的累加器
            total_noise_metrics = {
                'noise_mse': 0.0,
                'noise_mae': 0.0,
                'noise_psnr': 0.0,
                'noise_cosine_similarity': 0.0,
                'noise_correlation': 0.0
            }
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = model(images, class_labels=labels)
                            loss = outputs['loss']
                    else:
                        outputs = model(images, class_labels=labels)
                        loss = outputs['loss']
                    
                    val_loss += loss.item()
                    
                    # 计算噪声预测指标
                    if 'predicted_noise' in outputs and 'target_noise' in outputs:
                        batch_noise_metrics = diffusion_metrics.calculate_all_metrics(
                            outputs['predicted_noise'], 
                            outputs['target_noise']
                        )
                        
                        # 累加指标
                        for key in total_noise_metrics.keys():
                            if key in batch_noise_metrics:
                                total_noise_metrics[key] += batch_noise_metrics[key]
                    
                    val_steps += 1
            
            avg_val_loss = val_loss / val_steps
            
            # 计算平均噪声指标
            if val_steps > 0:
                for key in total_noise_metrics.keys():
                    noise_metrics_epoch[key] = total_noise_metrics[key] / val_steps
            
            print(f"  验证损失: {avg_val_loss:.4f}")
            if noise_metrics_epoch:
                print(f"  噪声MSE: {noise_metrics_epoch.get('noise_mse', 0):.6f}")
                print(f"  噪声PSNR: {noise_metrics_epoch.get('noise_psnr', 0):.2f}")
                print(f"  噪声余弦相似度: {noise_metrics_epoch.get('noise_cosine_similarity', 0):.4f}")
        else:
            avg_val_loss = avg_epoch_loss
        
        # FID评估 - 每10个epoch计算一次
        fid_score = None
        inception_score = None
        if (epoch + 1) % 10 == 0:
            print(f"  计算FID分数和IS分数...")
            try:
                fid_start_time = time.time()
                fid_score = fid_evaluator.evaluate_model(
                    model, 
                    num_samples=min(500, val_dataset_len),  # 限制样本数以节省时间
                    batch_size=4,  # 较小的批量大小避免内存问题
                    num_classes=config['unet']['num_classes']
                )
                fid_time = time.time() - fid_start_time
                print(f"  FID分数: {fid_score:.2f} (用时: {fid_time:.1f}s)")
                
                # 计算IS分数
                print(f"  计算IS分数...")
                is_start_time = time.time()
                
                # 生成样本用于IS计算
                model.eval()
                with torch.no_grad():
                    generated_samples = []
                    num_is_samples = min(200, val_dataset_len)  # IS分数样本数
                    
                    for i in range(0, num_is_samples, 8):  # 每次生成8个样本
                        batch_size = min(8, num_is_samples - i)
                        # 随机选择类别
                        class_labels = torch.randint(0, config['unet']['num_classes'], 
                                                    (batch_size,), device=device)
                        
                        generated_batch = model.generate(
                            batch_size=batch_size,
                            class_labels=class_labels,
                            num_inference_steps=20,  # 较少步数以节省时间
                            guidance_scale=config['inference']['guidance_scale'],
                        )
                        
                        # 反归一化到[0,1]
                        generated_batch = denormalize_for_metrics(generated_batch)
                        generated_samples.append(generated_batch)
                    
                    if generated_samples:
                        all_generated = torch.cat(generated_samples, dim=0)
                        is_mean, is_std = diffusion_metrics.inception_calculator.calculate_inception_score(
                            all_generated, splits=5
                        )
                        inception_score = {'mean': is_mean, 'std': is_std}
                        
                        is_time = time.time() - is_start_time
                        print(f"  IS分数: {is_mean:.2f}±{is_std:.2f} (用时: {is_time:.1f}s)")
                
                # 更新最佳FID
                if fid_score < best_fid:
                    best_fid = fid_score
                    best_fid_model_path = os.path.join(save_dir, 'best_fid_model')
                    model.save_pretrained(best_fid_model_path)
                    print(f"  🎉 FID新纪录！保存最佳FID模型: {best_fid_model_path}")
                
            except Exception as e:
                print(f"  ⚠️ 指标计算失败: {e}")
                fid_score = None
                inception_score = None
        
        # 记录训练日志
        training_log['epochs'].append(epoch + 1)
        training_log['train_loss'].append(avg_epoch_loss)
        training_log['val_loss'].append(avg_val_loss)
        training_log['fid_scores'].append(fid_score)
        training_log['inception_scores'].append(inception_score)
        training_log['noise_metrics'].append(noise_metrics_epoch)
        training_log['learning_rates'].append(current_lr)
        
        # 保存训练日志
        with open(os.path.join(log_dir, 'training_log.json'), 'w') as f:
            json.dump(training_log, f, indent=2)
        
        # 保存最佳损失模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, 'best_loss_model')
            model.save_pretrained(best_model_path)
            print(f"  保存最佳损失模型: {best_model_path}")
        
        # 定期保存checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}')
            model.save_pretrained(checkpoint_path)
            print(f"  保存checkpoint: {checkpoint_path}")
        
        # 生成样本图像
        if (epoch + 1) % config['training']['sample_interval'] == 0:
            print("  生成样本图像...")
            save_sample_images(model, device, config, epoch + 1, sample_dir)
        
        print("-" * 80)
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'final_model')
    model.save_pretrained(final_model_path)
    
    # 保存最终训练报告
    final_report = {
        'training_completed': True,
        'total_epochs': config['training']['epochs'],
        'best_train_loss': best_loss,
        'best_fid_score': best_fid if best_fid != float('inf') else None,
        'final_train_loss': avg_epoch_loss,
        'final_val_loss': avg_val_loss,
        'total_parameters': sum(p.numel() for p in model.unet.parameters()),
        'device': str(device),
        'config': config
    }
    
    with open(os.path.join(save_dir, 'training_report.json'), 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print("=" * 80)
    print(f"🎉 训练完成！")
    print(f"  最终模型: {final_model_path}")
    print(f"  最佳损失: {best_loss:.4f}")
    if best_fid != float('inf'):
        print(f"  最佳FID: {best_fid:.2f}")
    print(f"  训练日志: {os.path.join(log_dir, 'training_log.json')}")
    print("=" * 80)

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 开始训练
    train_ldm() 
