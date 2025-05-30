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
# 从VAE目录导入dataset功能
from dataset import build_dataloader
from fid_evaluation import FIDEvaluator
from metrics import DiffusionMetrics, denormalize_for_metrics

def check_tensor_health(tensor, name="tensor"):
    """检查张量的数值健康状况"""
    if torch.any(torch.isnan(tensor)):
        print(f"❌ {name} 包含 NaN 值")
        return False
    if torch.any(torch.isinf(tensor)):
        print(f"❌ {name} 包含 Inf 值")
        return False
    if tensor.abs().max() > 1000:
        print(f"⚠️ {name} 包含极大值: max={tensor.abs().max().item():.2f}")
        return False
    return True

def safe_mse_loss(pred, target, name="loss"):
    """安全的MSE损失计算，防止NaN/Inf"""
    # 检查输入
    if not check_tensor_health(pred, f"{name}_pred"):
        pred = torch.clamp(pred, -10, 10)
    if not check_tensor_health(target, f"{name}_target"):
        target = torch.clamp(target, -10, 10)
    
    # 计算损失
    loss = torch.nn.functional.mse_loss(pred, target)
    
    # 检查损失值
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"⚠️ {name} 计算出现异常，使用安全值")
        loss = torch.tensor(1.0, device=pred.device, requires_grad=True)
    
    # 限制损失范围
    loss = torch.clamp(loss, 0.0, 100.0)
    return loss

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

def get_cosine_schedule_with_restarts(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 3,  # 🔧 重启次数
    restart_decay: float = 0.8,  # 🔧 每次重启的学习率衰减
    last_epoch: int = -1,
):
    """
    🔧 创建带重启的余弦学习率调度器 - 更强的学习能力
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        # 计算当前在第几个周期
        cycle_length = 1.0 / num_cycles
        current_cycle = int(progress / cycle_length)
        cycle_progress = (progress % cycle_length) / cycle_length
        
        # 每次重启时降低基础学习率
        base_lr = restart_decay ** current_cycle
        
        # 余弦衰减
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
        
        return max(0.0, base_lr * cosine_factor)

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

def save_model_checkpoint(model, save_path, epoch=None, extra_info=None):
    """保存模型检查点的统一函数"""
    os.makedirs(save_path, exist_ok=True)
    
    # 保存模型权重
    model_path = os.path.join(save_path, 'model.pth')
    torch.save(model.state_dict(), model_path)
    
    # 保存模型配置和额外信息
    checkpoint_info = {
        'model_type': 'LatentDiffusionModel',
        'epoch': epoch,
        'model_path': model_path,
        'timestamp': time.time()
    }
    
    if extra_info:
        checkpoint_info.update(extra_info)
    
    info_path = os.path.join(save_path, 'checkpoint_info.json')
    with open(info_path, 'w') as f:
        json.dump(checkpoint_info, f, indent=2)
    
    return model_path

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
    
    # 🔧 更新VAE配置中的模型路径
    config['vae']['model_path'] = vae_path
    
    model = LatentDiffusionModel(config).to(device)
    
    # 检查模型权重健康状况
    print("🔍 检查模型权重...")
    total_params = 0
    healthy_params = 0
    for name, param in model.unet.named_parameters():
        total_params += param.numel()
        if torch.any(torch.isnan(param)) or torch.any(torch.isinf(param)):
            print(f"⚠️ 参数 {name} 包含异常值")
            # 重新初始化异常参数
            nn.init.xavier_uniform_(param)
        else:
            healthy_params += param.numel()
    
    print(f"✅ 模型权重检查完成: {healthy_params}/{total_params} 参数正常")
    
    # 初始化扩散指标计算器
    print("初始化扩散指标计算器...")
    diffusion_metrics = DiffusionMetrics(device=device)
    
    # 只有启用FID评估时才初始化
    fid_evaluator = None
    if config['fid_evaluation']['enabled']:
        print("初始化FID评估器...")
        fid_evaluator = FIDEvaluator(device=device)
        # 计算真实数据特征
        print("计算真实数据特征用于FID评估...")
        fid_evaluator.compute_real_features(val_loader, max_samples=1000)
    
    # 确保数值类型正确
    learning_rate = float(config['training']['lr'])
    weight_decay = float(config['training']['weight_decay'])
    
    print(f"📊 训练参数: LR={learning_rate:.6f}, 梯度裁剪={config['training']['grad_clip_norm']}")
    
    # 优化器
    optimizer = optim.AdamW(
        model.unet.parameters(),  # 只优化U-Net参数
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8,  # 增加数值稳定性
    )
    
    # 学习率调度器
    if config['training']['use_scheduler']:
        total_steps = len(train_loader) * config['training']['epochs']
        
        # 🔧 根据配置选择调度器类型
        scheduler_type = config['training'].get('scheduler_type', 'cosine')
        
        if scheduler_type == "cosine_with_restarts":
            print(f"📈 使用带重启的余弦学习率调度器")
            scheduler = get_cosine_schedule_with_restarts(
                optimizer,
                num_warmup_steps=config['training']['warmup_steps'],
                num_training_steps=total_steps,
                num_cycles=3,  # 3次重启周期
                restart_decay=0.8,  # 每次重启衰减到80%
            )
        else:
            print(f"📈 使用标准余弦学习率调度器")
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config['training']['warmup_steps'],
                num_training_steps=total_steps,
            )
    else:
        scheduler = None
    
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
        valid_batches = 0  # 跟踪有效批次数
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            try:
                images = images.to(device)
                labels = labels.to(device)
                
                # 检查输入数据健康状况
                if not check_tensor_health(images, "input_images"):
                    print(f"跳过批次 {batch_idx}：输入图像异常")
                    continue
                
                # 前向传播
                outputs = model(images, class_labels=labels)
                loss = outputs['total_loss']  # 使用新的损失键名
                
                # 检查损失
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"❌ 批次 {batch_idx} 损失异常: {loss.item()}")
                    # 尝试使用噪声损失作为备选
                    if 'noise_loss' in outputs:
                        loss = outputs['noise_loss']
                        print(f"  使用噪声损失: {loss.item():.4f}")
                    else:
                        print(f"跳过批次 {batch_idx}")
                        continue
                
                # 限制损失值
                if loss.item() > 100:
                    print(f"⚠️ 损失过大 ({loss.item():.2f})，限制为100")
                    loss = torch.clamp(loss, max=100.0)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 检查梯度健康
                total_norm = 0
                gradient_healthy = True
                for p in model.unet.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        if torch.isnan(param_norm) or torch.isinf(param_norm):
                            print(f"⚠️ 检测到异常梯度，跳过更新")
                            gradient_healthy = False
                            break
                        total_norm += param_norm.item() ** 2
                
                if gradient_healthy:
                    total_norm = total_norm ** (1. / 2)
                    
                    # 梯度裁剪
                    if config['training']['grad_clip_norm'] > 0:
                        torch.nn.utils.clip_grad_norm_(model.unet.parameters(), config['training']['grad_clip_norm'])
                    
                    optimizer.step()
                    
                    # 更新学习率
                    if scheduler is not None:
                        scheduler.step()
                    
                    # 记录损失
                    epoch_loss += loss.item()
                    valid_batches += 1
                    global_step += 1
                    
                    # 更新进度条
                    current_lr = optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{current_lr:.2e}',
                        'grad_norm': f'{total_norm:.2f}'
                    })
            
            except Exception as e:
                print(f"❌ 批次 {batch_idx} 出错: {e}")
                continue
        
        # 计算平均损失
        if valid_batches > 0:
            avg_epoch_loss = epoch_loss / valid_batches
        else:
            print("❌ 没有有效的训练批次！")
            avg_epoch_loss = float('inf')
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']} - Loss: {avg_epoch_loss:.4f} - Valid Batches: {valid_batches}/{len(train_loader)} - Time: {epoch_time:.1f}s - LR: {current_lr:.2e}")
        
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
                    
                    outputs = model(images, class_labels=labels)
                    loss = outputs['total_loss']  # 使用新的损失键名
                    
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
            
            # 整合验证结果到一行
            if noise_metrics_epoch:
                noise_mse = noise_metrics_epoch.get('noise_mse', 0)
                noise_psnr = noise_metrics_epoch.get('noise_psnr', 0)
                print(f"Val Loss: {avg_val_loss:.4f} | Noise MSE: {noise_mse:.6f} | PSNR: {noise_psnr:.2f}")
            else:
                print(f"Val Loss: {avg_val_loss:.4f}")
        else:
            avg_val_loss = avg_epoch_loss
        
        # FID评估 - 每10个epoch计算一次
        fid_score = None
        inception_score = None
        if fid_evaluator and (epoch + 1) % 10 == 0:
            print(f"计算评估指标...")
            try:
                fid_start_time = time.time()
                fid_score = fid_evaluator.evaluate_model(
                    model, 
                    num_samples=min(500, val_dataset_len),  # 限制样本数以节省时间
                    batch_size=4,  # 较小的批量大小避免内存问题
                    num_classes=config['unet']['num_classes']
                )
                fid_time = time.time() - fid_start_time
                
                # 计算IS分数（简化输出）
                is_start_time = time.time()
                
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
                        
                        generated_batch = model.sample(
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
                        total_eval_time = fid_time + is_time
                        print(f"FID: {fid_score:.2f} | IS: {is_mean:.2f}±{is_std:.2f} | Time: {total_eval_time:.1f}s")
                
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
                    print(f"🎉 新纪录！FID: {best_fid:.2f}")
                
            except Exception as e:
                print(f"⚠️ 指标计算失败: {e}")
        
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
            save_model_checkpoint(
                model, 
                best_model_path, 
                epoch=epoch+1, 
                extra_info={'val_loss': best_loss, 'best_loss': True}
            )
            print(f"🎉 新最佳损失: {best_loss:.4f}")
        
        # 定期保存checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}')
            save_model_checkpoint(
                model, 
                checkpoint_path, 
                epoch=epoch+1, 
                extra_info={'train_loss': avg_epoch_loss, 'val_loss': avg_val_loss}
            )
            print(f"💾 保存checkpoint: {checkpoint_path}")
        
        # 生成样本图像
        if (epoch + 1) % config['training']['sample_interval'] == 0:
            print("  生成样本图像...")
            save_sample_images(model, device, config, epoch + 1, sample_dir)
        
        print("-" * 80)
        
        # 如果损失连续异常，提前停止
        if math.isinf(avg_epoch_loss):
            print("❌ 训练损失异常，建议检查配置后重新开始")
            break
    
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
