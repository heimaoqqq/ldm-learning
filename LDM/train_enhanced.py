"""
🚀 增强版LDM训练脚本
使用包含真正Transformer的简化增强版U-Net
集成完整的指标评估系统
"""

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import time
from tqdm import tqdm
import numpy as np
from datetime import datetime

# 添加VAE模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VAE'))

from stable_ldm import create_stable_ldm
from fid_evaluation import FIDEvaluator
from metrics import DiffusionMetrics, denormalize_for_metrics
from dataset_adapter import build_dataloader

class EnhancedTrainer:
    """增强版LDM训练器"""
    
    def __init__(self, config_path: str, compile_model: bool = False):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("🔧 初始化增强版LDM训练器...")
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  使用设备: {self.device}")
        
        # 创建模型
        self.model = create_stable_ldm(self.config).to(self.device)
        
        # 🚀 模型编译优化
        if compile_model and hasattr(torch, 'compile'):
            try:
                print("🚀 编译模型以提高性能...")
                self.model = torch.compile(self.model)
                print("✅ 模型编译成功")
            except Exception as e:
                print(f"⚠️ 模型编译失败，继续使用未编译版本: {e}")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 模型参数统计:")
        print(f"   总参数: {total_params/1e6:.1f}M")
        print(f"   可训练参数: {trainable_params/1e6:.1f}M")
        
        # 创建指标计算器
        self.metrics_calculator = DiffusionMetrics(self.device)
        
        # 创建FID评估器
        self.fid_evaluator = FIDEvaluator(self.device)
        
        # 设置优化器
        self.optimizer = optim.AdamW(
            self.model.unet.parameters(),
            lr=self.config['training']['lr'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2']),
            eps=self.config['training']['eps'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # 梯度累积设置
        self.gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        
        # 早停设置
        self.best_fid = float('inf')
        self.best_loss = float('inf')
        self.best_is_score = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = self.config['training'].get('early_stopping_patience', 50)
        
        # 🚀 训练改进监控
        self.previous_metrics = {}
        self.improvement_history = []
        
        # 🚀 质量监控配置
        self.quality_config = self.config.get('quality_monitoring', {})
        self.fid_improvement_threshold = self.quality_config.get('improvement_threshold', 0.02)
        self.loss_improvement_threshold = self.quality_config.get('loss_improvement_threshold', 0.001)
        
        # 日志存储
        self.training_logs = []
        
        # 创建保存目录
        self.save_dir = self.config['training']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        print("✅ 训练器初始化完成")
        print(f"🎯 优化配置: batch_size=8, 梯度累积=3, 有效batch=24")
        print(f"📈 学习率调整: 0.00008 (适配batch size变化)")
        print(f"📂 数据集路径: {self.config['dataset']['root_dir']}")
        print(f"🚫 数据增强: 已禁用 (保护时频图物理意义)")
        
    def check_gradient_health(self) -> dict:
        """检查梯度健康状况"""
        total_norm = 0.0
        param_count = 0
        
        for param in self.model.unet.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        # 检查梯度爆炸
        gradient_threshold = self.quality_config.get('gradient_norm_threshold', 10.0)
        gradient_exploding = total_norm > gradient_threshold
        
        return {
            'gradient_norm': total_norm,
            'gradient_exploding': gradient_exploding,
            'param_count': param_count
        }
    
    def detect_improvements(self, current_metrics: dict, epoch: int) -> dict:
        """检测训练改进"""
        improvements = {
            'fid_improved': False,
            'loss_improved': False,
            'is_improved': False,
            'overall_improved': False
        }
        
        if not self.previous_metrics:
            self.previous_metrics = current_metrics.copy()
            return improvements
        
        # FID改进检测 (越小越好)
        if 'fid_score' in current_metrics and 'fid_score' in self.previous_metrics:
            fid_improvement = self.previous_metrics['fid_score'] - current_metrics['fid_score']
            if fid_improvement > self.fid_improvement_threshold:
                improvements['fid_improved'] = True
                print(f"🎉 FID显著改进! {self.previous_metrics['fid_score']:.3f} → {current_metrics['fid_score']:.3f} (改进: {fid_improvement:.3f})")
        
        # 损失改进检测 (越小越好)
        if 'loss' in current_metrics and 'loss' in self.previous_metrics:
            loss_improvement = self.previous_metrics['loss'] - current_metrics['loss']
            if loss_improvement > self.loss_improvement_threshold:
                improvements['loss_improved'] = True
                print(f"📉 训练损失改进! {self.previous_metrics['loss']:.4f} → {current_metrics['loss']:.4f} (改进: {loss_improvement:.4f})")
        
        # IS分数改进检测 (越大越好)
        if 'inception_score_mean' in current_metrics and 'inception_score_mean' in self.previous_metrics:
            is_improvement = current_metrics['inception_score_mean'] - self.previous_metrics['inception_score_mean']
            if is_improvement > 0.1:  # IS改进阈值
                improvements['is_improved'] = True
                print(f"🌟 IS分数改进! {self.previous_metrics['inception_score_mean']:.3f} → {current_metrics['inception_score_mean']:.3f} (改进: {is_improvement:.3f})")
        
        # 综合改进判断
        if any([improvements['fid_improved'], improvements['loss_improved'], improvements['is_improved']]):
            improvements['overall_improved'] = True
            print(f"🚀 Epoch {epoch}: 训练质量有显著提升!")
        
        # 记录改进历史
        self.improvement_history.append({
            'epoch': epoch,
            'improvements': improvements.copy(),
            'metrics': current_metrics.copy()
        })
        
        # 更新前一轮指标
        self.previous_metrics = current_metrics.copy()
        
        return improvements
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = []
        epoch_metrics = {
            'noise_mse': [],
            'noise_mae': [],
            'noise_psnr': [],
            'noise_cosine_similarity': [],
            'noise_correlation': []
        }
        
        # 🚀 梯度健康监控
        gradient_health_issues = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 前向传播 - 确保数据在正确的设备上
            images = batch['image'].to(self.device)
            labels = batch.get('label', None)
            if labels is not None:
                labels = labels.to(self.device)
            
            loss_dict = self.model(images, labels)
            
            loss = loss_dict['loss'] / self.gradient_accumulation_steps
            
            # 🚀 检查损失突增
            loss_spike_threshold = self.quality_config.get('loss_spike_threshold', 2.0)
            if epoch_losses and loss.item() > np.mean(epoch_losses[-10:]) * loss_spike_threshold:
                print(f"⚠️ 检测到损失突增: {loss.item():.4f} (平均: {np.mean(epoch_losses[-10:]):.4f})")
            
            # 计算噪声预测指标
            noise_metrics = self.metrics_calculator.calculate_all_metrics(
                loss_dict['predicted_noise'],
                loss_dict['target_noise']
            )
            
            # 更新指标
            for key in epoch_metrics:
                if key in noise_metrics:
                    epoch_metrics[key].append(noise_metrics[key])
            
            # 反向传播
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 🚀 梯度健康检查
                grad_health = self.check_gradient_health()
                if grad_health['gradient_exploding']:
                    gradient_health_issues += 1
                    print(f"⚠️ 梯度爆炸警告: 梯度范数 {grad_health['gradient_norm']:.2f}")
                
                # 梯度裁剪
                if 'grad_clip_norm' in self.config['training']:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.unet.parameters(), 
                        self.config['training']['grad_clip_norm']
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # 更新EMA
                if hasattr(self.model, 'ema') and self.model.ema is not None:
                    self.model.ema.update()
            
            epoch_losses.append(loss.item() * self.gradient_accumulation_steps)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{epoch_losses[-1]:.4f}',
                'PSNR': f'{noise_metrics.get("noise_psnr", 0):.4f}dB',
                'Cosine': f'{noise_metrics.get("noise_cosine_similarity", 0):.4f}',
                'GradNorm': f'{grad_health.get("gradient_norm", 0):.2f}' if (batch_idx + 1) % self.gradient_accumulation_steps == 0 else 'N/A'
            })
        
        # 🚀 输出梯度健康统计
        if gradient_health_issues > 0:
            print(f"⚠️ Epoch {epoch}: 检测到 {gradient_health_issues} 次梯度异常")
        
        # 计算epoch平均指标
        avg_metrics = {
            'epoch': epoch,
            'loss': np.mean(epoch_losses),
            'noise_mse': np.mean(epoch_metrics['noise_mse']),
            'noise_mae': np.mean(epoch_metrics['noise_mae']),
            'noise_psnr': np.mean(epoch_metrics['noise_psnr']),
            'noise_cosine_similarity': np.mean(epoch_metrics['noise_cosine_similarity']),
            'noise_correlation': np.mean(epoch_metrics['noise_correlation']),
            'gradient_health_issues': gradient_health_issues
        }
        
        return avg_metrics
    
    def evaluate_fid(self, epoch: int) -> float:
        """评估FID分数"""
        print(f"\n🔍 Epoch {epoch} FID评估")
        
        try:
            # 🚀 获取优化后的评估参数
            fid_config = self.config['fid_evaluation']
            
            fid_score = self.fid_evaluator.evaluate_model(
                model=self.model,
                num_samples=fid_config['num_samples'],
                batch_size=fid_config['batch_size'],  # 🚀 使用优化后的batch size (64)
                num_classes=self.config['unet']['num_classes'],
                num_inference_steps=fid_config.get('num_inference_steps', 75),  # 🚀 可配置推理步数
                guidance_scale=self.config['inference']['guidance_scale'],
                eta=self.config['inference']['eta'],
                eval_classes=fid_config.get('eval_classes', 3),  # 🚀 只评估3个类别
                parallel_generation=fid_config.get('parallel_generation', False)  # 🚀 并行生成优化
            )
            
            print(f"  ✅ FID评估完成: {fid_score:.2f}")
            return fid_score
            
        except Exception as e:
            print(f"  ❌ FID评估失败: {e}")
            return float('inf')
    
    def generate_samples(self, epoch: int, num_samples: int = 16):
        """生成样本并计算IS分数"""
        try:
            self.model.eval()
            
            with torch.no_grad():
                # 生成样本
                class_labels = torch.randint(0, self.config['unet']['num_classes'], 
                                           (num_samples,), device=self.device)
                
                generated_images = self.model.sample(
                    batch_size=num_samples,
                    class_labels=class_labels,
                    num_inference_steps=self.config['inference']['num_inference_steps'],
                    guidance_scale=self.config['inference']['guidance_scale'],
                    eta=self.config['inference']['eta'],
                    use_ema=True,
                    verbose=False
                )
                
                # 计算IS分数
                try:
                    # 确保图像在正确范围内
                    images_for_is = torch.clamp(generated_images.cpu(), 0, 1)
                    
                    is_mean, is_std = self.metrics_calculator.inception_calculator.calculate_inception_score(
                        images_for_is, splits=min(10, num_samples//2)
                    )
                    
                    # 验证IS分数合理性
                    if is_mean < 1.0 or is_mean > 50.0:
                        print(f"⚠️ IS分数异常: {is_mean:.3f}, 可能计算有误")
                    
                    return is_mean, is_std
                    
                except Exception as e:
                    print(f"⚠️ IS分数计算失败: {e}")
                    return None, None
                
                # 保存样本
                if epoch % self.config['training']['sample_interval'] == 0:
                    import torchvision
                    sample_path = os.path.join(self.save_dir, f'samples_epoch_{epoch}.png')
                    torchvision.utils.save_image(
                        generated_images, sample_path,
                        nrow=4, normalize=True, value_range=(0, 1)
                    )
                    print(f"📸 样本已保存: {sample_path}")
                
        except Exception as e:
            print(f"⚠️ 样本生成失败: {e}")
            return None, None
    
    def train(self):
        """主训练循环"""
        print("🚀 启动增强版LDM训练")
        print("=" * 80)
        print("📊 模型特点:")
        print("  - 简化增强版U-Net (115.8M参数)")
        print("  - 6个注意力层 × 2层Transformer = 12个Transformer块")
        print("  - 多头注意力机制增强空间理解能力")
        print("  - P100显存友好设计")
        print("  - 集成完整指标评估：IS、噪声精度、FID等")
        print("=" * 80)
        print("🎯 训练配置优化:")
        print(f"  - batch_size: {self.config['dataset']['batch_size']}")
        print(f"  - 梯度累积: {self.gradient_accumulation_steps}")
        print(f"  - 学习率: {self.config['training']['lr']}")
        print(f"  - 早停耐心: {self.early_stopping_patience}轮")
        print("=" * 80)
        
        # 🚀 构建优化的数据加载器
        print("\n🚀 构建数据加载器（应用性能优化）...")
        train_loader, val_loader = build_dataloader(self.config)
        
        print(f"✅ 数据加载器创建完成 (性能优化版本)")
        print(f"   每epoch约 {len(train_loader)} 个batch")
        
        # 预计算真实数据特征（用于FID）
        print("🔄 预计算真实数据特征...")
        self.fid_evaluator.compute_real_features(val_loader)
        
        start_time = time.time()
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            # 🚀 显存优化：定期清理缓存
            if epoch % self.config.get('performance', {}).get('empty_cache_interval', 30) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"🧹 已清理GPU缓存 (Epoch {epoch})")
            
            # 训练一个epoch
            epoch_metrics = self.train_epoch(train_loader, epoch)
            
            # 生成样本并计算IS
            is_mean, is_std = self.generate_samples(epoch)
            if is_mean is not None:
                epoch_metrics['inception_score_mean'] = is_mean
                epoch_metrics['inception_score_std'] = is_std
            
            # FID评估
            if epoch % self.config['fid_evaluation']['eval_interval'] == 0:
                fid_score = self.evaluate_fid(epoch)
                epoch_metrics['fid_score'] = fid_score
                
                # 🚀 检测训练改进
                improvements = self.detect_improvements(epoch_metrics, epoch)
                
                # 早停检查
                if fid_score < self.best_fid:
                    self.best_fid = fid_score
                    self.patience_counter = 0
                    
                    # 保存最佳模型
                    best_model_path = os.path.join(self.save_dir, 'best_fid_model.pth')
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"🏆 新的最佳FID: {fid_score:.2f}，模型已保存")
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"⏹️ 早停触发，{self.early_stopping_patience}个epoch无改善")
                    break
            else:
                # 即使不是FID评估轮次，也检测损失和IS改进
                improvements = self.detect_improvements(epoch_metrics, epoch)
            
            # 🚀 更新最佳指标记录
            if epoch_metrics['loss'] < self.best_loss:
                self.best_loss = epoch_metrics['loss']
                print(f"📉 新的最佳损失: {self.best_loss:.4f}")
            
            if 'inception_score_mean' in epoch_metrics and epoch_metrics['inception_score_mean'] > self.best_is_score:
                self.best_is_score = epoch_metrics['inception_score_mean']
                print(f"⭐ 新的最佳IS分数: {self.best_is_score:.3f}")
            
            # 定期保存模型
            if epoch % self.config['training']['save_interval'] == 0:
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_fid': self.best_fid,
                    'best_loss': self.best_loss,
                    'best_is_score': self.best_is_score,
                    'improvement_history': self.improvement_history,
                    'config': self.config
                }, checkpoint_path)
                print(f"💾 检查点已保存: {checkpoint_path}")
            
            # 记录日志
            self.training_logs.append(epoch_metrics)
            
            # 🚀 打印详细指标（改进版）
            print(f"\n📊 Epoch {epoch} 指标总结:")
            print(f"  训练损失: {epoch_metrics['loss']:.4f} {'🔥' if improvements.get('loss_improved', False) else ''}")
            print(f"  噪声PSNR: {epoch_metrics['noise_psnr']:.2f}dB")
            print(f"  余弦相似度: {epoch_metrics['noise_cosine_similarity']:.6f}")
            print(f"  相关系数: {epoch_metrics['noise_correlation']:.6f}")
            if 'inception_score_mean' in epoch_metrics:
                print(f"  IS分数: {epoch_metrics['inception_score_mean']:.3f} ± {epoch_metrics['inception_score_std']:.3f} {'🌟' if improvements.get('is_improved', False) else ''}")
            if 'fid_score' in epoch_metrics:
                print(f"  FID分数: {epoch_metrics['fid_score']:.2f} {'🎉' if improvements.get('fid_improved', False) else ''}")
            if epoch_metrics['gradient_health_issues'] > 0:
                print(f"  ⚠️ 梯度异常: {epoch_metrics['gradient_health_issues']} 次")
            
            # 🚀 显示历史最佳记录
            print(f"  📈 历史最佳: 损失={self.best_loss:.4f}, IS={self.best_is_score:.3f}, FID={self.best_fid:.2f}")
            print("-" * 60)
        
        total_time = time.time() - start_time
        print(f"\n✅ 训练完成！总用时: {total_time/3600:.2f}小时")
        print(f"🏆 最终成绩:")
        print(f"   最佳FID分数: {self.best_fid:.2f}")
        print(f"   最佳损失: {self.best_loss:.4f}")
        print(f"   最佳IS分数: {self.best_is_score:.3f}")
        
        # 🚀 统计改进次数
        improvement_count = sum(1 for record in self.improvement_history if record['improvements']['overall_improved'])
        print(f"   训练改进次数: {improvement_count} 次")
        
        # 保存最终模型和日志
        final_model_path = os.path.join(self.save_dir, 'final_model.pth')
        torch.save(self.model.state_dict(), final_model_path)
        
        import json
        log_path = os.path.join(self.save_dir, 'training_logs.json')
        with open(log_path, 'w') as f:
            json.dump(self.training_logs, f, indent=2)
        
        # 🚀 保存改进历史
        improvement_log_path = os.path.join(self.save_dir, 'improvement_history.json')
        with open(improvement_log_path, 'w') as f:
            json.dump(self.improvement_history, f, indent=2)
        
        print(f"💾 最终模型已保存: {final_model_path}")
        print(f"📊 训练日志已保存: {log_path}")
        print(f"📈 改进历史已保存: {improvement_log_path}")

def main(config_path: str = "config_enhanced_unet.yaml"):
    """主函数"""
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    # 验证配置
    try:
        config = load_config(config_path)
        
        if config['unet']['type'] != 'simple_enhanced':
            print("❌ 配置文件不是增强版类型")
            return
            
        print(f"✅ 配置验证成功: {config['unet']['type']}")
        print(f"✅ Transformer深度: {config['unet']['transformer_depth']}")
        
        # 🚀 设置性能优化
        print("\n🚀 应用性能优化...")
        compile_model = setup_performance_optimizations(config)
        
    except Exception as e:
        print(f"❌ 配置文件错误: {e}")
        return
    
    # 创建训练器并开始训练
    trainer = EnhancedTrainer(config_path, compile_model=compile_model)
    trainer.train()

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_performance_optimizations(config):
    """设置性能优化参数"""
    if 'performance' in config:
        perf_config = config['performance']
        
        # 设置PyTorch线程数
        if 'torch_num_threads' in perf_config:
            torch.set_num_threads(perf_config['torch_num_threads'])
            print(f"  🚀 设置PyTorch线程数: {perf_config['torch_num_threads']}")
        
        # 启用Tensor编译加速
        if perf_config.get('compile_model', False) and hasattr(torch, 'compile'):
            print(f"  🚀 启用torch.compile模型编译加速")
            return True
        
    return False

def create_optimized_dataloader(dataset, config):
    """创建优化的数据加载器"""
    dataset_config = config['dataset']
    
    # 基础参数
    batch_size = dataset_config['batch_size']
    num_workers = dataset_config['num_workers']
    
    # 性能优化参数
    pin_memory = dataset_config.get('pin_memory', True)
    persistent_workers = dataset_config.get('persistent_workers', True) and num_workers > 0
    
    # 预取参数
    prefetch_factor = 2  # 默认值
    if 'performance' in config:
        prefetch_factor = config['performance'].get('prefetch_factor', 2)
    
    print(f"  🚀 数据加载器优化:")
    print(f"     - batch_size: {batch_size}")
    print(f"     - num_workers: {num_workers}")
    print(f"     - pin_memory: {pin_memory}")
    print(f"     - persistent_workers: {persistent_workers}")
    print(f"     - prefetch_factor: {prefetch_factor}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataset_config.get('shuffle_train', True),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else 2,
        drop_last=True  # 确保batch size一致
    )

if __name__ == "__main__":
    main() 
