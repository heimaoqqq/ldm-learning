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
    
    def __init__(self, config_path: str):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 使用设备: {self.device}")
        
        # 创建模型
        self.model = create_stable_ldm(self.config).to(self.device)
        
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
        self.patience_counter = 0
        self.early_stopping_patience = self.config['training'].get('early_stopping_patience', 50)
        
        # 日志存储
        self.training_logs = []
        
        # 创建保存目录
        self.save_dir = self.config['training']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
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
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 前向传播 - 确保数据在正确的设备上
            images = batch['image'].to(self.device)
            labels = batch.get('label', None)
            if labels is not None:
                labels = labels.to(self.device)
            
            loss_dict = self.model(images, labels)
            
            loss = loss_dict['loss'] / self.gradient_accumulation_steps
            
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
                'Noise MSE': f'{noise_metrics.get("noise_mse", 0):.4f}',
                'Noise PSNR': f'{noise_metrics.get("noise_psnr", 0):.2f}',
                'Cosine Sim': f'{noise_metrics.get("noise_cosine_similarity", 0):.3f}'
            })
        
        # 计算epoch平均指标
        avg_metrics = {
            'epoch': epoch,
            'loss': np.mean(epoch_losses),
            'noise_mse': np.mean(epoch_metrics['noise_mse']),
            'noise_mae': np.mean(epoch_metrics['noise_mae']),
            'noise_psnr': np.mean(epoch_metrics['noise_psnr']),
            'noise_cosine_similarity': np.mean(epoch_metrics['noise_cosine_similarity']),
            'noise_correlation': np.mean(epoch_metrics['noise_correlation'])
        }
        
        return avg_metrics
    
    def evaluate_fid(self, epoch: int) -> float:
        """评估FID分数"""
        try:
            self.model.eval()
            
            # 使用EMA权重（如果可用）
            if hasattr(self.model, 'ema') and self.model.ema is not None:
                self.model.ema.apply_shadow()
            
            fid_score = self.fid_evaluator.evaluate_model(
                self.model,
                num_samples=self.config['fid_evaluation']['num_samples'],
                batch_size=self.config['fid_evaluation']['batch_size'],
                num_classes=self.config['unet']['num_classes'],
                num_inference_steps=self.config['inference']['num_inference_steps'],
                guidance_scale=self.config['inference']['guidance_scale'],
                eta=self.config['inference']['eta']
            )
            
            # 恢复原始权重
            if hasattr(self.model, 'ema') and self.model.ema is not None:
                self.model.ema.restore()
            
            return fid_score
            
        except Exception as e:
            print(f"⚠️ FID评估失败: {e}")
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
                is_mean, is_std = self.metrics_calculator.inception_calculator.calculate_inception_score(
                    generated_images.cpu()
                )
                
                # 保存样本
                if epoch % self.config['training']['sample_interval'] == 0:
                    import torchvision
                    sample_path = os.path.join(self.save_dir, f'samples_epoch_{epoch}.png')
                    torchvision.utils.save_image(
                        generated_images, sample_path,
                        nrow=4, normalize=True, value_range=(0, 1)
                    )
                    print(f"📸 样本已保存: {sample_path}")
                
                return is_mean, is_std
                
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
        
        # 构建数据加载器
        train_loader, val_loader = build_dataloader(self.config)
        
        # 预计算真实数据特征（用于FID）
        print("🔄 预计算真实数据特征...")
        self.fid_evaluator.compute_real_features(val_loader)
        
        start_time = time.time()
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
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
            
            # 定期保存模型
            if epoch % self.config['training']['save_interval'] == 0:
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_fid': self.best_fid,
                    'config': self.config
                }, checkpoint_path)
                print(f"💾 检查点已保存: {checkpoint_path}")
            
            # 记录日志
            self.training_logs.append(epoch_metrics)
            
            # 打印详细指标
            print(f"\n📊 Epoch {epoch} 指标总结:")
            print(f"  训练损失: {epoch_metrics['loss']:.4f}")
            print(f"  噪声MSE: {epoch_metrics['noise_mse']:.4f}")
            print(f"  噪声PSNR: {epoch_metrics['noise_psnr']:.2f}dB")
            print(f"  余弦相似度: {epoch_metrics['noise_cosine_similarity']:.3f}")
            print(f"  相关系数: {epoch_metrics['noise_correlation']:.3f}")
            if 'inception_score_mean' in epoch_metrics:
                print(f"  IS分数: {epoch_metrics['inception_score_mean']:.2f} ± {epoch_metrics['inception_score_std']:.2f}")
            if 'fid_score' in epoch_metrics:
                print(f"  FID分数: {epoch_metrics['fid_score']:.2f}")
            print("-" * 60)
        
        total_time = time.time() - start_time
        print(f"\n✅ 训练完成！总用时: {total_time/3600:.2f}小时")
        print(f"🏆 最佳FID分数: {self.best_fid:.2f}")
        
        # 保存最终模型和日志
        final_model_path = os.path.join(self.save_dir, 'final_model.pth')
        torch.save(self.model.state_dict(), final_model_path)
        
        import json
        log_path = os.path.join(self.save_dir, 'training_logs.json')
        with open(log_path, 'w') as f:
            json.dump(self.training_logs, f, indent=2)
        
        print(f"💾 最终模型已保存: {final_model_path}")
        print(f"📊 训练日志已保存: {log_path}")

def main(config_path: str = "config_enhanced_unet.yaml"):
    """主函数"""
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    # 验证配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config['unet']['type'] != 'simple_enhanced':
            print("❌ 配置文件不是增强版类型")
            return
            
        print(f"✅ 配置验证成功: {config['unet']['type']}")
        print(f"✅ Transformer深度: {config['unet']['transformer_depth']}")
        
    except Exception as e:
        print(f"❌ 配置文件错误: {e}")
        return
    
    # 创建训练器并开始训练
    trainer = EnhancedTrainer(config_path)
    trainer.train()

if __name__ == "__main__":
    main() 
