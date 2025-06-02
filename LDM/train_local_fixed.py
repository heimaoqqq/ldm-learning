"""
VAE-LDM本地修复版训练脚本
使用正确的数据集路径，解决FID=300+问题
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time

# 添加VAE路径
sys.path.append('../VAE')

from vae_ldm import create_vae_ldm
from dataset_adapter import build_dataloader
from fid_evaluation import FIDEvaluator
from metrics import denormalize_for_metrics

def check_dataset_loading(data_dir):
    """检查数据集是否正确加载"""
    print(f"🔍 检查数据集路径: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据集路径不存在: {data_dir}")
        return False
    
    # 检查ID文件夹
    id_folders = [d for d in os.listdir(data_dir) if d.startswith('ID_') and os.path.isdir(os.path.join(data_dir, d))]
    print(f"✅ 找到 {len(id_folders)} 个ID文件夹: {id_folders}")
    
    # 检查第一个文件夹的图像
    if id_folders:
        first_folder = os.path.join(data_dir, id_folders[0])
        image_files = [f for f in os.listdir(first_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"✅ {id_folders[0]} 中有 {len(image_files)} 个图像文件")
        
        if image_files:
            print(f"📷 示例文件: {image_files[0]}")
            return True
    
    return False

class LocalFixedVAELDMTrainer:
    """本地修复版VAE-LDM训练器"""
    
    def __init__(self, config_path="config_local_fixed.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 使用设备: {self.device}")
        
        # 🔧 关键：检查数据集
        data_dir = self.config['paths']['data_dir']
        if not check_dataset_loading(data_dir):
            raise ValueError(f"数据集加载失败: {data_dir}")
        
        # 创建保存目录
        self.save_dir = self.config['paths']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化模型
        print("🔧 初始化修复版VAE-LDM...")
        self.model = self._create_fixed_model()
        
        # 初始化数据加载器
        print("📊 初始化数据加载器...")
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # 验证数据加载器
        self._validate_dataloader()
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()
        
        # 训练状态
        self.current_epoch = 0
        self.current_step = 0
        self.best_fid = float('inf')
        
        print("✅ 本地修复版VAE-LDM训练器初始化完成")
    
    def _create_fixed_model(self):
        """创建修复版模型"""
        vae_checkpoint = self.config['paths']['vae_checkpoint']
        
        if not os.path.exists(vae_checkpoint):
            print(f"⚠️ VAE权重文件不存在: {vae_checkpoint}")
            print("🔄 将使用随机初始化VAE")
            vae_checkpoint = None
        else:
            print(f"✅ 找到VAE权重: {vae_checkpoint}")
        
        model = create_vae_ldm(
            vae_checkpoint_path=vae_checkpoint,
            image_size=self.config['model']['image_size'],
            num_classes=self.config['model']['num_classes'],
            diffusion_steps=self.config['model']['diffusion_steps'],
            noise_schedule=self.config['model']['noise_schedule'],
            device=self.device
        )
        
        return model
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        config = self.config['training']
        
        dataloader_config = {
            'dataset': {
                'root_dir': self.config['paths']['data_dir'],
                'batch_size': config['batch_size'],
                'num_workers': config.get('num_workers', 0),
                'val_split': 0.1,
                'shuffle_train': True
            },
            'unet': {
                'num_classes': self.config['model']['num_classes']
            }
        }
        
        return build_dataloader(dataloader_config)
    
    def _validate_dataloader(self):
        """验证数据加载器是否正确加载真实数据"""
        print("🔍 验证数据加载器...")
        
        try:
            sample_batch = next(iter(self.val_loader))
            images = sample_batch['image']
            labels = sample_batch['label']
            
            print(f"✅ 成功加载批次:")
            print(f"   图像形状: {images.shape}")
            print(f"   标签形状: {labels.shape}")
            print(f"   图像值范围: [{images.min():.3f}, {images.max():.3f}]")
            print(f"   图像均值: {images.mean():.3f}")
            print(f"   图像标准差: {images.std():.3f}")
            
            # 保存一个批次的真实图像样本用于验证
            from torchvision.utils import save_image
            display_images = denormalize_for_metrics(images[:16])
            os.makedirs('debug_output', exist_ok=True)
            save_image(
                display_images, 
                'debug_output/real_data_verification.png',
                nrow=4, 
                normalize=False
            )
            print("📷 真实数据样本已保存到: debug_output/real_data_verification.png")
            
        except Exception as e:
            print(f"❌ 数据加载器验证失败: {e}")
            raise
    
    def _create_optimizer(self):
        """创建优化器"""
        config = self.config['training']
        
        optimizer = torch.optim.AdamW(
            self.model.unet.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        return optimizer
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        epoch_losses = []
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch+1}/{self.config['training']['max_epochs']}"
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device, dtype=torch.long)
            
            # 前向传播
            losses = self.model(images, labels, current_epoch=self.current_epoch)
            loss = losses['loss'].mean()
            
            # 梯度累积
            accumulation_steps = self.config['training']['gradient_accumulation_steps']
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.unet.parameters(), 
                    self.config['gpu']['gradient_clip']
                )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.current_step += 1
                
                # 记录日志
                if self.current_step % self.config['training']['log_interval'] == 0:
                    progress_bar.set_postfix({
                        'loss': f'{loss.item() * accumulation_steps:.4f}',
                        'step': self.current_step,
                        'best_fid': f'{self.best_fid:.1f}'
                    })
            
            epoch_losses.append(loss.item() * accumulation_steps)
        
        return {'train_loss': np.mean(epoch_losses)}
    
    def evaluate_fid(self):
        """评估FID分数"""
        print("📊 评估FID分数...")
        self.model.eval()
        
        eval_config = self.config['evaluation']
        
        # 创建FID评估器
        fid_evaluator = FIDEvaluator(device=self.device)
        
        # 计算真实数据特征
        print("   计算真实数据特征...")
        fid_evaluator.compute_real_features(
            self.val_loader, 
            max_samples=eval_config.get('max_real_samples', 500)
        )
        
        # 生成样本
        print("   生成样本...")
        num_samples = eval_config['eval_classes'] * eval_config['samples_per_class']
        
        # 生成类别标签
        samples_per_class = eval_config['samples_per_class']
        class_labels = []
        for class_id in range(eval_config['eval_classes']):
            class_labels.extend([class_id] * samples_per_class)
        
        class_labels = torch.tensor(class_labels, device=self.device, dtype=torch.long)
        
        # 批量生成
        batch_size = eval_config.get('generation_batch_size', 8)
        generated_images = []
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                batch_labels = class_labels[i:i+current_batch_size]
                
                batch_images, _ = self.model.sample(
                    num_samples=current_batch_size,
                    class_labels=batch_labels,
                    use_ddim=True,
                    eta=eval_config.get('eta', 0.0)
                )
                
                generated_images.append(batch_images)
        
        # 合并并计算FID
        generated_images = torch.cat(generated_images, dim=0)
        fid_score = fid_evaluator.calculate_fid_score(generated_images)
        
        return fid_score
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'unet_state_dict': self.model.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_fid': self.best_fid,
            'config': self.config
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_fid_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 最佳FID检查点已保存: {best_path}")
    
    def save_sample_images(self):
        """保存生成样本"""
        print("🎨 生成样本图像...")
        self.model.eval()
        
        with torch.no_grad():
            num_samples = 16
            class_labels = torch.randint(0, self.config['model']['num_classes'], 
                                       (num_samples,), device=self.device)
            
            generated_images, _ = self.model.sample(
                num_samples=num_samples,
                class_labels=class_labels,
                use_ddim=True,
                eta=0.0
            )
            
            # 保存图像
            from torchvision.utils import save_image
            
            display_images = denormalize_for_metrics(generated_images)
            save_path = os.path.join(self.save_dir, f'samples_epoch_{self.current_epoch}.png')
            save_image(display_images, save_path, nrow=4, normalize=False)
            print(f"💾 样本已保存: {save_path}")
    
    def train(self):
        """主训练循环"""
        print("🚀 开始训练本地修复版VAE-LDM...")
        print(f"📊 配置: {self.config['training']['max_epochs']} epochs, "
              f"batch_size={self.config['training']['batch_size']}")
        
        start_time = time.time()
        
        for epoch in range(self.config['training']['max_epochs']):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            print(f"\nEpoch {epoch+1}/{self.config['training']['max_epochs']}")
            print(f"  训练损失: {train_metrics['train_loss']:.6f}")
            
            # 定期评估FID
            if (epoch + 1) % self.config['training']['eval_interval'] == 0:
                try:
                    fid_score = self.evaluate_fid()
                    print(f"  📊 FID分数: {fid_score:.2f}")
                    
                    # 检查是否是最佳分数
                    is_best = fid_score < self.best_fid
                    if is_best:
                        self.best_fid = fid_score
                        print(f"  🎉 新的最佳FID分数: {fid_score:.2f}")
                    
                    # 保存检查点
                    self.save_checkpoint(is_best=is_best)
                    
                except Exception as e:
                    print(f"  ⚠️ FID评估失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 定期保存样本
            if (epoch + 1) % self.config['debug'].get('save_interval_samples', 10) == 0:
                try:
                    self.save_sample_images()
                except Exception as e:
                    print(f"  ⚠️ 样本保存失败: {e}")
        
        total_time = time.time() - start_time
        print(f"\n✅ 训练完成")
        print(f"📊 最佳FID分数: {self.best_fid:.2f}")
        print(f"⏱️ 总训练时间: {total_time/3600:.2f} 小时")

def main():
    """主函数"""
    print("🔧 初始化本地修复版VAE-LDM训练...")
    
    # 检查配置文件
    config_file = "config_local_fixed.yaml"
    if not os.path.exists(config_file):
        print(f"⚠️ 配置文件 {config_file} 不存在")
        return
    
    try:
        trainer = LocalFixedVAELDMTrainer(config_file)
        trainer.train()
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
