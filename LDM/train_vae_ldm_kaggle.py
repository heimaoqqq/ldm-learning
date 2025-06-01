"""
VAE-LDM Kaggle完整训练脚本
使用YAML配置文件，包含完整训练器，适配P100 GPU资源限制
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
from typing import Dict, Any, Optional

# 添加VAE路径 - 适配不同环境
vae_path_options = [
    '/kaggle/working/VAE',  # Kaggle环境
    os.path.join(os.path.dirname(__file__), '..', 'VAE'),  # 相对路径
    '../VAE'  # 备用相对路径
]

for vae_path in vae_path_options:
    if os.path.exists(vae_path):
        sys.path.append(vae_path)
        print(f"✅ VAE路径已添加: {vae_path}")
        break
else:
    print("⚠️ 未找到VAE目录，可能导致导入错误")

from vae_ldm import create_vae_ldm
from dataset_adapter import build_dataloader
from fid_evaluation import FIDEvaluator
from metrics import denormalize_for_metrics

class VAELDMTrainer:
    """VAE-LDM训练器 - Kaggle优化版"""
    
    def __init__(
        self,
        vae_checkpoint_path: str,
        data_dir: str,
        save_dir: str,
        num_classes: int = 31,
        image_size: int = 32,
        diffusion_steps: int = 1000,
        noise_schedule: str = "cosine",
        batch_size: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        max_epochs: int = 100,
        log_interval: int = 10,
        save_interval_epochs: int = 50,  # epoch-based保存间隔
        eval_interval: int = 10,
        gradient_accumulation_steps: int = 2,
        enable_wandb: bool = False,
        device: str = 'cuda'
    ):
        self.vae_checkpoint_path = vae_checkpoint_path
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.log_interval = log_interval
        self.save_interval_epochs = save_interval_epochs  # 修改为epoch-based
        self.eval_interval = eval_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.enable_wandb = enable_wandb
        self.device = device
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化模型
        print("🔧 初始化VAE-LDM模型...")
        self.model = create_vae_ldm(
            vae_checkpoint_path=vae_checkpoint_path,
            image_size=image_size,
            num_classes=num_classes,
            diffusion_steps=diffusion_steps,
            noise_schedule=noise_schedule,
            device=device
        )
        
        # 初始化数据加载器
        print("📊 初始化数据加载器...")
        dataloader_config = {
            'dataset': {
                'root_dir': data_dir,
                'batch_size': batch_size,
                'num_workers': 4,
                'val_split': 0.1,
                'shuffle_train': True
            },
            'unet': {
                'num_classes': num_classes
            }
        }
        self.train_loader, self.val_loader = build_dataloader(dataloader_config)
        
        # 初始化优化器
        print("⚙️ 初始化优化器...")
        self.optimizer = self.model.configure_optimizers(lr=lr, weight_decay=weight_decay)
        
        # 初始化FID评估器
        print("📈 初始化FID评估器...")
        self.fid_evaluator = FIDEvaluator(device=self.device)
        self.fid_evaluator.compute_real_features(self.val_loader)
        
        # 训练状态
        self.current_epoch = 0
        self.current_step = 0
        self.best_fid = float('inf')
        
        print("✅ VAELDMTrainer 初始化完成")
        
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.max_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device, dtype=torch.long)
            
            # 前向传播
            losses = self.model(images, labels)
            loss = losses['loss'].mean()
            
            # 梯度累积
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.unet.parameters(), 1.0)
                
                # 优化器步骤
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                self.current_step += 1
                
                # 记录日志
                if self.current_step % self.log_interval == 0:
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'step': self.current_step
                    })
            
            epoch_losses.append(loss.item() * self.gradient_accumulation_steps)
            
        return {'train_loss': np.mean(epoch_losses)}
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型（计算FID）"""
        print("📊 开始FID评估...")
        self.model.eval()
        
        # 生成样本用于FID计算
        num_samples = 150  # 3类 × 50样本
        class_labels = torch.randint(0, 31, (num_samples,), device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            generated_images, _ = self.model.sample(
                num_samples=num_samples,
                class_labels=class_labels,
                use_ddim=True,
                eta=0.0
            )
        
        # 反归一化到[0,1]用于FID计算
        generated_images = denormalize_for_metrics(generated_images)
        
        # 计算FID
        fid_score = self.fid_evaluator.calculate_fid_score(generated_images)
        
        return {'fid': fid_score}
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{self.current_epoch}.pt')
        
        self.model.save_checkpoint(
            filepath=checkpoint_path,
            epoch=self.current_epoch,
            optimizer_state=self.optimizer.state_dict()
        )
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            self.model.save_checkpoint(
                filepath=best_path,
                epoch=self.current_epoch,
                optimizer_state=self.optimizer.state_dict()
            )
            print(f"💾 最佳模型已保存: {best_path}")
    
    def train(self):
        """主训练循环"""
        print("🚀 开始VAE-LDM训练")
        print(f"📊 训练数据: {len(self.train_loader)} batches")
        print(f"📈 验证数据: {len(self.val_loader)} batches")
        print(f"💾 每 {self.save_interval_epochs} epochs 保存模型")
        print(f"📊 每 {self.eval_interval} epochs 评估FID")
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            print(f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {train_metrics['train_loss']:.4f}")
            
            # 定期评估
            if (epoch + 1) % self.eval_interval == 0:
                eval_metrics = self.evaluate()
                fid_score = eval_metrics['fid']
                
                print(f"📊 FID Score: {fid_score:.2f}")
                
                # 检查是否是最佳模型
                is_best = fid_score < self.best_fid
                if is_best:
                    self.best_fid = fid_score
                    print(f"🎉 新的最佳FID: {fid_score:.2f}")
                
                # 保存最佳模型
                if is_best:
                    self.save_checkpoint(is_best=True)
            
            # 定期保存检查点
            if (epoch + 1) % self.save_interval_epochs == 0:
                self.save_checkpoint(is_best=False)
                print(f"💾 检查点已保存: epoch {epoch+1}")
        
        print("🎉 训练完成!")
        print(f"🏆 最佳FID: {self.best_fid:.2f}")

def load_config(config_path: str = "config_kaggle.yaml") -> Dict[str, Any]:
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 配置文件加载成功: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ 配置文件未找到: {config_path}")
        return create_default_config()
    except Exception as e:
        print(f"❌ 配置文件解析失败: {e}")
        return create_default_config()

def create_default_config() -> Dict[str, Any]:
    """创建默认配置（如果YAML文件不存在）"""
    print("📝 使用默认配置...")
    return {
        'paths': {
            'vae_checkpoint': '/kaggle/working/VAE/best_fid_model.pt',
            'data_dir': '/kaggle/input/dataset',
            'save_dir': '/kaggle/working/checkpoints'
        },
        'model': {
            'num_classes': 31,
            'image_size': 32,
            'diffusion_steps': 1000,
            'noise_schedule': 'cosine'
        },
        'training': {
            'batch_size': 4,
            'gradient_accumulation_steps': 2,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'max_epochs': 100,
            'log_interval': 10,
            'save_interval_epochs': 50,
            'eval_interval': 10,
            'enable_wandb': False,
            'num_workers': 2
        },
        'evaluation': {
            'eval_classes': 3,
            'samples_per_class': 50,
            'ddim_steps': 50,
            'eta': 0.0,
            'max_real_samples': 500
        },
        'gpu': {
            'device': 'cuda',
            'mixed_precision': False,
            'gradient_clip': 1.0
        },
        'experiment': {
            'name': 'VAE-LDM-Kaggle',
            'description': 'VAE潜在扩散模型 - Kaggle P100优化版',
            'version': 'v1.0'
        }
    }

def print_config_summary(config: Dict[str, Any]):
    """打印配置摘要"""
    print("\n" + "="*80)
    print("🎯 VAE-LDM Kaggle训练配置")
    print("="*80)
    
    print(f"📝 实验: {config['experiment']['name']} {config['experiment']['version']}")
    print(f"📄 描述: {config['experiment']['description']}")
    print()
    
    print("📁 路径配置:")
    for key, value in config['paths'].items():
        print(f"   {key}: {value}")
    print()
    
    print("🎛️ 模型配置:")
    for key, value in config['model'].items():
        print(f"   {key}: {value}")
    print()
    
    print("🏃 训练配置:")
    train_config = config['training']
    effective_batch = train_config['batch_size'] * train_config['gradient_accumulation_steps']
    print(f"   batch_size: {train_config['batch_size']} (有效batch: {effective_batch})")
    print(f"   learning_rate: {train_config['learning_rate']}")
    print(f"   max_epochs: {train_config['max_epochs']}")
    print(f"   eval_interval: {train_config['eval_interval']} epochs")
    print()
    
    print("📊 评估配置:")
    eval_config = config['evaluation']
    total_samples = eval_config['eval_classes'] * eval_config['samples_per_class']
    print(f"   eval_classes: {eval_config['eval_classes']}")
    print(f"   samples_per_class: {eval_config['samples_per_class']} (总计: {total_samples})")
    print(f"   ddim_steps: {eval_config['ddim_steps']}")
    print()
    
    print("🎮 GPU配置:")
    for key, value in config['gpu'].items():
        print(f"   {key}: {value}")
    
    print("="*80)

def validate_paths(config: Dict[str, Any]) -> bool:
    """验证路径配置"""
    paths = config['paths']
    
    # 检查VAE检查点
    vae_checkpoint_exists = os.path.exists(paths['vae_checkpoint'])
    if not vae_checkpoint_exists:
        print(f"⚠️ VAE检查点未找到: {paths['vae_checkpoint']}")
        print("🔄 将使用随机初始化的VAE（性能会下降）")
        print("💡 建议:")
        print("   1. 检查VAE检查点路径是否正确")
        print("   2. 确保VAE模型已训练完成并上传")
        print("   3. 可以先用随机VAE进行测试")
        
        # 询问是否继续
        print("\n❓ 是否继续训练？(Kaggle环境自动继续)")
        # 在Kaggle环境中自动继续，本地环境可以选择
        if '/kaggle/' not in paths.get('data_dir', ''):
            response = input("输入 'y' 继续，其他键退出: ").lower()
            if response != 'y':
                return False
    else:
        print(f"✅ VAE检查点找到: {paths['vae_checkpoint']}")
    
    # 检查数据集
    if not os.path.exists(paths['data_dir']):
        print(f"❌ 数据集未找到: {paths['data_dir']}")
        print("请确保数据集已上传到Kaggle")
        return False
    
    # 创建保存目录
    os.makedirs(paths['save_dir'], exist_ok=True)
    
    print("✅ 路径验证完成")
    return True

def convert_config_to_trainer_args(config: Dict[str, Any]) -> Dict[str, Any]:
    """将YAML配置转换为训练器参数"""
    return {
        # 路径配置
        'vae_checkpoint_path': config['paths']['vae_checkpoint'],
        'data_dir': config['paths']['data_dir'],
        'save_dir': config['paths']['save_dir'],
        
        # 模型配置
        'num_classes': config['model']['num_classes'],
        'image_size': config['model']['image_size'],
        'diffusion_steps': config['model']['diffusion_steps'],
        'noise_schedule': config['model']['noise_schedule'],
        
        # 训练配置
        'batch_size': config['training']['batch_size'],
        'lr': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'max_epochs': config['training']['max_epochs'],
        'log_interval': config['training']['log_interval'],
        'save_interval_epochs': config['training']['save_interval_epochs'],
        'eval_interval': config['training']['eval_interval'],
        'gradient_accumulation_steps': config['training']['gradient_accumulation_steps'],
        'enable_wandb': config['training']['enable_wandb'],
        
        # GPU配置
        'device': config['gpu']['device'],
    }

def save_training_info(config: Dict[str, Any], save_dir: str):
    """保存训练信息"""
    info = {
        'start_time': datetime.now().isoformat(),
        'config': config,
        'system_info': {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        }
    }
    
    info_path = os.path.join(save_dir, 'training_info.yaml')
    with open(info_path, 'w', encoding='utf-8') as f:
        yaml.dump(info, f, default_flow_style=False, allow_unicode=True)
    
    print(f"📋 训练信息已保存: {info_path}")

def main():
    """Kaggle环境训练主函数"""
    print("🚀 启动VAE-LDM Kaggle训练 (YAML配置版)")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载配置
    config = load_config()
    
    # 打印配置摘要
    print_config_summary(config)
    
    # 验证路径
    if not validate_paths(config):
        return
    
    # 保存训练信息
    save_training_info(config, config['paths']['save_dir'])
    
    print("\n🔧 开始初始化训练器...")
    
    try:
        # 转换配置并创建训练器
        trainer_args = convert_config_to_trainer_args(config)
        trainer = VAELDMTrainer(**trainer_args)
        
        print("🎯 开始训练...")
        
        # 开始训练
        trainer.train()
        
        print("🎉 训练完成!")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"🏁 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 
