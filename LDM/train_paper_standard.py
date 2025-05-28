"""
严格按照论文标准的CLDM训练脚本
基于ControlNet和Stable Diffusion论文参数
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

# 添加路径
sys.path.append('../VAE')
from adv_vq_vae import AdvVQVAE
from dataset import build_dataloader

# 导入论文标准模型
from cldm_paper_standard import PaperStandardCLDM

class EMA:
    """指数移动平均 - 论文推荐"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_model_and_optimizer(config):
    """设置模型和优化器 - 论文标准"""
    cldm_config = config['cldm']
    training_config = config['training']
    
    # 创建论文标准CLDM模型
    model = PaperStandardCLDM(
        latent_dim=cldm_config['latent_dim'],
        num_classes=cldm_config['num_classes'],
        num_timesteps=cldm_config['num_timesteps'],
        beta_schedule=cldm_config['beta_schedule'],
        # U-Net参数
        model_channels=cldm_config['model_channels'],
        time_emb_dim=cldm_config['time_emb_dim'],
        class_emb_dim=cldm_config['class_emb_dim'],
        channel_mult=cldm_config['channel_mult'],
        num_res_blocks=cldm_config['num_res_blocks'],
        attention_resolutions=cldm_config['attention_resolutions'],
        dropout=cldm_config['dropout']
    )
    
    # 论文标准Adam优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config['lr'],
        betas=(training_config.get('beta1', 0.9), training_config.get('beta2', 0.999)),
        eps=training_config.get('eps', 1e-8),
        weight_decay=training_config.get('weight_decay', 0.0)
    )
    
    # 论文推荐的余弦衰减调度器
    if training_config.get('use_scheduler', False):
        if training_config['scheduler_type'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=training_config['epochs'],
                eta_min=training_config.get('min_lr', 1e-6)
            )
        else:
            scheduler = None
    else:
        scheduler = None
    
    # EMA - 论文推荐
    ema = None
    if cldm_config.get('use_ema', False):
        ema = EMA(model, decay=cldm_config.get('ema_decay', 0.9999))
    
    return model, optimizer, scheduler, ema

def load_vqvae(config):
    """加载预训练VQ-VAE"""
    vqvae_config = config['vqvae']
    
    # 创建VQ-VAE模型
    vqvae = AdvVQVAE(
        in_channels=vqvae_config['in_channels'],
        latent_dim=vqvae_config['latent_dim'],
        num_embeddings=vqvae_config['num_embeddings'],
        beta=vqvae_config['beta'],
        decay=vqvae_config['decay'],
        groups=vqvae_config['groups']
    )
    
    # 加载预训练权重
    checkpoint = torch.load(vqvae_config['model_path'], map_location='cpu')
    vqvae.load_state_dict(checkpoint)
    vqvae.eval()
    
    return vqvae

def train_epoch(model, vqvae, dataloader, optimizer, device, ema=None, grad_clip_norm=None):
    """训练一个epoch - 论文标准流程"""
    model.train()
    vqvae.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        
        # 使用VQ-VAE编码到潜在空间
        with torch.no_grad():
            z_e = vqvae.encoder(images)
            z_q, _, _ = vqvae.vq(z_e)
        
        # CLDM前向传播
        loss, pred_noise, target_noise = model(z_q, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪 - 防止梯度爆炸
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
        optimizer.step()
        
        # EMA更新
        if ema is not None:
            ema.update()
        
        # 统计
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'Avg Loss': f'{total_loss/num_batches:.6f}'
        })
    
    return total_loss / num_batches

@torch.no_grad()
def validate_epoch(model, vqvae, dataloader, device, ema=None):
    """验证 - 使用EMA权重"""
    if ema is not None:
        ema.apply_shadow()
    
    model.eval()
    vqvae.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for images, labels in tqdm(dataloader, desc="Validation"):
        images = images.to(device)
        labels = labels.to(device)
        
        # VQ-VAE编码
        z_e = vqvae.encoder(images)
        z_q, _, _ = vqvae.vq(z_e)
        
        # CLDM前向传播
        loss, _, _ = model(z_q, labels)
        
        total_loss += loss.item()
        num_batches += 1
    
    if ema is not None:
        ema.restore()
    
    return total_loss / num_batches

@torch.no_grad()
def sample_images(model, vqvae, device, num_classes, num_samples_per_class=4, 
                 sampling_steps=50, eta=0.0, ema=None):
    """生成样本图像"""
    if ema is not None:
        ema.apply_shadow()
    
    model.eval()
    vqvae.eval()
    
    all_samples = []
    
    for class_id in range(min(8, num_classes)):  # 限制类别数避免内存问题
        class_labels = torch.full((num_samples_per_class,), class_id, device=device, dtype=torch.long)
        
        # CLDM采样潜在表示
        latent_samples = model.sample(
            class_labels=class_labels,
            device=device,
            num_inference_steps=sampling_steps,
            eta=eta
        )
        
        # VQ-VAE解码到图像空间
        image_samples = vqvae.decoder(latent_samples)
        all_samples.append(image_samples)
    
    if ema is not None:
        ema.restore()
    
    return torch.cat(all_samples, dim=0)

def save_checkpoint(model, optimizer, scheduler, ema, epoch, loss, save_dir):
    """保存检查点"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if ema is not None:
        checkpoint['ema_shadow'] = ema.shadow
    
    # 保存最新检查点
    torch.save(checkpoint, os.path.join(save_dir, 'latest_checkpoint.pth'))
    
    # 保存epoch检查点
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))

def main():
    parser = argparse.ArgumentParser(description='论文标准CLDM训练')
    parser.add_argument('--config', type=str, default='config_paper_standard.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设备配置
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(config['system']['seed'])
    np.random.seed(config['system']['seed'])
    
    # 加载数据
    print("加载数据集...")
    train_loader, val_loader, train_size, val_size = build_dataloader(
        root_dir=config['dataset']['root_dir'],
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        val_split=0.2
    )
    print(f"训练集: {train_size}, 验证集: {val_size}")
    
    # 加载VQ-VAE
    print("加载VQ-VAE...")
    vqvae = load_vqvae(config).to(device)
    
    # 设置模型和优化器
    print("初始化CLDM模型...")
    model, optimizer, scheduler, ema = setup_model_and_optimizer(config)
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计{total_params:,}, 可训练{trainable_params:,}")
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        print(f"从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if ema and 'ema_shadow' in checkpoint:
            ema.shadow = checkpoint['ema_shadow']
        start_epoch = checkpoint['epoch'] + 1
        print(f"从epoch {start_epoch}开始训练")
    
    # 训练配置
    training_config = config['training']
    epochs = training_config['epochs']
    val_interval = training_config['val_interval']
    save_interval = training_config['save_interval']
    sample_interval = training_config['sample_interval']
    grad_clip_norm = training_config.get('grad_clip_norm')
    save_dir = training_config['save_dir']
    
    # 训练循环
    print("开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        
        # 训练
        train_loss = train_epoch(
            model, vqvae, train_loader, optimizer, device, ema, grad_clip_norm
        )
        print(f"训练损失: {train_loss:.6f}")
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
            print(f"学习率: {scheduler.get_last_lr()[0]:.8f}")
        
        # 验证
        if (epoch + 1) % val_interval == 0:
            val_loss = validate_epoch(model, vqvae, val_loader, device, ema)
            print(f"验证损失: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, ema, epoch, val_loss, 
                              os.path.join(save_dir, 'best'))
                print(f"保存最佳模型 (验证损失: {val_loss:.6f})")
        
        # 定期保存
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, ema, epoch, train_loss, save_dir)
            print(f"保存检查点: epoch {epoch+1}")
        
        # 生成样本
        if (epoch + 1) % sample_interval == 0:
            print("生成样本图像...")
            try:
                samples = sample_images(
                    model, vqvae, device, 
                    num_classes=config['dataset']['num_classes'],
                    sampling_steps=training_config.get('sampling_steps', 50),
                    ema=ema
                )
                
                # 保存样本（这里可以添加保存图像的代码）
                sample_dir = os.path.join(save_dir, 'samples')
                os.makedirs(sample_dir, exist_ok=True)
                torch.save(samples.cpu(), os.path.join(sample_dir, f'samples_epoch_{epoch+1}.pt'))
                print(f"样本已保存到: {sample_dir}")
                
            except Exception as e:
                print(f"样本生成失败: {e}")
    
    print("训练完成!")

if __name__ == "__main__":
    main() 