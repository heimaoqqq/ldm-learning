"""
严格按照论文标准的CLDM训练脚本
基于ControlNet和Stable Diffusion论文参数
优化版本：存储管理 + FID评估 + 智能检查点
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
import gc
import glob
import torchvision.utils as vutils
import shutil
from datetime import datetime

# 添加路径
sys.path.append('../VAE')
from adv_vq_vae import AdvVQVAE
from dataset import build_dataloader

# 导入论文标准模型
from cldm_paper_standard import PaperStandardCLDM

# 正确的FID计算库
try:
    from torch_fidelity import calculate_metrics
    HAS_FIDELITY = True
    print("✅ 成功导入torch-fidelity库")
    # 设置环境变量避免OpenMP冲突
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
except ImportError:
    HAS_FIDELITY = False
    print("❌ 警告: 未找到torch-fidelity库，FID计算将被跳过")
    print("请运行: pip install torch-fidelity")

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
                # 确保shadow权重在与参数相同的设备上
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                else:
                    # 如果设备不匹配，移动shadow到参数的设备
                    if self.shadow[name].device != param.device:
                        self.shadow[name] = self.shadow[name].to(param.device)
                    self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                # 确保shadow权重在正确的设备上
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 确保backup权重在正确的设备上
                if self.backup[name].device != param.device:
                    self.backup[name] = self.backup[name].to(param.device)
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
        lr=float(training_config['lr']),
        betas=(float(training_config.get('beta1', 0.9)), float(training_config.get('beta2', 0.999))),
        eps=float(training_config.get('eps', 1e-8)),
        weight_decay=float(training_config.get('weight_decay', 0.0))
    )
    
    # 论文推荐的余弦衰减调度器
    if training_config.get('use_scheduler', False):
        if training_config['scheduler_type'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(training_config['epochs']),
                eta_min=float(training_config.get('min_lr', 1e-6))
            )
        else:
            scheduler = None
    else:
        scheduler = None
    
    # EMA - 论文推荐
    ema = None
    if cldm_config.get('use_ema', False):
        ema = EMA(model, decay=float(cldm_config.get('ema_decay', 0.9999)))
    
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

def clean_old_checkpoints(save_dir, keep_best=True, keep_latest=True):
    """清理旧的检查点文件，节省存储空间"""
    try:
        # 获取所有检查点文件
        checkpoint_files = glob.glob(os.path.join(save_dir, 'checkpoint_epoch_*.pth'))
        
        # 保留最新的2个检查点
        if len(checkpoint_files) > 2:
            # 按修改时间排序
            checkpoint_files.sort(key=os.path.getmtime)
            # 删除较旧的文件
            for old_file in checkpoint_files[:-2]:
                try:
                    os.remove(old_file)
                    print(f"清理旧检查点: {os.path.basename(old_file)}")
                except OSError:
                    pass
        
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"清理检查点时出错: {e}")

def calculate_proper_fid(model, vqvae, dataloader, device, num_samples=1000, ema=None):
    """计算正确的FID分数 - 使用Inception网络特征"""
    if not HAS_FIDELITY:
        print("跳过FID计算：torch-fidelity库未安装")
        return float('inf')
    
    # 创建临时目录
    temp_dir = "temp_fid_eval"
    real_dir = os.path.join(temp_dir, "real")
    fake_dir = os.path.join(temp_dir, "fake")
    
    # 清理可能存在的旧目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    try:
        if ema is not None:
            ema.apply_shadow()
        
        model.eval()
        vqvae.eval()
        
        print(f"生成{num_samples}个样本用于FID计算...")
        
        with torch.no_grad():
            # 第一步：收集真实图像
            real_count = 0
            for images, labels in tqdm(dataloader, desc="保存真实图像"):
                if real_count >= num_samples:
                    break
                    
                images = images.to(device)
                batch_size = images.shape[0]
                
                # 保存真实图像
                for i in range(batch_size):
                    if real_count >= num_samples:
                        break
                    
                    # 反归一化到[0,1] - 假设输入是[-1,1]
                    real_img = (images[i] + 1.0) / 2.0
                    real_img = torch.clamp(real_img, 0, 1)
                    
                    vutils.save_image(
                        real_img,
                        os.path.join(real_dir, f"real_{real_count}.png")
                    )
                    real_count += 1
            
            print(f"已保存{real_count}张真实图像")
            
            # 第二步：生成假图像
            fake_count = 0
            for images, labels in tqdm(dataloader, desc="生成假图像"):
                if fake_count >= num_samples:
                    break
                    
                images = images.to(device)
                labels = labels.to(device)
                batch_size = images.shape[0]
                
                # VQ-VAE编码真实图像获得形状
                z_e = vqvae.encoder(images)
                z_q, _, _ = vqvae.vq(z_e)
                
                # 确定本批次要生成的数量
                current_batch_samples = min(batch_size, num_samples - fake_count)
                
                # DDIM采样生成假图像
                fake_latents = model.sample(
                    class_labels=labels[:current_batch_samples],
                    device=device,
                    shape=z_q[:current_batch_samples].shape,
                    num_inference_steps=50,  # 使用足够的采样步数
                    eta=0.0  # DDIM确定性采样
                )
                
                # VQ-VAE解码
                fake_imgs = vqvae.decoder(fake_latents)
                
                # 保存生成图像
                for i in range(current_batch_samples):
                    # 反归一化到[0,1]
                    fake_img = (fake_imgs[i] + 1.0) / 2.0
                    fake_img = torch.clamp(fake_img, 0, 1)
                    
                    vutils.save_image(
                        fake_img,
                        os.path.join(fake_dir, f"fake_{fake_count}.png")
                    )
                    fake_count += 1
            
            print(f"已生成{fake_count}张假图像")
        
        if ema is not None:
            ema.restore()
        
        # 验证样本数量
        if real_count < 50 or fake_count < 50:
            print(f"警告：样本数量太少 (真实:{real_count}, 假:{fake_count})，FID可能不准确")
            return float('inf')
        
        # 计算FID
        print("计算FID分数...")
        try:
            metrics = calculate_metrics(
                input1=real_dir,
                input2=fake_dir,
                cuda=torch.cuda.is_available(),
                fid=True,
                verbose=False
            )
            
            fid_score = metrics['frechet_inception_distance']
            print(f"FID计算完成: {fid_score:.4f}")
            
        except Exception as e:
            print(f"torch-fidelity计算失败: {e}")
            fid_score = float('inf')
        
        # 清理临时文件
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
        return fid_score
        
    except Exception as e:
        print(f"FID计算失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理临时文件
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except:
            pass
        
        return float('inf')

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
        
        # 定期清理内存
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
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

def save_checkpoint(model, optimizer, scheduler, ema, epoch, loss, save_dir, is_best=False):
    """优化的检查点保存"""
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
    latest_path = os.path.join(save_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    if is_best:
        # 保存最佳模型时，先删除旧的最佳模型
        best_path = os.path.join(save_dir, 'best_checkpoint.pth')
        if os.path.exists(best_path):
            try:
                os.remove(best_path)
            except OSError:
                pass
        torch.save(checkpoint, best_path)
        print(f"保存最佳模型 (验证损失: {loss:.6f})")
    
    # 每50轮保存检查点
    if (epoch + 1) % 50 == 0:
        epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, epoch_path)
        print(f"保存周期检查点: epoch {epoch+1}")
        
        # 清理旧检查点
        clean_old_checkpoints(save_dir)

def main():
    parser = argparse.ArgumentParser(description='论文标准CLDM训练 - 优化版')
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
    best_val_loss = float('inf')
    best_fid = float('inf')
    
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
        best_val_loss = checkpoint.get('loss', float('inf'))
        print(f"从epoch {start_epoch}开始训练")
    
    # 训练配置
    training_config = config['training']
    epochs = training_config['epochs']
    save_dir = training_config['save_dir']
    grad_clip_norm = training_config.get('grad_clip_norm')
    
    # 评估间隔
    val_interval = 10  # 每10轮验证
    fid_interval = 20  # 每20轮FID评估
    
    # 训练循环
    print("开始训练...")
    print("优化策略:")
    print("- 每次更新最佳模型时清除旧模型")
    print("- 每50轮保存检查点，自动清理旧检查点")
    print("- 每20轮进行FID评估")
    print("- 不保存生成图片，节省存储空间")
    
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
        
        # 验证和FID评估
        if (epoch + 1) % val_interval == 0:
            val_loss = validate_epoch(model, vqvae, val_loader, device, ema)
            print(f"验证损失: {val_loss:.6f}")
            
            # FID评估
            if (epoch + 1) % fid_interval == 0:
                print("计算FID...")
                try:
                    fid_score = calculate_proper_fid(
                        model, vqvae, val_loader, device, 
                        num_samples=1000, ema=ema  # 增加样本数量
                    )
                    print(f"FID分数: {fid_score:.2f}")
                    
                    # 根据FID保存最佳模型
                    if fid_score < best_fid:
                        best_fid = fid_score
                        save_checkpoint(model, optimizer, scheduler, ema, epoch, val_loss, save_dir, is_best=True)
                        print(f"FID改善! 新最佳FID: {fid_score:.2f}")
                    
                except Exception as e:
                    print(f"FID计算失败: {e}")
            
            # 根据验证损失保存模型
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, ema, epoch, val_loss, save_dir, is_best=True)
        
        # 定期保存和清理
        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, optimizer, scheduler, ema, epoch, train_loss, save_dir)
        
        # 内存清理
        if (epoch + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"最佳FID分数: {best_fid:.2f}")

if __name__ == "__main__":
    main() 
