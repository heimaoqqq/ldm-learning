import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import time
import math
from tqdm import tqdm
from cldm import CLDM
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# 添加上级目录到路径，以便导入VQ-VAE模型
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VAE'))
from adv_vq_vae import AdvVQVAE
from dataset import build_dataloader

# 尝试导入混合精度训练
try:
    from torch.cuda.amp import autocast, GradScaler
    HAS_AMP = True
    print("支持混合精度训练")
except ImportError:
    HAS_AMP = False
    print("不支持混合精度训练，使用常规训练")

# 尝试导入FID计算
HAS_FIDELITY = False
try:
    from torch_fidelity import calculate_metrics
    HAS_FIDELITY = True
    print("成功导入torch-fidelity库，将计算FID指标")
except ImportError:
    print("警告: 无法导入torch-fidelity库，无法计算FID")

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def denormalize(x):
    """反归一化，将[-1,1]转换为[0,1]"""
    return (x * 0.5 + 0.5).clamp(0, 1)

def extract_latent_features(vqvae_model, dataloader, max_samples=None):
    """从数据集中提取潜在特征"""
    latent_features = []
    class_labels = []
    
    with torch.no_grad():
        sample_count = 0
        for images, labels in tqdm(dataloader, desc="提取潜在特征"):
            if max_samples and sample_count >= max_samples:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            # 通过VQ-VAE编码器获取潜在表示
            z = vqvae_model.encoder(images)
            z_q, _, _ = vqvae_model.vq(z)
            
            latent_features.append(z_q.cpu())
            class_labels.append(labels.cpu())
            
            sample_count += images.size(0)
    
    return torch.cat(latent_features, dim=0), torch.cat(class_labels, dim=0)

def sample_and_visualize(cldm_model, vqvae_model, epoch, config, device):
    """采样并可视化生成的图像"""
    cldm_model.eval()
    
    num_samples = config['training']['num_sample_images']
    num_classes = min(8, config['cldm']['num_classes'])  # 限制为8个类别
    samples_per_class = max(1, num_samples // num_classes)
    
    class_labels = []
    for class_id in range(num_classes):
        class_labels.extend([class_id] * samples_per_class)
    class_labels = class_labels[:num_samples]
    class_labels = torch.tensor(class_labels, device=device)
    
    with torch.no_grad():
        # 生成潜在向量
        generated_z = cldm_model.sample(class_labels, device)
        
        # 通过VQ-VAE解码器生成图像
        generated_images = vqvae_model.decoder(generated_z)
        generated_images = denormalize(generated_images)
        
        # 保存生成的图像
        save_dir = config['training']['save_dir']
        os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
        
        if isinstance(epoch, str):
            save_path = os.path.join(save_dir, 'samples', f'{epoch}_samples.png')
        else:
            save_path = os.path.join(save_dir, 'samples', f'epoch_{epoch:03d}_samples.png')
            
        vutils.save_image(
            generated_images,
            save_path,
            nrow=int(math.sqrt(num_samples)),
            padding=2,
            normalize=False
        )
        
        print(f"  保存采样图像: {save_path}")
    
    cldm_model.train()

def calculate_fid_score(cldm_model, vqvae_model, val_loader, sample_size=500):
    """计算FID分数（简化版本）"""
    if not HAS_FIDELITY:
        return float('inf')
    
    try:
        import tempfile
        import uuid
        
        # 创建临时目录
        temp_id = str(uuid.uuid4())[:8]
        temp_dir = os.path.join("temp_fid", temp_id)
        real_dir = os.path.join(temp_dir, "real")
        gen_dir = os.path.join(temp_dir, "generated")
        
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(gen_dir, exist_ok=True)
        
        cldm_model.eval()
        
        with torch.no_grad():
            sample_count = 0
            
            # 保存真实图像和生成图像
            for images, labels in val_loader:
                if sample_count >= sample_size:
                    break
                    
                images = images.to(device)
                labels = labels.to(device)
                batch_size = images.size(0)
                
                # 生成对应的潜在向量
                generated_z = cldm_model.sample(labels, device)
                generated_images = vqvae_model.decoder(generated_z)
                
                # 反归一化
                real_images = denormalize(images)
                gen_images = denormalize(generated_images)
                
                # 保存图像
                for i in range(min(batch_size, sample_size - sample_count)):
                    real_path = os.path.join(real_dir, f"real_{sample_count + i}.png")
                    gen_path = os.path.join(gen_dir, f"gen_{sample_count + i}.png")
                    
                    vutils.save_image(real_images[i], real_path)
                    vutils.save_image(gen_images[i], gen_path)
                
                sample_count += batch_size
        
        # 计算FID
        metrics = calculate_metrics(
            input1=real_dir,
            input2=gen_dir,
            cuda=torch.cuda.is_available(),
            fid=True,
            verbose=False
        )
        
        fid_score = metrics.get('frechet_inception_distance', float('inf'))
        
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        cldm_model.train()
        return fid_score
        
    except Exception as e:
        print(f"FID计算出错: {e}")
        return float('inf')

def main():
    # 加载配置
    config = load_config()
    
    # 设置随机种子
    set_seed(config['system']['seed'])
    
    # 设备配置
    global device
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
    
    # 构建数据加载器
    train_loader, val_loader, train_dataset_len, val_dataset_len = build_dataloader(
        config['dataset']['root_dir'],
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        shuffle_train=True,
        shuffle_val=False,
        val_split=0.3
    )
    
    print(f"数据集信息:")
    print(f"  训练集样本数: {train_dataset_len}")
    print(f"  验证集样本数: {val_dataset_len}")
    print(f"  类别数量: {config['dataset']['num_classes']}")
    
    # 加载预训练的VQ-VAE模型
    print("加载预训练的VQ-VAE模型...")
    vqvae = AdvVQVAE(
        in_channels=config['vqvae']['in_channels'],
        latent_dim=config['vqvae']['latent_dim'],
        num_embeddings=config['vqvae']['num_embeddings'],
        beta=config['vqvae']['beta'],
        decay=config['vqvae']['decay'],
        groups=config['vqvae']['groups']
    ).to(device)
    
    # 加载VQ-VAE权重
    vqvae_path = config['vqvae']['model_path']
    if os.path.exists(vqvae_path):
        vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
        print(f"成功加载VQ-VAE权重: {vqvae_path}")
    else:
        print(f"警告: 找不到VQ-VAE权重文件: {vqvae_path}")
        print("请确保已经训练好VQ-VAE模型")
        return
    
    # 冻结VQ-VAE参数
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False
    
    # 创建CLDM模型
    print("创建CLDM模型...")
    cldm = CLDM(
        latent_dim=config['cldm']['latent_dim'],
        num_classes=config['cldm']['num_classes'],
        num_timesteps=config['cldm']['num_timesteps'],
        beta_start=config['cldm']['beta_start'],
        beta_end=config['cldm']['beta_end'],
        time_emb_dim=config['cldm']['time_emb_dim'],
        class_emb_dim=config['cldm']['class_emb_dim'],
        base_channels=config['cldm']['base_channels'],
        groups=config['cldm']['groups']
    ).to(device)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in cldm.parameters())
    trainable_params = sum(p.numel() for p in cldm.parameters() if p.requires_grad)
    print(f"CLDM模型参数量:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  模型大小约: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # 创建优化器
    optimizer = optim.AdamW(
        cldm.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    use_scheduler = config['training']['use_scheduler']
    if use_scheduler:
        scheduler_type = config['training']['scheduler_type']
        warmup_epochs = config['training']['warmup_epochs']
        total_epochs = config['training']['epochs']
        
        if scheduler_type == "cosine":
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return float(epoch) / float(max(1, warmup_epochs))
                else:
                    progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif scheduler_type == "linear":
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return float(epoch) / float(max(1, warmup_epochs))
                else:
                    return 1.0 - float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        print(f"使用学习率调度器: {scheduler_type}")
    else:
        scheduler = None
        print("不使用学习率调度器")
    
    # 混合精度训练
    use_amp = config['system']['mixed_precision'] and HAS_AMP
    if use_amp:
        scaler = GradScaler()
        print("启用混合精度训练")
    else:
        scaler = None
        print("使用常规精度训练")
    
    # 梯度裁剪
    grad_clip_norm = config['training']['grad_clip_norm']
    
    # 提取训练集的潜在特征（用于训练CLDM）
    print("提取训练集潜在特征...")
    train_latents, train_labels = extract_latent_features(vqvae, train_loader)
    print(f"提取的潜在特征形状: {train_latents.shape}")
    
    # 创建潜在特征的数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    latent_dataset = TensorDataset(train_latents, train_labels)
    latent_loader = DataLoader(
        latent_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # 训练循环
    print(f"开始训练CLDM，总共{config['training']['epochs']}轮...")
    
    best_val_loss = float('inf')
    best_fid = float('inf')
    
    for epoch in range(config['training']['epochs']):
        cldm.train()
        running_loss = 0.0
        num_batches = 0
        
        # 训练阶段
        for batch_idx, (z_0, class_labels) in enumerate(tqdm(latent_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")):
            z_0 = z_0.to(device)
            class_labels = class_labels.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    loss = cldm(z_0, class_labels)
                
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(cldm.parameters(), grad_clip_norm)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = cldm(z_0, class_labels)
                loss.backward()
                
                # 梯度裁剪
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(cldm.parameters(), grad_clip_norm)
                
                optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
        
        # 计算平均训练损失
        avg_train_loss = running_loss / num_batches
        
        # 验证阶段
        if (epoch + 1) % config['training']['val_interval'] == 0:
            cldm.eval()
            val_running_loss = 0.0
            val_num_batches = 0
            
            # 提取验证集潜在特征
            val_latents, val_labels = extract_latent_features(vqvae, val_loader, max_samples=500)
            val_latent_dataset = TensorDataset(val_latents, val_labels)
            val_latent_loader = DataLoader(val_latent_dataset, batch_size=config['dataset']['batch_size'], shuffle=False)
            
            with torch.no_grad():
                for z_0, class_labels in val_latent_loader:
                    z_0 = z_0.to(device)
                    class_labels = class_labels.to(device)
                    
                    if use_amp:
                        with autocast():
                            val_loss = cldm(z_0, class_labels)
                    else:
                        val_loss = cldm(z_0, class_labels)
                    
                    val_running_loss += val_loss.item()
                    val_num_batches += 1
            
            avg_val_loss = val_running_loss / val_num_batches
            
            print(f"Epoch {epoch+1}:")
            print(f"  训练损失: {avg_train_loss:.6f}")
            print(f"  验证损失: {avg_val_loss:.6f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(cldm.state_dict(), os.path.join(save_dir, 'cldm_best_loss.pth'))
                print(f"  ✓ 保存最佳验证损失模型: {best_val_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}: 训练损失: {avg_train_loss:.6f}")
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  当前学习率: {current_lr:.6f}")
        
        # 采样可视化
        if (epoch + 1) % config['training']['sample_interval'] == 0:
            print(f"  生成采样图像...")
            sample_and_visualize(cldm, vqvae, epoch + 1, config, device)
        
        # FID评估
        if (epoch + 1) % config['training']['fid_eval_freq'] == 0 and HAS_FIDELITY:
            print(f"  计算FID分数...")
            fid_score = calculate_fid_score(cldm, vqvae, val_loader, 500)
            print(f"  FID分数: {fid_score:.4f}")
            
            if fid_score < best_fid:
                best_fid = fid_score
                torch.save(cldm.state_dict(), os.path.join(save_dir, 'cldm_best_fid.pth'))
                print(f"  ✓ 保存最佳FID模型: {best_fid:.4f}")
        
        # 保存checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': cldm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_fid': best_fid,
                'config': config
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint, os.path.join(save_dir, f'cldm_epoch_{epoch+1}.pth'))
            print(f"  保存checkpoint: cldm_epoch_{epoch+1}.pth")
    
    print("CLDM训练完成！")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    if HAS_FIDELITY:
        print(f"最佳FID分数: {best_fid:.4f}")
    
    # 最终采样
    print("生成最终采样图像...")
    sample_and_visualize(cldm, vqvae, "final", config, device)

if __name__ == "__main__":
    main() 