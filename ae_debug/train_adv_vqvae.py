import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
from tqdm import tqdm
from adv_vq_vae import AdvVQVAE
from dataset import build_dataloader
from perceptual_loss import VGGPerceptualLoss
import torchvision.utils as vutils
try:
    from torch_fidelity import calculate_metrics
    HAS_FIDELITY = True
except ImportError:
    print("警告: torch-fidelity未安装，无法计算FID。请运行 'pip install torch-fidelity' 安装。")
    HAS_FIDELITY = False

# 启用异常检测，帮助识别梯度问题
torch.autograd.set_detect_anomaly(True)

# 加载配置
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 指定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# FID相关路径和设置
fid_eval_freq = config['training'].get('fid_eval_freq', 5)  # 默认每5个epoch计算一次FID
fid_sample_size = config['training'].get('fid_sample_size', 1000)  # 用于计算FID的样本数量
fid_batch_size = config['training'].get('fid_batch_size', 32)  # FID计算的批量大小

# FID临时图像目录
temp_dir = os.path.join("ae_debug", "temp_fid_images")
temp_real_dir = os.path.join(temp_dir, "real")
temp_gen_dir = os.path.join(temp_dir, "generated")

# 构建数据加载器
train_loader, test_loader, train_dataset_len, test_dataset_len = build_dataloader(
    config['dataset']['root_dir'],
    batch_size=config['dataset']['batch_size'],
    num_workers=0, # 建议在Windows上调试时设为0，或者根据CPU核心数调整
    shuffle_train=True,
    shuffle_test=False, # 测试集通常不需要打乱
    test_split=0.3 # 30% 作为测试集
)
dataset_len = train_dataset_len # 用于旧的平均损失计算基数，现在主要用 train_dataset_len

# 初始化模型
model = AdvVQVAE(
    in_channels=config['model']['in_channels'],
    latent_dim=config['model']['latent_dim'],
    num_embeddings=config['model']['num_embeddings'],
    beta=config['model']['beta'],
    groups=config['model']['groups'],
    disc_ndf=64  # 判别器特征数量
).to(device)

# 创建优化器
gen_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.vq.parameters())
gen_optimizer = optim.Adam(
    gen_params,
    lr=config['training']['lr'],
    weight_decay=config['training']['weight_decay']
)
disc_optimizer = optim.Adam(
    model.discriminator.parameters(),
    lr=config['training']['lr'] * 0.5,  # 判别器学习率通常略小
    weight_decay=config['training']['weight_decay']
)

# 感知损失
perceptual_loss_fn = VGGPerceptualLoss().to(device) # 重命名以区分变量和类实例
perceptual_weight = config['training'].get('perceptual_weight', 0.1) # 从配置读取或默认

# 用于反归一化
def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

# 创建保存目录
save_dir = os.path.join("ae_debug", "adv_vqvae_models")
os.makedirs(save_dir, exist_ok=True)

# FID计算函数
def calculate_fid(model, test_loader, temp_real_dir, temp_gen_dir, sample_size=1000):
    if not HAS_FIDELITY:
        print("  无法计算FID: torch-fidelity未安装")
        return float('inf')
    
    # 清空或创建临时目录
    for dir_path in [temp_real_dir, temp_gen_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    
    model.eval()
    with torch.no_grad():
        sample_count = 0
        for test_images, _ in tqdm(test_loader, desc="生成FID图像样本"):
            test_images = test_images.to(device)
            batch_size = test_images.size(0)
            
            # 通过模型获取重建图像
            test_z = model.encoder(test_images)
            test_z_q, _ = model.vq(test_z)
            test_recons = model.decoder(test_z_q)
            
            # 反归一化
            real_images = denorm(test_images)
            gen_images = denorm(test_recons)
            
            # 保存图像
            for i in range(batch_size):
                if sample_count >= sample_size:
                    break
                
                # 保存真实图像
                vutils.save_image(
                    real_images[i], 
                    os.path.join(temp_real_dir, f"{sample_count}.png")
                )
                
                # 保存生成图像
                vutils.save_image(
                    gen_images[i], 
                    os.path.join(temp_gen_dir, f"{sample_count}.png")
                )
                
                sample_count += 1
                
            if sample_count >= sample_size:
                break
    
    print(f"  计算FID中 (基于{sample_count}个图像样本)...")
    
    # 计算FID
    try:
        metrics = calculate_metrics(
            input1=temp_real_dir, 
            input2=temp_gen_dir,
            cuda=torch.cuda.is_available(), 
            fid=True
        )
        fid_score = metrics['fid']
        print(f"  FID评分: {fid_score:.4f}")
        return fid_score
    except Exception as e:
        print(f"  计算FID时出错: {e}")
        return float('inf')

# 训练循环
best_test_loss = float('inf') # 使用测试集损失来保存最佳模型
best_fid = float('inf')       # 使用FID评分来保存最佳模型
epochs = config['training']['epochs']
disc_train_interval = config['training'].get('disc_train_interval', 5) # 判别器训练间隔

print(f"开始训练，总共{epochs}轮...")
print(f"训练集样本数: {train_dataset_len}, 测试集样本数: {test_dataset_len}")
print(f"FID评估频率: 每{fid_eval_freq}个epoch计算一次")


for epoch in range(epochs):
    model.train() # 设置为训练模式
    running_rec_loss = 0.0
    running_vq_loss = 0.0
    running_disc_loss = 0.0
    running_gen_adv_loss = 0.0 # 用于生成器的对抗损失
    running_perc_loss = 0.0
    
    disc_batches_trained = 0 # 记录判别器训练了多少个批次

    for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
        images = images.to(device)
        batch_size = images.size(0)
        
        train_disc = (batch_idx % disc_train_interval == 0)

        #----------------------------------------------
        # 1. 首先训练判别器（如果需要）
        #----------------------------------------------
        if train_disc:
            model.discriminator.requires_grad_(True) # 确保判别器梯度开启
            disc_optimizer.zero_grad()
            
            with torch.no_grad(): # 生成器部分不计算梯度
                z = model.encoder(images)
                z_q, _ = model.vq(z) 
                fake_images = model.decoder(z_q)
            
            real_preds = model.discriminator(images)
            fake_preds = model.discriminator(fake_images.detach()) # .detach() 很重要
            
            d_real_loss = nn.BCEWithLogitsLoss()(real_preds, torch.ones_like(real_preds))
            d_fake_loss = nn.BCEWithLogitsLoss()(fake_preds, torch.zeros_like(fake_preds))
            d_loss = (d_real_loss + d_fake_loss) * 0.5 # 论文中常见的做法是乘以0.5
            
            d_loss.backward()
            disc_optimizer.step()
            running_disc_loss += d_loss.item() * batch_size # 乘以 batch_size 以便后续正确平均
            disc_batches_trained += 1


        #----------------------------------------------
        # 2. 然后训练生成器（编码器+解码器+VQ层）
        #----------------------------------------------
        model.discriminator.requires_grad_(False) # 训练生成器时冻结判别器参数
        gen_optimizer.zero_grad()
        
        z = model.encoder(images)
        z_q, vq_loss_commit = model.vq(z) # vq_loss_commit 是 commitment_loss
        recons = model.decoder(z_q)
        
        rec_loss = nn.MSELoss()(recons, images)
        p_loss = perceptual_loss_fn(denorm(recons), denorm(images))
        
        # 对抗损失（让判别器认为生成图像是真实的）
        fake_preds_for_gen = model.discriminator(recons)
        g_adv_loss = nn.BCEWithLogitsLoss()(fake_preds_for_gen, torch.ones_like(fake_preds_for_gen))
        
        # 总生成器损失
        # 在 AdvVQVAE 类的 calculate_losses 中 gen_total_loss = rec_weight * rec_loss + vq_loss - gen_loss
        # 这里我们遵循论文中的结构，通常 GAN 的生成器损失是 -log(D(G(z))) 或类似的鼓励 D 输出高分的项
        # VQ-VAE论文中 L_VQ = L_rec + L_vq + beta * L_commit
        # 这里的 vq_loss_commit 就是 commitment loss
        # 而 VQ-GAN (Taming Transformers) 中, G loss = L_rec + lambda * L_perc + L_vq + L_GAN
        
        # 修改 gen_total_loss 组成，参考 VQ-GAN 论文和常见的 GAN 结构
        # vq_loss (commitment loss) 已经在 AdvVQVAE.vq 返回
        # L_G = lambda_rec * L_rec + lambda_vq * (log_perplexity + commitment_loss) + lambda_adv * L_adv
        # 这里简化： L_G = L_rec + commitment_loss + weight_perc * L_perc + weight_adv * L_GAN
        
        # 从 config 中获取权重
        rec_weight = config['training'].get('rec_weight', 1.0)
        vq_commit_weight = config['training'].get('vq_commit_weight', 1.0) # beta in VQ
        # perceptual_weight 已经在上面定义
        gen_adv_weight = config['training'].get('gen_adv_weight', 0.1)


        g_total = (rec_weight * rec_loss + 
                   vq_commit_weight * vq_loss_commit + 
                   perceptual_weight * p_loss + 
                   gen_adv_weight * g_adv_loss)
        
        g_total.backward()
        gen_optimizer.step()
        
        running_rec_loss += rec_loss.item() * batch_size
        running_vq_loss += vq_loss_commit.item() * batch_size # 累加 commitment loss
        running_gen_adv_loss += g_adv_loss.item() * batch_size
        running_perc_loss += p_loss.item() * batch_size
    
    # 计算训练集上的平均损失
    epoch_rec_loss = running_rec_loss / train_dataset_len
    epoch_vq_loss = running_vq_loss / train_dataset_len # 这是 commitment loss
    epoch_gen_adv_loss = running_gen_adv_loss / train_dataset_len
    epoch_perc_loss = running_perc_loss / train_dataset_len
    
    if disc_batches_trained > 0:
        epoch_disc_loss = running_disc_loss / (disc_batches_trained * config['dataset']['batch_size'])
    else:
        epoch_disc_loss = 0.0

    print(f"Epoch {epoch+1} Training Losses:")
    print(f"  Rec Loss: {epoch_rec_loss:.6f}, VQ Commit Loss: {epoch_vq_loss:.6f}")
    print(f"  Gen Adversarial Loss: {epoch_gen_adv_loss:.6f}, Perceptual Loss: {epoch_perc_loss:.6f}")
    print(f"  Disc Loss: {epoch_disc_loss:.6f} (trained on {disc_batches_trained} batches)")

    # --- 在测试集上评估 ---
    model.eval() # 设置为评估模式
    test_running_rec_loss = 0.0
    test_running_vq_loss = 0.0
    test_running_perc_loss = 0.0
    
    with torch.no_grad():
        for test_images, _ in tqdm(test_loader, desc=f"Epoch {epoch+1} Testing", leave=False):
            test_images = test_images.to(device)
            
            test_z = model.encoder(test_images)
            test_z_q, test_vq_commit_loss = model.vq(test_z)
            test_recons = model.decoder(test_z_q)
            
            test_rec = nn.MSELoss()(test_recons, test_images)
            test_perceptual = perceptual_loss_fn(denorm(test_recons), denorm(test_images))
            
            test_running_rec_loss += test_rec.item() * test_images.size(0)
            test_running_vq_loss += test_vq_commit_loss.item() * test_images.size(0)
            test_running_perc_loss += test_perceptual.item() * test_images.size(0)
            
    epoch_test_rec_loss = test_running_rec_loss / test_dataset_len
    epoch_test_vq_loss = test_running_vq_loss / test_dataset_len
    epoch_test_perc_loss = test_running_perc_loss / test_dataset_len
    
    # 计算用于比较和保存模型的测试集总损失
    # 这个组合可以根据实际效果调整，这里简单地将主要的生成器相关损失加起来
    current_test_loss = (rec_weight * epoch_test_rec_loss + 
                         vq_commit_weight * epoch_test_vq_loss + 
                         perceptual_weight * epoch_test_perc_loss)

    print(f"Epoch {epoch+1} Testing Losses:")
    print(f"  Test Rec Loss: {epoch_test_rec_loss:.6f}, Test VQ Commit Loss: {epoch_test_vq_loss:.6f}")
    print(f"  Test Perceptual Loss: {epoch_test_perc_loss:.6f}")
    print(f"  Current Combined Test Loss (for saving): {current_test_loss:.6f}")

    # 保存基于测试损失的最佳模型
    if current_test_loss < best_test_loss:
        best_test_loss = current_test_loss
        torch.save(model.state_dict(), os.path.join(save_dir, 'adv_vqvae_best_loss.pth'))
        print(f"  保存基于损失的最佳模型，Test Loss: {best_test_loss:.6f}")
    
    # 每fid_eval_freq个epoch计算一次FID
    if (epoch + 1) % fid_eval_freq == 0 and HAS_FIDELITY:
        fid_score = calculate_fid(
            model, 
            test_loader, 
            temp_real_dir, 
            temp_gen_dir, 
            sample_size=fid_sample_size
        )
        
        # 保存基于FID评分的最佳模型
        if fid_score < best_fid:
            best_fid = fid_score
            torch.save(model.state_dict(), os.path.join(save_dir, 'adv_vqvae_best_fid.pth'))
            print(f"  保存基于FID的最佳模型，FID Score: {best_fid:.6f}")
    
    # 每20轮保存一次checkpoint
    if (epoch + 1) % config['training'].get('save_interval', 20) == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'gen_optimizer_state_dict': gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': disc_optimizer.state_dict(),
            'best_test_loss': best_test_loss,
            'best_fid': best_fid if 'best_fid' in locals() else float('inf'),
        }, os.path.join(save_dir, f'adv_vqvae_epoch_{epoch+1}.pth'))
        print(f"  保存checkpoint: adv_vqvae_epoch_{epoch+1}.pth")

print("训练完成！")
# 删除临时文件目录
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir) 