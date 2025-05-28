import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shutil
import time
from tqdm import tqdm
from adv_vq_vae import AdvVQVAE
from dataset import build_dataloader
from perceptual_loss import VGGPerceptualLoss
import torchvision.utils as vutils
import uuid
import math

# --- torch-fidelity Setup ---
HAS_FIDELITY = False
try:
    from torch_fidelity import calculate_metrics as original_calculate_metrics
    def calculate_metrics_wrapped(*args, **kwargs):
        import io
        import sys
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            result = original_calculate_metrics(*args, **kwargs)
        if result and 'frechet_inception_distance' in result:
            # Minimal print from wrapper
            pass # print(f"FID from calculate_metrics_wrapped: {result['frechet_inception_distance']:.4f}")
        return result
    
    HAS_FIDELITY = True
    print("✅ 成功导入torch-fidelity库，将计算FID指标")
except ImportError as e:
    print(f"❌ 警告: 无法导入torch-fidelity库: {e}. FID计算将被跳过。")
    print("   请运行 'pip install torch-fidelity' 安装。")

# --- Configuration Loading ---
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("错误: config.yaml 未找到。请确保配置文件存在于当前目录。")
    exit(1)
except Exception as e:
    print(f"加载config.yaml时出错: {e}")
    exit(1)

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# --- FID Configuration ---
fid_eval_freq = config['training'].get('fid_eval_freq', 5)
fid_sample_size = config['training'].get('fid_sample_size', 1000)
# fid_batch_size = config['training'].get('fid_batch_size', 32) # Not directly used in calculate_fid_simple

temp_base_dir = os.path.join("temp_fid_images_vae") # Unique name for VAE FID temps
os.makedirs(temp_base_dir, exist_ok=True)

def get_temp_fid_dirs():
    unique_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join(temp_base_dir, f"run_{unique_id}")
    temp_real_dir = os.path.join(temp_dir, "real")
    temp_gen_dir = os.path.join(temp_dir, "generated")
    os.makedirs(temp_real_dir, exist_ok=True) # Ensure subdirectories are also created
    os.makedirs(temp_gen_dir, exist_ok=True)
    return temp_dir, temp_real_dir, temp_gen_dir

# --- DataLoaders ---
print("加载数据集...")
train_loader, val_loader, train_dataset_len, val_dataset_len = build_dataloader(
    config['dataset']['root_dir'],
    batch_size=config['dataset']['batch_size'],
    num_workers=config['dataset'].get('num_workers', 0),
    shuffle_train=True,
    shuffle_val=False,
    val_split=0.3 # Example, can be configured
)
print(f"训练集样本数: {train_dataset_len}, 验证集样本数: {val_dataset_len}")

# --- Model Initialization ---
print("初始化AdvVQVAE模型...")
model = AdvVQVAE(
    in_channels=config['model']['in_channels'],
    latent_dim=config['model']['latent_dim'],
    num_embeddings=config['model']['num_embeddings'],
    beta=config['model']['beta'],
    decay=config['model'].get('vq_ema_decay', 0.99),
    groups=config['model']['groups'],
    disc_ndf=64 # Example, can be configured
).to(device)

if hasattr(model, 'vq') and hasattr(model.vq, 'reset_cumulative_usage'):
    print("  模型支持重置累积码本使用率计数器。")
else:
    print("  ⚠️ 警告: 模型VQ层不支持 reset_cumulative_usage。码本利用率监控可能不按epoch重置。")

# --- Optimizers ---
print("设置优化器...")
gen_params = list(model.encoder.parameters()) + \
             list(model.decoder.parameters()) + \
             list(model.vq.parameters())
gen_optimizer = optim.Adam(
    gen_params,
    lr=config['training']['lr'],
    betas=(0.9, 0.999), # Common beta values
    weight_decay=config['training'].get('weight_decay', 0.0)
)
disc_optimizer = optim.Adam(
    model.discriminator.parameters(),
    lr=config['training']['lr'] * 0.5, # Often discriminator LR is same or slightly lower
    betas=(0.9, 0.999),
    weight_decay=config['training'].get('weight_decay', 0.0)
)

# --- Loss Functions and Weights ---
perceptual_loss_fn = VGGPerceptualLoss().to(device)
rec_weight = config['training'].get('rec_weight', 1.0)
# Note: vq_commit_weight (beta) is handled inside model.vq. Beta is passed during model init.
# The config key 'vq_commit_weight' might be redundant if beta in model config is the true source.
# We will rely on the beta passed to model.vq for the commitment loss scaling.
perceptual_weight = config['training'].get('perceptual_weight', 0.1)
gen_adv_weight = config['training'].get('gen_adv_weight', 0.1)

print(f"损失权重: Rec={rec_weight}, Perceptual={perceptual_weight}, GenAdv={gen_adv_weight}, VQ_Beta(Commitment)={model.vq.beta}")

# --- Utility Functions ---
def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

save_dir = config['training']['save_dir']
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, 'reconstructions'), exist_ok=True)


# --- FID Calculation Function ---
def calculate_fid_for_training(model_to_eval, current_val_loader, num_samples_for_fid):
    if not HAS_FIDELITY:
        print("   FID计算跳过: torch-fidelity 未加载.")
        return float('inf')

    temp_fid_run_dir, temp_real_dir_fid, temp_gen_dir_fid = get_temp_fid_dirs()
    fid_score_value = float('inf')
    
    try:
        model_to_eval.eval() # Set model to evaluation mode
        images_saved_count = 0
        with torch.no_grad():
            for real_imgs_batch, _ in tqdm(current_val_loader, desc="  生成FID评估图像", leave=False, ncols=80):
                if images_saved_count >= num_samples_for_fid:
                    break
                real_imgs_batch = real_imgs_batch.to(device)
                
                # VQ-VAE forward returns: x_recon, commitment_loss, codebook_loss, batch_usage, cumulative_usage
                # For FID, we only need x_recon
                recon_imgs_batch, _, _, _, _ = model_to_eval(real_imgs_batch)

                denorm_real_batch = denorm(real_imgs_batch)
                denorm_recon_batch = denorm(recon_imgs_batch)

                for i in range(denorm_real_batch.size(0)):
                    if images_saved_count < num_samples_for_fid:
                        vutils.save_image(denorm_real_batch[i], os.path.join(temp_real_dir_fid, f"img_{images_saved_count}_real.png"))
                        vutils.save_image(denorm_recon_batch[i], os.path.join(temp_gen_dir_fid, f"img_{images_saved_count}_gen.png"))
                        images_saved_count += 1
                    else:
                        break
        
        if images_saved_count < min(10, num_samples_for_fid): # Need at least a few samples
            print(f"  ⚠️ FID警告: 样本太少 ({images_saved_count}), 结果可能不准。")
            return float('inf')

        print(f"  计算FID (基于 {images_saved_count} 真实/重建图像对)...")
        metrics = calculate_metrics_wrapped(
            input1=temp_real_dir_fid,
            input2=temp_gen_dir_fid,
            cuda=torch.cuda.is_available(),
            fid=True,
            verbose=False # Wrapper handles minimal output
        )
        fid_score_value = metrics.get('frechet_inception_distance', float('inf'))
        if fid_score_value != float('inf'):
             print(f"  FID计算完成: {fid_score_value:.4f}")
        else:
             print("  FID值未从torch-fidelity返回。")

    except Exception as e_fid:
        print(f"  FID 计算时出错: {e_fid}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(temp_fid_run_dir):
            try:
                shutil.rmtree(temp_fid_run_dir)
            except OSError as e_clean:
                print(f"  清理临时FID目录失败 {temp_fid_run_dir}: {e_clean}")
    return fid_score_value

# --- Learning Rate Scheduler ---
use_scheduler = config['training'].get('use_scheduler', False)
warmup_epochs_cfg = config['training'].get('warmup_epochs', 0)
epochs_cfg = config['training']['epochs']

gen_scheduler, disc_scheduler = None, None
if use_scheduler:
    print(f"使用学习率调度器: 预热 {warmup_epochs_cfg} epochs, 总共 {epochs_cfg} epochs.")
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs_cfg:
            return float(current_epoch + 1) / float(max(1, warmup_epochs_cfg)) # +1 for 1-based epoch
        progress = float(current_epoch - warmup_epochs_cfg) / float(max(1, epochs_cfg - warmup_epochs_cfg))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress))) # Cosine decay half-period

    gen_scheduler = optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda)
    disc_scheduler = optim.lr_scheduler.LambdaLR(disc_optimizer, lr_lambda)

# --- Checkpoint Loading ---
start_epoch = 0
best_val_loss = float('inf')
best_fid = float('inf')
checkpoint_path = os.path.join(save_dir, 'adv_vqvae_latest.pth')

if os.path.exists(checkpoint_path):
    try:
        print(f"尝试从checkpoint加载: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        if use_scheduler and gen_scheduler and disc_scheduler and 'gen_scheduler_state_dict' in checkpoint:
            gen_scheduler.load_state_dict(checkpoint['gen_scheduler_state_dict'])
            disc_scheduler.load_state_dict(checkpoint['disc_scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1 # Resume from next epoch
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_fid = checkpoint.get('best_fid', float('inf'))
        print(f"  ✅ 成功从epoch {start_epoch-1}恢复。下一个epoch: {start_epoch}")
        print(f"     之前最佳验证损失: {best_val_loss:.4f}, 最佳FID: {best_fid:.4f}")
    except Exception as e_load:
        print(f"  ❌ 加载checkpoint失败: {e_load}。将从头开始训练。")
        start_epoch = 0
        best_val_loss = float('inf')
        best_fid = float('inf')

# --- Main Training Loop ---
epochs = config['training']['epochs']
disc_train_interval = config['training'].get('disc_train_interval', 1)

print(f"\n🚀 开始训练，从epoch {start_epoch + 1} 到 {epochs} 🚀")

for epoch in range(start_epoch, epochs):
    epoch_start_time = time.time()
    print(f"\n--- Epoch {epoch+1}/{epochs} ---")

    if hasattr(model.vq, 'reset_cumulative_usage'):
        model.vq.reset_cumulative_usage()
        if epoch == start_epoch : print("  已重置累积码本使用计数器。")

    model.train()
    
    total_train_gen_loss = 0.0
    total_train_disc_loss = 0.0
    total_train_rec_loss = 0.0
    total_train_perceptual_loss = 0.0
    total_train_vq_commit_loss = 0.0
    total_train_vq_codebook_loss = 0.0
    epoch_final_cumulative_usage = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", ncols=100, leave=False)
    
    for batch_idx, (real_images, _) in enumerate(progress_bar):
        real_images = real_images.to(device)

        # --- 1. Train Discriminator ---
        disc_optimizer.zero_grad()
        with torch.no_grad():
            # model.forward returns: x_recon, commit_loss, codebook_loss, batch_usage, cumulative_usage
            recon_images_detached, _, _, _, _ = model(real_images)
        
        pred_real = model.discriminator(real_images)
        loss_disc_real = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
        
        pred_fake = model.discriminator(recon_images_detached.detach()) # Detach again for safety
        loss_disc_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
        
        loss_disc = (loss_disc_real + loss_disc_fake) * 0.5
        loss_disc.backward()
        disc_optimizer.step()
        total_train_disc_loss += loss_disc.item()

        # --- 2. Train Generator (Encoder, VQ, Decoder) ---
        gen_optimizer.zero_grad()
        
        recon_images, commit_loss, codebook_loss, batch_usage, cumulative_usage = model(real_images)
        epoch_final_cumulative_usage = cumulative_usage # Keep updating, last batch's cumulative is epoch's approx.
        
        loss_reconstruction = F.mse_loss(recon_images, real_images) * rec_weight
        loss_perceptual = perceptual_loss_fn(recon_images, real_images) * perceptual_weight # perceptual_loss_fn expects inputs in [-1,1] or [0,1] depending on its normalization. Ours are [-1,1]
        
        # VQ losses (commitment_loss already has beta)
        loss_vq = codebook_loss + commit_loss
        
        pred_gen_fake = model.discriminator(recon_images)
        loss_gen_adv = F.binary_cross_entropy_with_logits(pred_gen_fake, torch.ones_like(pred_gen_fake)) * gen_adv_weight
        
        loss_gen_total = loss_reconstruction + loss_perceptual + loss_vq + loss_gen_adv
        loss_gen_total.backward()
        gen_optimizer.step()

        total_train_gen_loss += loss_gen_total.item()
        total_train_rec_loss += loss_reconstruction.item() # Already weighted
        total_train_perceptual_loss += loss_perceptual.item() # Already weighted
        total_train_vq_commit_loss += commit_loss.item() # Already has beta
        total_train_vq_codebook_loss += codebook_loss.item()

        progress_bar.set_postfix({
            "G_Loss": f"{loss_gen_total.item():.3f}", "D_Loss": f"{loss_disc.item():.3f}",
            "Rec": f"{loss_reconstruction.item():.3f}", "Perc": f"{loss_perceptual.item():.3f}",
            "VQ": f"{loss_vq.item():.3f}", "Adv": f"{loss_gen_adv.item():.3f}",
            "B_Usage%": f"{batch_usage*100:.1f}",
            "LR_G": f"{gen_optimizer.param_groups[0]['lr']:.2e}"
        })

    avg_train_gen_loss = total_train_gen_loss / len(train_loader)
    avg_train_disc_loss = total_train_disc_loss / len(train_loader)
    avg_train_rec_loss = total_train_rec_loss / len(train_loader)
    avg_train_perceptual_loss = total_train_perceptual_loss / len(train_loader)
    avg_train_vq_commit_loss = total_train_vq_commit_loss / len(train_loader)
    avg_train_vq_codebook_loss = total_train_vq_codebook_loss / len(train_loader)

    if use_scheduler and gen_scheduler and disc_scheduler:
        gen_scheduler.step()
        disc_scheduler.step()

    print(f"  📊 Epoch {epoch+1} Training Summary:")
    print(f"     Avg Gen Loss: {avg_train_gen_loss:.4f}, Avg Disc Loss: {avg_train_disc_loss:.4f}")
    print(f"     Avg Rec Loss: {avg_train_rec_loss:.4f}, Avg Perceptual Loss: {avg_train_perceptual_loss:.4f}")
    print(f"     Avg VQ Commit: {avg_train_vq_commit_loss:.4f}, Avg VQ Codebook: {avg_train_vq_codebook_loss:.4f}")
    if epoch_final_cumulative_usage > 0:
        print(f"     📈 Cumulative Codebook Usage (end of epoch): {epoch_final_cumulative_usage*100:.2f}%")
    
    # --- Validation Step ---
    model.eval()
    total_val_rec_loss = 0.0
    total_val_perceptual_loss = 0.0
    total_val_vq_commit_loss = 0.0
    total_val_vq_codebook_loss = 0.0
    val_epoch_final_cumulative_usage = 0.0
    
    print("  🔎 Starting validation...")
    with torch.no_grad():
        for val_batch_images, _ in tqdm(val_loader, desc="  Validation", ncols=80, leave=False):
            val_batch_images = val_batch_images.to(device)
            
            val_recon_images, val_commit_loss, val_codebook_loss, _, val_cumulative_usage = model(val_batch_images)
            val_epoch_final_cumulative_usage = val_cumulative_usage # Take from last validation batch

            total_val_rec_loss += F.mse_loss(val_recon_images, val_batch_images).item() * rec_weight
            total_val_perceptual_loss += perceptual_loss_fn(val_recon_images, val_batch_images).item() * perceptual_weight
            total_val_vq_commit_loss += val_commit_loss.item()
            total_val_vq_codebook_loss += val_codebook_loss.item()
            
    avg_val_rec_loss = total_val_rec_loss / len(val_loader)
    avg_val_perceptual_loss = total_val_perceptual_loss / len(val_loader)
    avg_val_vq_commit_loss = total_val_vq_commit_loss / len(val_loader)
    avg_val_vq_codebook_loss = total_val_vq_codebook_loss / len(val_loader)
    
    # Combined validation loss for checkpointing (can be adjusted)
    current_combined_val_loss = avg_val_rec_loss + avg_val_perceptual_loss + avg_val_vq_commit_loss + avg_val_vq_codebook_loss

    print(f"  📉 Epoch {epoch+1} Validation Summary - Combined Loss: {current_combined_val_loss:.4f}")
    print(f"     Avg Rec: {avg_val_rec_loss:.4f}, Avg Perc: {avg_val_perceptual_loss:.4f}")
    print(f"     Avg VQ Commit: {avg_val_vq_commit_loss:.4f}, Avg VQ Codebook: {avg_val_vq_codebook_loss:.4f}")
    if val_epoch_final_cumulative_usage > 0:
         print(f"     📈 Cumulative Codebook Usage (from last val batch): {val_epoch_final_cumulative_usage*100:.2f}%")

    # Save best model based on validation loss
    if current_combined_val_loss < best_val_loss:
        best_val_loss = current_combined_val_loss
        torch.save(model.state_dict(), os.path.join(save_dir, 'adv_vqvae_best_loss.pth'))
        print(f"    ⭐ New best validation loss: {best_val_loss:.4f}. Model (best_loss) saved.")

    # --- FID Evaluation ---
    if (epoch + 1) % fid_eval_freq == 0:
        print(f"\n  🏆 Epoch {epoch+1}: Calculating FID score...")
        current_fid = calculate_fid_for_training(model, val_loader, fid_sample_size)
        if current_fid < best_fid:
            best_fid = current_fid
            torch.save(model.state_dict(), os.path.join(save_dir, 'adv_vqvae_best_fid.pth'))
            print(f"    ⭐ New best FID: {best_fid:.4f}! Model (best_fid) saved.")
        elif current_fid != float('inf'):
            print(f"    FID {current_fid:.4f} (Best: {best_fid:.4f})")


    # --- Save Latest Checkpoint ---
    checkpoint_data = {
        'epoch': epoch, # Save current epoch, so resume starts from epoch + 1
        'model_state_dict': model.state_dict(),
        'gen_optimizer_state_dict': gen_optimizer.state_dict(),
        'disc_optimizer_state_dict': disc_optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'best_fid': best_fid
    }
    if use_scheduler and gen_scheduler and disc_scheduler:
        checkpoint_data['gen_scheduler_state_dict'] = gen_scheduler.state_dict()
        checkpoint_data['disc_scheduler_state_dict'] = disc_scheduler.state_dict()
    
    torch.save(checkpoint_data, os.path.join(save_dir, 'adv_vqvae_latest.pth'))

    # --- Periodic Model Save (e.g., every N epochs) ---
    if (epoch + 1) % config['training'].get('save_interval', 20) == 0:
        torch.save(model.state_dict(), os.path.join(save_dir, f'adv_vqvae_epoch_{epoch+1}.pth'))
        print(f"  💾 Checkpoint saved for epoch {epoch+1}.")
        
    # --- Visualize Reconstructions ---
    if (epoch + 1) % 10 == 0: # Example: every 10 epochs
        model.eval()
        with torch.no_grad():
            vis_images_batch, _ = next(iter(val_loader)) # Get one batch from validation
            vis_images_batch = vis_images_batch.to(device)
            vis_recon_batch, _, _, _, _ = model(vis_images_batch)
            
            # Create a comparison image (e.g., first 8 images)
            comparison = torch.cat([vis_images_batch[:8], vis_recon_batch[:8]])
            vutils.save_image(denorm(comparison), 
                              os.path.join(save_dir, 'reconstructions', f'reconstruction_epoch_{epoch+1}.png'),
                              nrow=8) # Display 8 images per row
            print(f"  🖼️ Reconstruction samples saved for epoch {epoch+1}.")
    
    epoch_duration = time.time() - epoch_start_time
    print(f"  ⏱️ Epoch {epoch+1} duration: {epoch_duration:.2f}s")


print("\n🏁 Training finished! 🏁")

# --- Final Cleanup of FID temp directories ---
try:
    if os.path.exists(temp_base_dir):
        cleaned_count = 0
        for item in os.listdir(temp_base_dir):
            item_path = os.path.join(temp_base_dir, item)
            if os.path.isdir(item_path) and item.startswith("run_"): # Be specific
                 shutil.rmtree(item_path)
                 cleaned_count +=1
        if cleaned_count > 0:
            print(f"  🧹 All {cleaned_count} temporary FID run directories in '{temp_base_dir}' have been cleaned up.")
except Exception as e_final_clean:
    print(f"  ⚠️ Error during final cleanup of FID directories: {e_final_clean}") 
