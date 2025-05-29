import os
import yaml
import torch
import torch.nn as nn
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

# 改进的torch-fidelity导入方式
HAS_FIDELITY = False
try:
    from torch_fidelity import calculate_metrics as original_calculate_metrics
    # 创建一个包装函数来控制输出
    def calculate_metrics(*args, **kwargs):
        """
        对torch_fidelity.calculate_metrics的包装，用于控制其输出冗余度
        """
        # 导入必要的模块
        import io
        import sys
        from contextlib import redirect_stdout
        
        # 捕获原始函数的所有输出
        f = io.StringIO()
        with redirect_stdout(f):
            # 调用原始函数
            result = original_calculate_metrics(*args, **kwargs)
        
        # 输出只包含结果的简化信息
        if result and 'frechet_inception_distance' in result:
            print(f"FID分: {result['frechet_inception_distance']:.4f}")
        
        return result
    
    HAS_FIDELITY = True
    print("成功导入torch-fidelity库，将计算FID指标")
except ImportError as e:
    print(f"警告: 无法导入torch-fidelity库，无法计算FID。错误信息: {e}")
    print("请运行 'pip install torch-fidelity' 安装后重试")
    print("继续训练，但不会计算FID指标")

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
temp_base_dir = os.path.join("temp_fid_images")  # 基础临时目录
os.makedirs(temp_base_dir, exist_ok=True)

# 在每次FID计算中使用唯一的子目录避免冲突
def get_temp_fid_dirs():
    """创建并返回用于FID计算的临时目录"""
    # 使用UUID创建唯一的目录名
    unique_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join(temp_base_dir, f"run_{unique_id}")
    temp_real_dir = os.path.join(temp_dir, "real")
    temp_gen_dir = os.path.join(temp_dir, "generated")
    
    # 创建目录
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(temp_real_dir, exist_ok=True)
    os.makedirs(temp_gen_dir, exist_ok=True)
    
    return temp_dir, temp_real_dir, temp_gen_dir

# 构建数据加载器
train_loader, val_loader, train_dataset_len, val_dataset_len = build_dataloader(
    config['dataset']['root_dir'],
    batch_size=config['dataset']['batch_size'],
    num_workers=0, # 建议在Windows上调试时设为0，或者根据CPU核心数调整
    shuffle_train=True,
    shuffle_val=False, # 验证集通常不需要打乱
    val_split=0.3 # 30% 作为验证集
)
dataset_len = train_dataset_len # 用于旧的平均损失计算基数，现在主要用 train_dataset_len

# 初始化模型
model = AdvVQVAE(
    in_channels=config['model']['in_channels'],
    latent_dim=config['model']['latent_dim'],
    num_embeddings=config['model']['num_embeddings'],
    beta=config['model']['beta'],
    decay=config['model'].get('vq_ema_decay', 0.99),  # 使用配置的 EMA 衰减率
    groups=config['model']['groups'],
    disc_ndf=64  # 判别器特征数量
).to(device)

# 创建优化器
gen_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.vq.parameters())
gen_optimizer = optim.Adam(
    gen_params,
    lr=config['training']['lr'],
    betas=(0.9, 0.99), # 根据论文更新
    weight_decay=config['training']['weight_decay']
)
disc_optimizer = optim.Adam(
    model.discriminator.parameters(),
    lr=config['training']['lr'] * 0.5,  # 判别器学习率通常略小
    betas=(0.5, 0.9), # 根据论文更新
    weight_decay=config['training']['weight_decay']
)

# 感知损失
perceptual_loss_fn = VGGPerceptualLoss().to(device) # 重命名以区分变量和类实例
perceptual_weight = config['training'].get('perceptual_weight', 0.2) # 从配置读取或默认

# 用于反归一化
def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

# 创建保存目录
save_dir = config['training']['save_dir']
os.makedirs(save_dir, exist_ok=True)

# FID计算函数
def calculate_fid(model, val_loader, temp_real_dir, temp_gen_dir, sample_size=1000):
    """计算FID分数"""
    if not HAS_FIDELITY:
        print("  无法计算FID: torch-fidelity库未成功导入")
        return float('inf')
    
    try:
        # 临时目录已由调用者创建，这里只需确认目录存在
        for dir_path in [temp_real_dir, temp_gen_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"  创建缺失目录: {dir_path}")
        
        model.eval()
        with torch.no_grad():
            sample_count = 0
            for val_images, _ in tqdm(val_loader, desc="生成FID图像样本"):
                val_images = val_images.to(device)
                batch_size = val_images.size(0)
                
                # 通过模型获取重建图像
                val_z = model.encoder(val_images)
<<<<<<< HEAD
                val_z_q, _, _ = model.vq(val_z)
=======
                val_z_q, _, _, _ = model.vq(val_z)  # 新增usage_info返回值
>>>>>>> 126d4ad24fc08805125e6779329dd8caece6ed5d
                val_recons = model.decoder(val_z_q)
                
                # 反归一化
                real_images = denorm(val_images)
                gen_images = denorm(val_recons)
                
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
        
        if sample_count < 10:
            print(f"  警告: 样本数量过少 ({sample_count}), FID计算可能不准确")
            return float('inf')
            
        # 计算FID
        # 使用io.StringIO捕获标准输出
        import io
        import sys
        from contextlib import redirect_stdout

        # 创建StringIO对象来捕获输出
        f = io.StringIO()
        with redirect_stdout(f):
            metrics = calculate_metrics(
                input1=temp_real_dir, 
                input2=temp_gen_dir,
                cuda=torch.cuda.is_available(), 
                fid=True,
                verbose=True
            )
        
        # 获取捕获的输出
        output = f.getvalue()
        print(output)  # 仍然打印原始输出
        
        # 尝试从输出中解析FID值
        import re
        fid_match = re.search(r'Frechet Inception Distance: (\d+\.\d+)', output)
        parsed_fid = None
        if fid_match:
            parsed_fid = float(fid_match.group(1))
            print(f"  从输出解析的FID值: {parsed_fid:.4f}")
        
        # 首先尝试从metrics字典获取
        if 'frechet_inception_distance' in metrics:
            fid_score = metrics['frechet_inception_distance']
            print(f"  从metrics字典获取的FID ('frechet_inception_distance'): {fid_score:.4f}")
        elif 'fid' in metrics:
            fid_score = metrics['fid']
            print(f"  从metrics字典获取的FID ('fid'): {fid_score:.4f}")
        else:
            # 如果metrics字典中没有找到，使用解析的值
            print(f"  metrics字典中没有标准FID键，可用键: {list(metrics.keys())}")
            if parsed_fid is not None:
                fid_score = parsed_fid
                print(f"  使用从输出解析的FID值: {fid_score:.4f}")
            elif metrics:
                # 尝试使用字典中的第一个值
                fid_score = list(metrics.values())[0]
                print(f"  使用metrics字典中的第一个值作为FID: {fid_score:.4f}")
            else:
                print("  未找到任何FID值，返回无穷大")
                fid_score = float('inf')

        print(f"  FID评分: {fid_score:.4f}")
        return fid_score
    except Exception as e:
        print(f"  计算FID时出错: {str(e)}")
        print(f"  错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()  # 打印完整错误堆栈
        
        # 如果错误是KeyError或其他已知错误，并且有输出供解析
        if isinstance(e, KeyError) and 'output' in locals():
            # 尝试从已捕获的输出中解析FID值
            fid_match = re.search(r'Frechet Inception Distance: (\d+\.\d+)', output)
            if fid_match:
                fid_score = float(fid_match.group(1))
                print(f"  尽管发生错误，但成功从输出解析FID值: {fid_score:.4f}")
                return fid_score
        
        print("  无法恢复FID值，返回无穷大")
        return float('inf')

# 修改FID计算相关代码段，使用简化版本
def calculate_fid_simple(model, val_loader, sample_size=1000):
    """简化的FID计算函数，只输出关键信息"""
    if not HAS_FIDELITY:
        print("无法计算FID: torch-fidelity库未成功导入")
        return float('inf')
    
    try:
        # 创建临时目录
        temp_dir, temp_real_dir, temp_gen_dir = get_temp_fid_dirs()
        
        # 生成样本和保存图像的代码保持不变，但不输出详细进度
        sample_count = 0
        model.eval()
        with torch.no_grad():
            for val_images, _ in val_loader:
                val_images = val_images.to(device)
                
                # 使用模型生成重建图像
                z = model.encoder(val_images)
<<<<<<< HEAD
                z_q, _, _ = model.vq(z)
=======
                z_q, _, _, _ = model.vq(z)  # 新增usage_info返回值
>>>>>>> 126d4ad24fc08805125e6779329dd8caece6ed5d
                recons = model.decoder(z_q)
                
                # 保存真实图像和生成图像用于FID计算
                for i in range(val_images.size(0)):
                    if sample_count >= sample_size:
                        break
                    
                    # 保存真实图像
                    real_img = denorm(val_images[i])
                    real_path = os.path.join(temp_real_dir, f"real_{sample_count}.png")
                    vutils.save_image(real_img, real_path)
                    
                    # 保存生成的图像
                    gen_img = denorm(recons[i])
                    gen_path = os.path.join(temp_gen_dir, f"gen_{sample_count}.png")
                    vutils.save_image(gen_img, gen_path)
                    
                    sample_count += 1
                    
                if sample_count >= sample_size:
                    break
        
        if sample_count < 10:
            print("警告: 样本数量过少, FID计算可能不准确")
            return float('inf')
        
        # 计算FID - 使用我们的包装函数会自动简化输出
        metrics = calculate_metrics(
            input1=temp_real_dir, 
            input2=temp_gen_dir,
            cuda=torch.cuda.is_available(), 
            fid=True,
            verbose=False  # 关闭详细输出
        )
        
        # 从metrics字典获取FID
        if 'frechet_inception_distance' in metrics:
            fid_score = metrics['frechet_inception_distance']
        elif 'fid' in metrics:
            fid_score = metrics['fid']
        else:
            # 如果找不到FID值
            print("未找到FID值，返回无穷大")
            return float('inf')
            
        # 计算完成后删除临时目录
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception:
            pass
            
        return fid_score
    except Exception as e:
        print(f"计算FID时出错: {str(e)}")
        return float('inf')

# 训练循环
best_val_loss = float('inf') # 使用验证集损失来保存最佳模型
<<<<<<< HEAD
best_fid = float('inf')      # 使用FID评分来保存最佳模型
=======
best_val_rec_loss = float('inf') # 新增：使用验证集重建损失来保存最佳模型
best_fid = float('inf') # 初始化最佳FID分数
>>>>>>> 126d4ad24fc08805125e6779329dd8caece6ed5d
epochs = config['training']['epochs']
disc_train_interval = config['training'].get('disc_train_interval', 5) # 判别器训练间隔

# 添加学习率调度器 - 线性预热+余弦退火
use_scheduler = config['training'].get('use_scheduler', True)
warmup_epochs = config['training'].get('warmup_epochs', 10)  # 默认为训练总轮数的10%

if use_scheduler:
    # 创建基于epoch的学习率调度器
    
    # 创建学习率调度器
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性预热
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            # 余弦退火
            progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    gen_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda)
    disc_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_optimizer, lr_lambda)
    print(f"已启用学习率调度器: 线性预热({warmup_epochs}轮，占总训练的{warmup_epochs/epochs*100:.1f}%) + 余弦退火 (epoch级)")
else:
    gen_scheduler = None
    disc_scheduler = None
    print("不使用学习率调度器")

print(f"开始训练，总共{epochs}轮...")
print(f"训练集样本数: {train_dataset_len}, 验证集样本数: {val_dataset_len}")
print(f"FID评估频率: 每{fid_eval_freq}个epoch计算一次")
print(f"FID计算状态: {'启用' if HAS_FIDELITY else '禁用'}")

# 从 config 中获取权重
rec_weight = config['training'].get('rec_weight', 1.0)
vq_commit_weight = config['training'].get('vq_commit_weight', 0.5) # beta in VQ
# perceptual_weight 已经在上面定义
gen_adv_weight = config['training'].get('gen_adv_weight', 0.2)

print(f"损失权重: rec_weight={rec_weight}, vq_commit_weight(beta)={vq_commit_weight}, perceptual_weight={perceptual_weight}, gen_adv_weight={gen_adv_weight}")
print(f"注意: vq_commit_weight 是 VQ commitment loss 中的 beta。总的 VQ regularization loss (codebook + commit) 权重为 1.0")

for epoch in range(epochs):
    model.train() # 设置为训练模式
<<<<<<< HEAD
=======
    
    # 在每个epoch开始时重置码本使用统计
    if hasattr(model.vq, 'reset_usage_stats'):
        model.vq.reset_usage_stats()
        print(f"  已重置累积码本使用计数器。")
    
>>>>>>> 126d4ad24fc08805125e6779329dd8caece6ed5d
    running_rec_loss = 0.0
    running_vq_commit_loss = 0.0 # VQ commitment loss (beta * ||E(x) - sg(z_q)||^2)
    running_vq_codebook_loss = 0.0 # VQ codebook loss (||sg(E(x)) - z_q||^2)
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
<<<<<<< HEAD
                z_q, _, _ = model.vq(z) 
=======
                z_q, _, _, _ = model.vq(z)  # 新增usage_info返回值
>>>>>>> 126d4ad24fc08805125e6779329dd8caece6ed5d
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
        
        # z 是编码器输出 E(x)
        # z_q_straight_through 是直通估计器的输出，用于解码器
        # vq_commit_loss_val 是 beta * ||E(x) - sg(z_q)||^2
        # vq_codebook_loss_val 是 ||sg(E(x)) - z_q||^2
<<<<<<< HEAD
        z = model.encoder(images)
        z_q_straight_through, vq_commit_loss_val, vq_codebook_loss_val = model.vq(z)
=======
        # usage_info 是码本使用信息
        z = model.encoder(images)
        z_q_straight_through, vq_commit_loss_val, vq_codebook_loss_val, usage_info = model.vq(z)
>>>>>>> 126d4ad24fc08805125e6779329dd8caece6ed5d
        recons = model.decoder(z_q_straight_through)
        
        rec_loss = nn.MSELoss()(recons, images)
        p_loss = perceptual_loss_fn(denorm(recons), denorm(images)) # 感知损失的输入需要denorm
        
        # 对抗损失（让判别器认为生成图像是真实的）
        fake_preds_for_gen = model.discriminator(recons)
        g_adv_loss = nn.BCEWithLogitsLoss()(fake_preds_for_gen, torch.ones_like(fake_preds_for_gen))
        
        # VQ Regularization Loss (L_reg) = codebook_loss + commitment_loss
        # commitment_loss_val 已经乘以 beta
        l_reg = vq_codebook_loss_val + vq_commit_loss_val

        # 总生成器损失 (L_total) = L_rec + L_reg + lambda_perc * L_perc + lambda_adv_G * L_adv_G
        # rec_weight (来自config) = 1.0
        # L_reg 权重隐式为 1.0
        # perceptual_weight (来自config) = 0.1
        # gen_adv_weight (来自config) = 0.1
        g_total = (rec_weight * rec_loss + 
                   l_reg + 
                   perceptual_weight * p_loss + 
                   gen_adv_weight * g_adv_loss)
        
        g_total.backward()
        gen_optimizer.step()
        
        running_rec_loss += rec_loss.item() * batch_size
        running_vq_commit_loss += vq_commit_loss_val.item() * batch_size
        running_vq_codebook_loss += vq_codebook_loss_val.item() * batch_size
        running_gen_adv_loss += g_adv_loss.item() * batch_size
        running_perc_loss += p_loss.item() * batch_size
        
        # 批次级学习率调度已移除，改为epoch级
    
    # 计算训练集上的平均损失
    epoch_rec_loss = running_rec_loss / train_dataset_len
    epoch_vq_commit_loss = running_vq_commit_loss / train_dataset_len # 这是 commitment loss
    epoch_vq_codebook_loss = running_vq_codebook_loss / train_dataset_len # 这是 codebook loss
    epoch_gen_adv_loss = running_gen_adv_loss / train_dataset_len
    epoch_perc_loss = running_perc_loss / train_dataset_len
    
    if disc_batches_trained > 0:
        epoch_disc_loss = running_disc_loss / (disc_batches_trained * config['dataset']['batch_size'])
    else:
        epoch_disc_loss = 0.0

<<<<<<< HEAD
    print(f"Epoch {epoch+1} Training Losses:")
    print(f"  Rec Loss: {epoch_rec_loss:.6f}, VQ Commit Loss: {epoch_vq_commit_loss:.6f}, VQ Codebook Loss: {epoch_vq_codebook_loss:.6f}")
    print(f"  Gen Adversarial Loss: {epoch_gen_adv_loss:.6f}, Perceptual Loss: {epoch_perc_loss:.6f}")
    print(f"  Disc Loss: {epoch_disc_loss:.6f} (trained on {disc_batches_trained} batches)")

=======
    # 获取码本使用统计
    if hasattr(model.vq, 'get_usage_stats'):
        usage_stats = model.vq.get_usage_stats()
        usage_rate = usage_stats['usage_rate']
        used_codes = usage_stats['used_codes']
        total_codes = usage_stats['total_codes']
    else:
        usage_rate = 0.0
        used_codes = 0
        total_codes = config['model']['num_embeddings']

    print(f"📊 Epoch {epoch+1} Training Summary:")
    print(f"     Avg Gen Loss: {epoch_rec_loss + epoch_vq_commit_loss + epoch_vq_codebook_loss + epoch_gen_adv_loss + epoch_perc_loss:.4f}, Avg Disc Loss: {epoch_disc_loss:.4f}")
    print(f"     Avg Rec Loss: {epoch_rec_loss:.4f}, Avg Perceptual Loss: {epoch_perc_loss:.4f}")
    print(f"     Avg VQ Commit: {epoch_vq_commit_loss:.4f}, Avg VQ Codebook: {epoch_vq_codebook_loss:.4f}")
    print(f"     📈 Cumulative Codebook Usage (end of epoch): {usage_rate:.2%}")
    
>>>>>>> 126d4ad24fc08805125e6779329dd8caece6ed5d
    # --- 在验证集上评估 ---
    model.eval() # 设置为评估模式
    val_running_rec_loss = 0.0
    val_running_vq_commit_loss = 0.0
    val_running_vq_codebook_loss = 0.0
    val_running_perc_loss = 0.0
    
    with torch.no_grad():
        for val_images, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
            val_images = val_images.to(device)
            
            val_z = model.encoder(val_images)
<<<<<<< HEAD
            val_z_q_st, val_vq_commit, val_vq_codebook = model.vq(val_z)
=======
            val_z_q_st, val_vq_commit, val_vq_codebook, val_usage_info = model.vq(val_z)  # 新增usage_info返回值
>>>>>>> 126d4ad24fc08805125e6779329dd8caece6ed5d
            val_recons = model.decoder(val_z_q_st)
            
            val_rec = nn.MSELoss()(val_recons, val_images)
            val_perceptual = perceptual_loss_fn(denorm(val_recons), denorm(val_images))
            
            val_running_rec_loss += val_rec.item() * val_images.size(0)
            val_running_vq_commit_loss += val_vq_commit.item() * val_images.size(0)
            val_running_vq_codebook_loss += val_vq_codebook.item() * val_images.size(0)
            val_running_perc_loss += val_perceptual.item() * val_images.size(0)
            
    epoch_val_rec_loss = val_running_rec_loss / val_dataset_len
    epoch_val_vq_commit_loss = val_running_vq_commit_loss / val_dataset_len
    epoch_val_vq_codebook_loss = val_running_vq_codebook_loss / val_dataset_len
    epoch_val_perc_loss = val_running_perc_loss / val_dataset_len
    
    # 计算用于比较和保存模型的验证集总损失
    # 这个组合可以根据实际效果调整，这里简单地将主要的生成器相关损失加起来
    # 根据论文 L_total = L_rec + L_reg + 0.1 * L_perc (对抗损失不用于验证)
    val_l_reg = epoch_val_vq_codebook_loss + epoch_val_vq_commit_loss
    current_val_loss = (rec_weight * epoch_val_rec_loss + 
                         val_l_reg + 
                         perceptual_weight * epoch_val_perc_loss)

<<<<<<< HEAD
    print(f"Epoch {epoch+1} Validation Losses:")
    print(f"  Val Rec Loss: {epoch_val_rec_loss:.6f}, Val VQ Commit: {epoch_val_vq_commit_loss:.6f}, Val VQ Codebook: {epoch_val_vq_codebook_loss:.6f}")
    print(f"  Val Perceptual Loss: {epoch_val_perc_loss:.6f}")
    print(f"  Current Combined Val Loss (for saving): {current_val_loss:.6f}")

    # 保存基于验证损失的最佳模型
    if current_val_loss < best_val_loss:
        old_best = best_val_loss  # 保存旧的最佳值用于日志
        best_val_loss = current_val_loss
        torch.save(model.state_dict(), os.path.join(save_dir, 'adv_vqvae_best_loss.pth'))
        print(f"  ✓ 发现更好的验证损失: {best_val_loss:.6f}，已保存模型")
        if epoch == 0:
            print(f"    → 初始模型，设置为基准值")
        else:
            print(f"    → 之前最佳: {old_best:.6f} | 改善: {old_best - current_val_loss:.6f}")
    else:
        print(f"  ✗ 当前验证损失 {current_val_loss:.6f} 未改善 (当前最佳: {best_val_loss:.6f})")
=======
    print(f"📉 Epoch {epoch+1} Validation Summary - Combined Loss: {current_val_loss:.4f}")
    print(f"     Avg Rec: {epoch_val_rec_loss:.4f}, Avg Perc: {epoch_val_perc_loss:.4f}")
    print(f"     Avg VQ Commit: {epoch_val_vq_commit_loss:.4f}, Avg VQ Codebook: {epoch_val_vq_codebook_loss:.4f}")
    print(f"     📊 当前累积码本使用率: {usage_rate:.2%} ({used_codes}/{total_codes} 码字)")

    # 保存基于验证重建损失的最佳模型（主要策略）
    if epoch_val_rec_loss < best_val_rec_loss:
        old_best = best_val_rec_loss  # 保存旧的最佳值用于日志
        best_val_rec_loss = epoch_val_rec_loss
        
        # 删除旧的模型文件（如果存在）
        best_rec_model_path = os.path.join(save_dir, 'adv_vqvae_best_loss.pth')
        if os.path.exists(best_rec_model_path):
            try:
                os.remove(best_rec_model_path)
                print(f"  🗑️ 已删除旧的最佳重建损失模型")
            except Exception as e:
                print(f"  ⚠️ 删除旧模型时出错: {e}")
        
        # 保存新的最佳模型
        torch.save(model.state_dict(), best_rec_model_path)
        print(f"  ⭐ 发现更好的验证重建损失: {best_val_rec_loss:.6f}，已保存模型")
        if epoch == 0:
            print(f"    → 初始模型，设置为基准值")
        else:
            print(f"    → 之前最佳: {old_best:.6f} | 改善: {old_best - epoch_val_rec_loss:.6f}")
    else:
        print(f"  ✗ 当前验证重建损失 {epoch_val_rec_loss:.6f} 未改善 (当前最佳: {best_val_rec_loss:.6f})")
    
    # 可选：也保存基于综合损失的模型（用于对比）
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        
        # 删除旧的综合损失模型文件（如果存在）
        best_combined_model_path = os.path.join(save_dir, 'adv_vqvae_best_combined_loss.pth')
        if os.path.exists(best_combined_model_path):
            try:
                os.remove(best_combined_model_path)
                print(f"  🗑️ 已删除旧的最佳综合损失模型")
            except Exception as e:
                print(f"  ⚠️ 删除旧综合损失模型时出错: {e}")
        
        # 保存新的最佳综合损失模型
        torch.save(model.state_dict(), best_combined_model_path)
        print(f"  📊 综合损失最佳: {best_val_loss:.6f}，已保存对比模型")
>>>>>>> 126d4ad24fc08805125e6779329dd8caece6ed5d
    
    # 更新学习率调度器（epoch级）
    if use_scheduler:
        gen_scheduler.step()
        disc_scheduler.step()
        current_gen_lr = gen_optimizer.param_groups[0]['lr']
        current_disc_lr = disc_optimizer.param_groups[0]['lr']
        print(f"  学习率更新 - Gen: {current_gen_lr:.6f}, Disc: {current_disc_lr:.6f}")
    
    # 每fid_eval_freq个epoch计算一次FID
    if (epoch + 1) % fid_eval_freq == 0:
        print(f"\nEpoch {epoch+1}: 正在计算FID评分...")
        if HAS_FIDELITY:
            try:
                # 记录开始时间
                start_time = time.time()
                fid_score = calculate_fid_simple(model, val_loader, sample_size=fid_sample_size)
                elapsed_time = time.time() - start_time
                print(f"FID评分: {fid_score:.4f}，耗时: {elapsed_time:.2f}秒")
                
                # 保存基于FID评分的最佳模型
                if fid_score < best_fid:
                    old_best = best_fid  # 保存旧的最佳值用于日志
                    best_fid = fid_score
<<<<<<< HEAD
=======
                    
                    # 删除旧的FID模型文件（如果存在）
                    best_fid_model_path = os.path.join(save_dir, 'adv_vqvae_best_fid.pth')
                    if os.path.exists(best_fid_model_path):
                        try:
                            os.remove(best_fid_model_path)
                            print(f"  🗑️ 已删除旧的最佳FID模型")
                        except Exception as e:
                            print(f"  ⚠️ 删除旧FID模型时出错: {e}")
                    
                    # 保存新的最佳FID模型
                    torch.save(model.state_dict(), best_fid_model_path)
>>>>>>> 126d4ad24fc08805125e6779329dd8caece6ed5d
                    print(f"  ✓ 发现更好的FID评分: {best_fid:.4f}，已保存模型")
                    if epoch == 0:
                        print(f"    → 初始FID评估，设置为基准值")
                    else:
                        print(f"    → 之前最佳: {old_best:.4f} | 改善: {old_best - fid_score:.4f}")
<<<<<<< HEAD
                    torch.save(model.state_dict(), os.path.join(save_dir, 'adv_vqvae_best_fid.pth'))
=======
>>>>>>> 126d4ad24fc08805125e6779329dd8caece6ed5d
                else:
                    print(f"  ✗ 当前FID评分 {fid_score:.4f} 未改善 (当前最佳: {best_fid:.4f})")
            except Exception as e:
                print(f"FID计算过程中出现错误: {str(e)}")
        else:
            print("跳过FID计算：torch-fidelity库未成功导入")
    
    # 每20轮保存一次checkpoint
    if (epoch + 1) % config['training'].get('save_interval', 20) == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'gen_optimizer_state_dict': gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': disc_optimizer.state_dict(),
            'best_val_loss': best_val_loss,
<<<<<<< HEAD
=======
            'best_val_rec_loss': best_val_rec_loss,  # 新增
>>>>>>> 126d4ad24fc08805125e6779329dd8caece6ed5d
            'best_fid': best_fid if 'best_fid' in locals() else float('inf'),
        }, os.path.join(save_dir, f'adv_vqvae_epoch_{epoch+1}.pth'))
        print(f"  保存checkpoint: adv_vqvae_epoch_{epoch+1}.pth")

print("训练完成！")
# 删除临时文件目录
if os.path.exists(temp_base_dir):
<<<<<<< HEAD
    shutil.rmtree(temp_base_dir) 
=======
    shutil.rmtree(temp_base_dir) 
>>>>>>> 126d4ad24fc08805125e6779329dd8caece6ed5d
