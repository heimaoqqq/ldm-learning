import os
import sys
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from cldm import CLDM
import torchvision.utils as vutils

# 添加上级目录到路径，以便导入VQ-VAE模型
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VAE'))
from adv_vq_vae import AdvVQVAE
from dataset import build_dataloader

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def denormalize(x):
    """反归一化，将[-1,1]转换为[0,1]"""
    return (x * 0.5 + 0.5).clamp(0, 1)

def generate_class_samples(model, vqvae_model, num_classes, samples_per_class=4, device='cuda', sampling_steps=100):
    """为每个类别生成样本"""
    model.eval()
    
    all_generated_images = []
    class_labels = []
    
    with torch.no_grad():
        for class_id in range(num_classes):
            # 为当前类别生成多个样本
            labels = torch.full((samples_per_class,), class_id, device=device)
            
            # 生成潜在向量（使用DDIM加速）
            generated_z = model.sample(labels, device, sampling_steps=sampling_steps)
            
            # 通过VQ-VAE解码器生成图像
            generated_images = vqvae_model.decoder(generated_z)
            generated_images = denormalize(generated_images)
            
            all_generated_images.append(generated_images)
            class_labels.extend([class_id] * samples_per_class)
    
    return torch.cat(all_generated_images, dim=0), class_labels

def create_class_comparison_grid(generated_images, class_labels, num_classes, samples_per_class):
    """创建按类别排列的图像网格"""
    fig, axes = plt.subplots(num_classes, samples_per_class, 
                            figsize=(samples_per_class * 3, num_classes * 3))
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    if samples_per_class == 1:
        axes = axes.reshape(-1, 1)
    
    for class_id in range(num_classes):
        for sample_id in range(samples_per_class):
            idx = class_id * samples_per_class + sample_id
            img = generated_images[idx].permute(1, 2, 0).cpu().numpy()
            
            axes[class_id, sample_id].imshow(img)
            axes[class_id, sample_id].set_title(f'ID_{class_id+1}', fontsize=10)
            axes[class_id, sample_id].axis('off')
    
    plt.tight_layout()
    return fig

def compare_real_vs_generated(vqvae_model, cldm_model, val_loader, num_samples=8, device='cuda', sampling_steps=100):
    """比较真实图像与生成图像"""
    cldm_model.eval()
    
    # 获取一批真实图像
    real_images, real_labels = next(iter(val_loader))
    real_images = real_images[:num_samples].to(device)
    real_labels = real_labels[:num_samples].to(device)
    
    with torch.no_grad():
        # 生成对应类别的图像（使用DDIM）
        generated_z = cldm_model.sample(real_labels, device, sampling_steps=sampling_steps)
        generated_images = vqvae_model.decoder(generated_z)
        
        # 反归一化
        real_images_norm = denormalize(real_images)
        generated_images_norm = denormalize(generated_images)
        
        # 创建对比图
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
        
        for i in range(num_samples):
            # 真实图像
            real_img = real_images_norm[i].permute(1, 2, 0).cpu().numpy()
            axes[0, i].imshow(real_img)
            axes[0, i].set_title(f'真实 ID_{real_labels[i].item()+1}', fontsize=10)
            axes[0, i].axis('off')
            
            # 生成图像
            gen_img = generated_images_norm[i].permute(1, 2, 0).cpu().numpy()
            axes[1, i].imshow(gen_img)
            axes[1, i].set_title(f'生成 ID_{real_labels[i].item()+1}', fontsize=10)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        return fig, real_images_norm, generated_images_norm

def test_interpolation(cldm_model, vqvae_model, class1, class2, num_steps=8, device='cuda', sampling_steps=100):
    """测试类别间的插值"""
    cldm_model.eval()
    
    with torch.no_grad():
        # 生成两个类别的潜在向量（使用DDIM）
        labels1 = torch.full((1,), class1, device=device)
        labels2 = torch.full((1,), class2, device=device)
        
        z1 = cldm_model.sample(labels1, device, sampling_steps=sampling_steps)
        z2 = cldm_model.sample(labels2, device, sampling_steps=sampling_steps)
        
        # 在潜在空间中插值
        interpolated_images = []
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # 解码插值后的潜在向量
            img_interp = vqvae_model.decoder(z_interp)
            img_interp = denormalize(img_interp)
            interpolated_images.append(img_interp)
        
        interpolated_images = torch.cat(interpolated_images, dim=0)
        
        # 创建插值可视化
        fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 3, 3))
        
        for i in range(num_steps):
            img = interpolated_images[i].permute(1, 2, 0).cpu().numpy()
            axes[i].imshow(img)
            if i == 0:
                axes[i].set_title(f'ID_{class1+1}', fontsize=10)
            elif i == num_steps - 1:
                axes[i].set_title(f'ID_{class2+1}', fontsize=10)
            else:
                axes[i].set_title(f'插值 {i}', fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig, interpolated_images

def main():
    # 加载配置
    config = load_config()
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建可视化输出目录
    vis_dir = "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
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
        return
    
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
    
    # 尝试加载训练好的CLDM权重
    save_dir = config['training']['save_dir']
    cldm_paths = [
        os.path.join(save_dir, 'cldm_best_loss.pth'),
        os.path.join(save_dir, 'cldm_best_fid.pth')
    ]
    
    cldm_loaded = False
    for path in cldm_paths:
        if os.path.exists(path):
            cldm.load_state_dict(torch.load(path, map_location=device))
            print(f"成功加载CLDM权重: {path}")
            cldm_loaded = True
            break
    
    if not cldm_loaded:
        print("警告: 找不到训练好的CLDM权重文件")
        print("将使用随机初始化的模型进行测试（结果可能不理想）")
    
    cldm.eval()
    
    # 加载数据集用于对比
    _, val_loader, _, val_dataset_len = build_dataloader(
        config['dataset']['root_dir'],
        batch_size=8,
        num_workers=0,
        shuffle_train=False,
        shuffle_val=False,
        val_split=0.3
    )
    
    print(f"验证集样本数: {val_dataset_len}")
    
    # 主要可视化流程
    print("开始生成可视化结果...")
    
    # 获取采样步数配置
    sampling_steps = config['training'].get('sampling_steps', 100)
    print(f"使用DDIM {sampling_steps}步采样")
    
    # 1. 为每个类别生成样本（限制为前8个类别）
    max_classes_to_show = min(8, config['cldm']['num_classes'])
    samples_per_class = 4
    
    print(f"为前{max_classes_to_show}个类别生成样本...")
    generated_images, class_labels = generate_class_samples(
        cldm, vqvae, max_classes_to_show, samples_per_class, device, sampling_steps
    )
    
    # 保存类别样本网格
    print("创建类别样本网格...")
    class_grid_fig = create_class_comparison_grid(
        generated_images, class_labels, max_classes_to_show, samples_per_class
    )
    class_grid_path = os.path.join(vis_dir, "class_samples_grid.png")
    class_grid_fig.savefig(class_grid_path, dpi=200, bbox_inches='tight')
    plt.close(class_grid_fig)
    print(f"保存类别样本网格: {class_grid_path}")
    
    # 保存为单个图像文件
    class_samples_path = os.path.join(vis_dir, "class_samples.png")
    vutils.save_image(
        generated_images,
        class_samples_path,
        nrow=samples_per_class,
        padding=2,
        normalize=False
    )
    print(f"保存类别样本图像: {class_samples_path}")
    
    # 2. 真实图像与生成图像对比
    print("创建真实vs生成图像对比...")
    comparison_fig, real_imgs, gen_imgs = compare_real_vs_generated(vqvae, cldm, val_loader, 8, device, sampling_steps)
    comparison_path = os.path.join(vis_dir, "real_vs_generated.png")
    comparison_fig.savefig(comparison_path, dpi=200, bbox_inches='tight')
    plt.close(comparison_fig)
    print(f"保存对比图像: {comparison_path}")
    
    # 3. 类别间插值
    if max_classes_to_show >= 2:
        print("创建类别间插值...")
        class1, class2 = 0, min(1, max_classes_to_show - 1)
        interp_fig, interp_imgs = test_interpolation(cldm, vqvae, class1, class2, 8, device, sampling_steps)
        interp_path = os.path.join(vis_dir, f"interpolation_ID{class1+1}_to_ID{class2+1}.png")
        interp_fig.savefig(interp_path, dpi=200, bbox_inches='tight')
        plt.close(interp_fig)
        print(f"保存插值图像: {interp_path}")
    
    # 4. 大量随机采样
    print("生成大量随机采样...")
    num_random_samples = 64
    random_labels = torch.randint(0, max_classes_to_show, (num_random_samples,), device=device)
    
    with torch.no_grad():
        random_z = cldm.sample(random_labels, device, sampling_steps=sampling_steps)
        random_images = vqvae.decoder(random_z)
        random_images = denormalize(random_images)
    
    random_samples_path = os.path.join(vis_dir, "random_samples.png")
    vutils.save_image(
        random_images,
        random_samples_path,
        nrow=8,
        padding=2,
        normalize=False
    )
    print(f"保存随机采样图像: {random_samples_path}")
    
    # 5. 生成统计信息
    print("\n生成统计信息:")
    print(f"  CLDM模型参数量: {sum(p.numel() for p in cldm.parameters()):,}")
    print(f"  生成的图像数量: {len(generated_images)}")
    print(f"  图像分辨率: {generated_images.shape[-2:]} (潜在空间) -> {real_imgs.shape[-2:]} (像素空间)")
    print(f"  支持的类别数量: {config['cldm']['num_classes']}")
    
    print(f"\n可视化完成！请查看 {vis_dir} 目录下的结果图像:")
    print(f"  - class_samples_grid.png: 按类别排列的生成样本")
    print(f"  - class_samples.png: 类别样本（网格格式）")
    print(f"  - real_vs_generated.png: 真实图像与生成图像对比")
    if max_classes_to_show >= 2:
        print(f"  - interpolation_ID1_to_ID2.png: 类别间插值")
    print(f"  - random_samples.png: 随机采样结果")

if __name__ == "__main__":
    main() 