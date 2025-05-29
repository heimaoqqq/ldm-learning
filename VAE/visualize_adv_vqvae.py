import os
import torch
import matplotlib.pyplot as plt
import yaml
from adv_vq_vae import AdvVQVAE
from dataset import build_dataloader

# 加载配置
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载数据
_, val_loader, _, val_dataset_len = build_dataloader(
    config['dataset']['root_dir'],
    batch_size=8,
    num_workers=0,
    shuffle_train=False,
    shuffle_val=False,
    val_split=0.3
)

print(f"使用验证集进行可视化，验证集样本数: {val_dataset_len}")

# 创建保存目录路径
# save_dir = config['training']['save_dir'] # 不再使用配置文件中的save_dir
vis_dir = os.path.join("vqvae_vis")  # 改为相对于当前目录的路径

# 定义要可视化的模型配置
model_configs = [
    {
        'name': '预训练FID模型', # 修改名称以反映实际加载的模型
        'filename': '/kaggle/input/vae-best-fid2/adv_vqvae_best_fid2.pth', # 直接使用你的路径
        'short_name': 'pretrained_fid',
        'model': None,
        'available': False
    }
]

# 初始化和加载模型
for config_item in model_configs:
    # 创建模型实例
    model = AdvVQVAE(
        in_channels=config['model']['in_channels'],
        latent_dim=config['model']['latent_dim'],
        num_embeddings=config['model']['num_embeddings'],
        beta=config['model']['beta'],
        decay=config['model'].get('vq_ema_decay', 0.99),
        groups=config['model']['groups'],
        disc_ndf=64
    ).to(device)
    
    # 尝试加载权重
    weights_path = config_item['filename'] # 直接使用filename作为完整路径
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        config_item['model'] = model
        config_item['available'] = True
        print(f"✅ 成功加载{config_item['name']}: {weights_path}")
    else:
        print(f"❌ 未找到{config_item['name']}: {weights_path}")

# 检查至少有一个模型可用
available_models = [item for item in model_configs if item['available']]
if not available_models:
    print("❌ 没有找到任何可用的模型，请先训练模型")
    exit(1)

print(f"📊 找到 {len(available_models)} 个可用模型，将进行对比可视化")

# 设置所有可用模型为评估模式
for config_item in available_models:
    config_item['model'].eval()

# 创建可视化输出目录
os.makedirs(vis_dir, exist_ok=True)

# 可视化和比较
def denormalize(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

# 绘图函数，支持多个模型的对比
def create_multi_model_comparison_plot(original_images, model_results, batch_idx, save_path):
    """
    创建多模型对比图
    model_results: list of (reconstructed_images, model_name)
    """
    num_images = min(6, original_images.size(0))  # 每批最多显示6张图像
    num_models = len(model_results)
    
    # 计算子图布局：原始图像 + 各模型重建
    rows = num_models + 1
    cols = num_images
    
    plt.figure(figsize=(cols * 3, rows * 2.5))
    
    # 第一行：显示原始图像
    for j in range(num_images):
        plt.subplot(rows, cols, j + 1)
        plt.imshow(original_images[j].permute(1, 2, 0).cpu().numpy())
        plt.title(f"原始图像 #{j+1}", fontsize=10)
        plt.axis('off')
    
    # 后续行：显示各模型的重建结果
    for model_idx, (recon_images, model_name) in enumerate(model_results):
        row_start = (model_idx + 1) * cols + 1
        for j in range(num_images):
            plt.subplot(rows, cols, row_start + j)
            plt.imshow(recon_images[j].permute(1, 2, 0).cpu().numpy())
            plt.title(f"{model_name} #{j+1}", fontsize=10)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return save_path

# 进行多模型对比可视化
print("\n开始生成多模型对比可视化...")
with torch.no_grad():
    for i, (images, _) in enumerate(val_loader):
        if i >= 5:  # 只显示前5批次
            break
            
        images = images.to(device)
        
        # 反归一化原始图像
        norm_images = denormalize(images)
        
        # 收集所有可用模型的重建结果
        model_results = []
        for config_item in available_models:
            recon, _, _ = config_item['model'](images)
            recon_norm = denormalize(recon)
            model_results.append((recon_norm, config_item['name']))
        
        # 创建多模型对比图
        comparison_path = os.path.join(vis_dir, f"multi_model_comparison_batch_{i+1}.png")
        create_multi_model_comparison_plot(norm_images, model_results, i+1, comparison_path)
        print(f"✅ 已保存多模型对比图: {comparison_path}")
        
        # 同时为每个模型单独保存图像（保持兼容性）
        for config_item in available_models:
            recon, _, _ = config_item['model'](images)
            recon_norm = denormalize(recon)
            single_path = os.path.join(vis_dir, f"{config_item['short_name']}_batch_{i+1}.png")
            
            # 为单个模型创建对比图
            plt.figure(figsize=(16, 8))
            num_images = min(8, images.size(0))
            
            # 上排：原始图像
            for j in range(num_images):
                plt.subplot(2, num_images, j+1)
                plt.imshow(norm_images[j].permute(1, 2, 0).cpu().numpy())
                plt.title(f"原始 #{j+1}", fontsize=10)
                plt.axis('off')
            
            # 下排：重建图像
            for j in range(num_images):
                plt.subplot(2, num_images, num_images + j + 1)
                plt.imshow(recon_norm[j].permute(1, 2, 0).cpu().numpy())
                plt.title(f"{config_item['name']} #{j+1}", fontsize=10)
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(single_path, dpi=200)
            plt.close()
            print(f"✅ 已保存{config_item['name']}单独对比图: {single_path}")

print(f"\n🎉 可视化完成！")
print(f"📁 多模型对比图: {vis_dir}/multi_model_comparison_batch_*.png")
print(f"📁 单模型对比图: {vis_dir}/*_batch_*.png")
print(f"📁 总共生成了 {len(available_models) * 5 + 5} 张图像") 
