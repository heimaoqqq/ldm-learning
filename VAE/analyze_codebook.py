import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from adv_vq_vae import AdvVQVAE
from dataset import build_dataloader
from collections import Counter
import seaborn as sns

def analyze_codebook_usage():
    """详细分析VQ-VAE码本的使用情况"""
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    train_loader, val_loader, train_len, val_len = build_dataloader(
        config['dataset']['root_dir'],
        batch_size=config['dataset']['batch_size'],
        num_workers=0,
        shuffle_train=False,
        shuffle_val=False,
        val_split=0.3
    )
    
    # 定义要分析的模型
    model_configs = [
        {
            'name': '预训练FID模型',
            'filename': '/kaggle/input/vae-best-fid2/adv_vqvae_best_fid2.pth',
            'short_name': 'pretrained_fid'
        }
    ]
    
    # 分析每个可用的模型
    analysis_results = {}
    
    for model_config in model_configs:
        model_path = model_config['filename']
        if not os.path.exists(model_path):
            print(f"❌ 未找到模型: {model_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"📊 分析模型: {model_config['name']}")
        print(f"{'='*60}")
        
        # 加载模型
        model = AdvVQVAE(
            in_channels=config['model']['in_channels'],
            latent_dim=config['model']['latent_dim'],
            num_embeddings=config['model']['num_embeddings'],
            beta=config['model']['beta'],
            decay=config['model'].get('vq_ema_decay', 0.99),
            groups=config['model']['groups'],
            disc_ndf=64
        ).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ 成功加载模型: {model_path}")
        
        # 收集编码索引
        print("🔍 收集编码索引...")
        all_indices = []
        sample_count = 0
        max_samples = 1000  # 分析1000个样本
        
        with torch.no_grad():
            # 分析训练集
            for images, _ in train_loader:
                if sample_count >= max_samples:
                    break
                    
                images = images.to(device)
                z = model.encoder(images)
                
                # 直接计算编码索引，不依赖于usage_info
                z_flattened = z.permute(0,2,3,1).contiguous().view(-1, model.vq.embedding_dim)
                dist = (z_flattened.pow(2).sum(1, keepdim=True)
                        - 2 * torch.matmul(z_flattened, model.vq.embedding.weight.t())
                        + model.vq.embedding.weight.pow(2).sum(1))
                encoding_indices = torch.argmin(dist, dim=1)
                
                indices = encoding_indices.cpu().numpy()
                all_indices.extend(indices.flatten())
                sample_count += images.size(0)
            
            # 分析验证集
            val_sample_count = 0
            for images, _ in val_loader:
                if val_sample_count >= 200:  # 验证集分析200个样本
                    break
                    
                images = images.to(device)
                z = model.encoder(images)
                
                # 直接计算编码索引
                z_flattened = z.permute(0,2,3,1).contiguous().view(-1, model.vq.embedding_dim)
                dist = (z_flattened.pow(2).sum(1, keepdim=True)
                        - 2 * torch.matmul(z_flattened, model.vq.embedding.weight.t())
                        + model.vq.embedding.weight.pow(2).sum(1))
                encoding_indices = torch.argmin(dist, dim=1)
                
                indices = encoding_indices.cpu().numpy()
                all_indices.extend(indices.flatten())
                val_sample_count += images.size(0)
        
        # 检查是否收集到了索引
        if len(all_indices) == 0:
            print("❌ 未能收集到任何编码索引，跳过此模型")
            continue
            
        # 统计分析
        print(f"📈 总共分析了 {len(all_indices)} 个编码")
        
        # 基本统计
        unique_indices = set(all_indices)
        total_codes = config['model']['num_embeddings']
        usage_rate = len(unique_indices) / total_codes
        
        print(f"📊 码本使用统计:")
        print(f"  总码字数: {total_codes}")
        print(f"  使用的码字数: {len(unique_indices)}")
        print(f"  使用率: {usage_rate:.2%}")
        
        # 使用频率分析
        usage_counts = Counter(all_indices)
        most_common = usage_counts.most_common(10)
        least_common = usage_counts.most_common()[-10:]
        
        print(f"\n📈 使用频率分析:")
        print(f"  最常用的10个码字:")
        for idx, count in most_common:
            percentage = count / len(all_indices) * 100
            print(f"    码字 {idx}: 使用 {count} 次 ({percentage:.2f}%)")
        
        print(f"  最少用的10个码字:")
        for idx, count in least_common:
            percentage = count / len(all_indices) * 100
            print(f"    码字 {idx}: 使用 {count} 次 ({percentage:.2f}%)")
        
        # 统计分布分析
        counts_array = np.array(list(usage_counts.values()))
        
        # 检查counts_array是否为空
        if len(counts_array) == 0:
            print(f"\n⚠️ 警告: 没有收集到码字使用统计，跳过统计分析")
            continue
            
        print(f"\n📈 使用分布统计:")
        print(f"  平均使用次数: {counts_array.mean():.2f}")
        print(f"  使用次数标准差: {counts_array.std():.2f}")
        print(f"  使用次数中位数: {np.median(counts_array):.2f}")
        print(f"  最大使用次数: {counts_array.max()}")
        print(f"  最小使用次数: {counts_array.min()}")
        
        # 保存分析结果
        analysis_results[model_config['short_name']] = {
            'usage_rate': usage_rate,
            'used_codes': len(unique_indices),
            'total_codes': total_codes,
            'usage_counts': usage_counts,
            'counts_array': counts_array,
            'all_indices': all_indices
        }
        
        # 生成可视化
        os.makedirs("codebook_analysis", exist_ok=True)
        
        # 1. 使用频率直方图
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(counts_array, bins=50, alpha=0.7, color='skyblue')
        plt.xlabel('使用次数')
        plt.ylabel('码字数量')
        plt.title(f'{model_config["name"]} - 使用频率分布')
        plt.grid(True, alpha=0.3)
        
        # 2. 累积分布
        plt.subplot(2, 2, 2)
        sorted_counts = np.sort(counts_array)[::-1]
        cumulative_usage = np.cumsum(sorted_counts) / np.sum(sorted_counts)
        plt.plot(range(len(sorted_counts)), cumulative_usage, linewidth=2)
        plt.xlabel('码字排序（按使用频率）')
        plt.ylabel('累积使用比例')
        plt.title('累积使用分布')
        plt.grid(True, alpha=0.3)
        
        # 3. Top 50 码字使用情况
        plt.subplot(2, 2, 3)
        top_50 = usage_counts.most_common(50)
        indices, counts = zip(*top_50)
        plt.bar(range(len(indices)), counts, alpha=0.7, color='lightcoral')
        plt.xlabel('码字排序（按使用频率）')
        plt.ylabel('使用次数')
        plt.title('Top 50 最常用码字')
        plt.grid(True, alpha=0.3)
        
        # 4. 使用率vs未使用码字
        plt.subplot(2, 2, 4)
        used_codes = len(unique_indices)
        unused_codes = total_codes - used_codes
        plt.pie([used_codes, unused_codes], 
                labels=[f'已使用: {used_codes}', f'未使用: {unused_codes}'],
                autopct='%1.1f%%',
                colors=['lightgreen', 'lightgray'])
        plt.title('码字使用率')
        
        plt.tight_layout()
        save_path = f"codebook_analysis/{model_config['short_name']}_analysis.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"📁 分析图表已保存: {save_path}")
    
    # 如果有多个模型，生成对比分析
    if len(analysis_results) > 1:
        print(f"\n{'='*60}")
        print(f"📊 多模型对比分析")
        print(f"{'='*60}")
        
        plt.figure(figsize=(15, 10))
        
        model_names = []
        usage_rates = []
        used_codes_list = []
        mean_usage = []
        std_usage = []
        
        for short_name, results in analysis_results.items():
            model_names.append(short_name)
            usage_rates.append(results['usage_rate'])
            used_codes_list.append(results['used_codes'])
            mean_usage.append(results['counts_array'].mean())
            std_usage.append(results['counts_array'].std())
        
        # 1. 使用率对比
        plt.subplot(2, 3, 1)
        bars = plt.bar(model_names, usage_rates, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)])
        plt.ylabel('码本使用率')
        plt.title('不同模型的码本使用率对比')
        plt.ylim(0, max(usage_rates) * 1.2)
        for i, (bar, rate) in enumerate(zip(bars, usage_rates)):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, 
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 2. 使用码字数对比
        plt.subplot(2, 3, 2)
        bars = plt.bar(model_names, used_codes_list, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)])
        plt.ylabel('使用的码字数')
        plt.title('不同模型使用的码字数对比')
        for i, (bar, count) in enumerate(zip(bars, used_codes_list)):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, 
                    f'{count}', ha='center', va='bottom')
        
        # 3. 平均使用次数对比
        plt.subplot(2, 3, 3)
        bars = plt.bar(model_names, mean_usage, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)])
        plt.ylabel('平均使用次数')
        plt.title('不同模型的平均码字使用次数')
        for i, (bar, mean_val) in enumerate(zip(bars, mean_usage)):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, 
                    f'{mean_val:.1f}', ha='center', va='bottom')
        
        # 4. 使用分布的标准差对比
        plt.subplot(2, 3, 4)
        bars = plt.bar(model_names, std_usage, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)])
        plt.ylabel('使用次数标准差')
        plt.title('不同模型的使用分布均匀性')
        for i, (bar, std_val) in enumerate(zip(bars, std_usage)):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, 
                    f'{std_val:.1f}', ha='center', va='bottom')
        
        # 5. 使用频率分布对比
        plt.subplot(2, 3, 5)
        for short_name, results in analysis_results.items():
            counts_array = results['counts_array']
            plt.hist(counts_array, bins=30, alpha=0.5, label=short_name, density=True)
        plt.xlabel('使用次数')
        plt.ylabel('密度')
        plt.title('使用频率分布对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. 总结表格
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        table_data = []
        headers = ['模型', '使用率', '使用码字', '平均使用次数', '标准差']
        
        for i, short_name in enumerate(model_names):
            results = analysis_results[short_name]
            table_data.append([
                short_name,
                f'{results["usage_rate"]:.1%}',
                f'{results["used_codes"]}',
                f'{results["counts_array"].mean():.1f}',
                f'{results["counts_array"].std():.1f}'
            ])
        
        table = plt.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        plt.title('码本使用统计汇总', pad=20)
        
        plt.tight_layout()
        comparison_path = "codebook_analysis/model_comparison.png"
        plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"📁 对比分析图表已保存: {comparison_path}")
        
        # 打印总结
        print(f"\n📋 总结:")
        print(f"  最高使用率模型: {model_names[np.argmax(usage_rates)]} ({max(usage_rates):.1%})")
        print(f"  最均匀分布模型: {model_names[np.argmin(std_usage)]} (标准差: {min(std_usage):.1f})")
        print(f"  使用码字最多模型: {model_names[np.argmax(used_codes_list)]} ({max(used_codes_list)} 个码字)")

if __name__ == "__main__":
    analyze_codebook_usage() 
