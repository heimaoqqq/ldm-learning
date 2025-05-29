import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from typing import Dict, List, Optional

def load_training_log(log_path: str) -> Dict:
    """加载训练日志"""
    with open(log_path, 'r') as f:
        return json.load(f)

def plot_training_curves(log_data: Dict, save_path: Optional[str] = None):
    """绘制训练曲线"""
    
    epochs = log_data['epochs']
    train_loss = log_data['train_loss']
    val_loss = log_data['val_loss']
    fid_scores = log_data['fid_scores']
    learning_rates = log_data['learning_rates']
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LDM训练监控', fontsize=16, fontweight='bold')
    
    # 1. 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, label='训练损失', color='blue', linewidth=2)
    ax1.plot(epochs, val_loss, label='验证损失', color='red', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.set_title('训练和验证损失')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. FID分数曲线
    ax2 = axes[0, 1]
    # 过滤掉None值
    fid_epochs = [epochs[i] for i, fid in enumerate(fid_scores) if fid is not None]
    fid_values = [fid for fid in fid_scores if fid is not None]
    
    if fid_values:
        ax2.plot(fid_epochs, fid_values, label='FID分数', color='green', 
                linewidth=2, marker='o', markersize=6)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('FID分数')
        ax2.set_title('FID分数变化 (越低越好)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 标注最佳FID
        if fid_values:
            best_fid = min(fid_values)
            best_epoch = fid_epochs[fid_values.index(best_fid)]
            ax2.annotate(f'最佳FID: {best_fid:.2f}\nEpoch: {best_epoch}', 
                        xy=(best_epoch, best_fid), xytext=(10, 10),
                        textcoords='offset points', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    else:
        ax2.text(0.5, 0.5, '暂无FID数据', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('FID分数变化 (越低越好)')
    
    # 3. 学习率曲线
    ax3 = axes[1, 0]
    ax3.plot(epochs, learning_rates, label='学习率', color='orange', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('学习率')
    ax3.set_title('学习率调度')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. 统计摘要
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 准备统计信息
    stats_text = []
    stats_text.append(f"训练轮数: {len(epochs)}")
    stats_text.append(f"最终训练损失: {train_loss[-1]:.4f}")
    stats_text.append(f"最终验证损失: {val_loss[-1]:.4f}")
    stats_text.append(f"最佳验证损失: {min(val_loss):.4f} (Epoch {epochs[val_loss.index(min(val_loss))]})")
    
    if fid_values:
        stats_text.append(f"最佳FID分数: {min(fid_values):.2f}")
        stats_text.append(f"最新FID分数: {fid_values[-1]:.2f}")
        stats_text.append(f"FID评估次数: {len(fid_values)}")
    
    stats_text.append(f"当前学习率: {learning_rates[-1]:.2e}")
    
    # 显示统计信息
    for i, text in enumerate(stats_text):
        ax4.text(0.1, 0.9 - i * 0.1, text, fontsize=11, 
                transform=ax4.transAxes, verticalalignment='top')
    
    ax4.set_title('训练统计', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存: {save_path}")
    
    plt.show()

def plot_fid_analysis(log_data: Dict, save_path: Optional[str] = None):
    """详细的FID分析图"""
    
    epochs = log_data['epochs']
    train_loss = log_data['train_loss']
    fid_scores = log_data['fid_scores']
    
    # 过滤数据
    fid_epochs = [epochs[i] for i, fid in enumerate(fid_scores) if fid is not None]
    fid_values = [fid for fid in fid_scores if fid is not None]
    fid_train_loss = [train_loss[i] for i, fid in enumerate(fid_scores) if fid is not None]
    
    if not fid_values:
        print("没有FID数据可供分析")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. FID vs Epoch 详细分析
    ax1.plot(fid_epochs, fid_values, 'g-o', linewidth=2, markersize=8, label='FID分数')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('FID分数')
    ax1.set_title('FID分数详细变化')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 添加趋势线
    if len(fid_values) > 1:
        z = np.polyfit(fid_epochs, fid_values, 1)
        p = np.poly1d(z)
        ax1.plot(fid_epochs, p(fid_epochs), "r--", alpha=0.8, label=f'趋势线 (斜率: {z[0]:.3f})')
        ax1.legend()
    
    # 2. FID vs 训练损失 相关性分析
    ax2.scatter(fid_train_loss, fid_values, alpha=0.7, s=100, c=fid_epochs, cmap='viridis')
    ax2.set_xlabel('训练损失')
    ax2.set_ylabel('FID分数')
    ax2.set_title('FID分数 vs 训练损失')
    ax2.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Epoch')
    
    # 计算相关系数
    if len(fid_values) > 1:
        correlation = np.corrcoef(fid_train_loss, fid_values)[0, 1]
        ax2.text(0.05, 0.95, f'相关系数: {correlation:.3f}', 
                transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"FID分析图已保存: {save_path}")
    
    plt.show()

def print_training_summary(log_data: Dict):
    """打印训练摘要"""
    
    epochs = log_data['epochs']
    train_loss = log_data['train_loss']
    val_loss = log_data['val_loss']
    fid_scores = log_data['fid_scores']
    
    print("=" * 60)
    print("🎯 LDM训练摘要")
    print("=" * 60)
    
    print(f"📊 基本统计:")
    print(f"  • 总训练轮数: {len(epochs)}")
    print(f"  • 最终训练损失: {train_loss[-1]:.4f}")
    print(f"  • 最终验证损失: {val_loss[-1]:.4f}")
    print(f"  • 最佳验证损失: {min(val_loss):.4f} (Epoch {epochs[val_loss.index(min(val_loss))]})")
    
    # 损失改善情况
    loss_improvement = ((train_loss[0] - train_loss[-1]) / train_loss[0]) * 100
    print(f"  • 损失改善: {loss_improvement:.1f}%")
    
    # FID统计
    fid_values = [fid for fid in fid_scores if fid is not None]
    if fid_values:
        print(f"\n🎨 FID评估:")
        print(f"  • FID评估次数: {len(fid_values)}")
        print(f"  • 最佳FID分数: {min(fid_values):.2f}")
        print(f"  • 最新FID分数: {fid_values[-1]:.2f}")
        
        if len(fid_values) > 1:
            fid_improvement = ((fid_values[0] - fid_values[-1]) / fid_values[0]) * 100
            print(f"  • FID改善: {fid_improvement:.1f}%")
    else:
        print(f"\n🎨 FID评估: 暂无数据")
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="LDM训练日志可视化")
    parser.add_argument("--log_dir", type=str, default="ldm_models/logs", 
                       help="训练日志目录")
    parser.add_argument("--save_plots", action="store_true", 
                       help="是否保存图表")
    parser.add_argument("--output_dir", type=str, default="training_plots", 
                       help="图表保存目录")
    
    args = parser.parse_args()
    
    # 加载训练日志
    log_path = os.path.join(args.log_dir, "training_log.json")
    
    if not os.path.exists(log_path):
        print(f"❌ 找不到训练日志: {log_path}")
        return
    
    print(f"📈 加载训练日志: {log_path}")
    log_data = load_training_log(log_path)
    
    # 打印摘要
    print_training_summary(log_data)
    
    # 创建输出目录
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 绘制训练曲线
    plot_save_path = os.path.join(args.output_dir, "training_curves.png") if args.save_plots else None
    plot_training_curves(log_data, plot_save_path)
    
    # 绘制FID分析
    fid_save_path = os.path.join(args.output_dir, "fid_analysis.png") if args.save_plots else None
    plot_fid_analysis(log_data, fid_save_path)

if __name__ == "__main__":
    main() 