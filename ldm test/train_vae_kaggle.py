"""
针对宠物数据集的VAE训练启动脚本
使用微调后的stable diffusion VAE组件
"""
import os
import sys
import argparse
from VAE.vae_finetuning_pet import PetVAEFineTuner

def parse_args():
    parser = argparse.ArgumentParser(description="训练VAE用于宠物图像生成")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="/kaggle/input/the-oxfordiiit-pet-dataset/images",
        help="数据集路径"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="批次大小"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=25, 
        help="训练轮数"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-5, 
        help="学习率"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="./vae_finetuned", 
        help="模型保存路径"
    )
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 检查数据集路径
    if not os.path.exists(args.data_dir):
        print(f"⚠️ 数据路径 {args.data_dir} 不存在，尝试查找本地数据集...")
        # 尝试本地路径
        local_path = "./data/oxford-pets/images"
        if os.path.exists(local_path):
            args.data_dir = local_path
            print(f"✅ 找到本地数据集: {local_path}")
        else:
            print(f"❌ 未找到数据集，请检查路径")
            return
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建并配置VAE微调器
    finetuner = PetVAEFineTuner(
        data_dir=args.data_dir
    )
    
    # 更新配置
    finetuner.config.update({
        'batch_size': args.batch_size,
        'max_epochs': args.epochs,
        'learning_rate': args.lr,
        'save_dir': args.save_dir
    })
    
    # 开始训练
    print(f"🚀 开始训练VAE，数据集: {args.data_dir}")
    print(f"📊 批次大小: {args.batch_size}, 训练轮数: {args.epochs}")
    
    best_loss = finetuner.finetune()
    
    print(f"✅ 训练完成! 最佳重建损失: {best_loss:.4f}")
    print(f"💾 模型已保存到: {args.save_dir}")

if __name__ == "__main__":
    main() 
