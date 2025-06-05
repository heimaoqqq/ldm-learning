#!/usr/bin/env python3
"""
在Kaggle环境中训练VAE (AutoencoderKL) for Oxford-IIIT Pet Dataset
使用方法: python VAE/train_vae_kaggle.py
"""

import os
import sys
import argparse
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Kaggle环境路径设置
sys.path.append('/kaggle/working')
sys.path.append('/kaggle/working/latent-diffusion')

# 导入必要模块
try:
    # 尝试从latent-diffusion导入
    from main import instantiate_from_config, DataModuleFromConfig
except ImportError:
    # 备选导入路径
    sys.path.insert(0, '/kaggle/working/latent-diffusion')
    from main import instantiate_from_config, DataModuleFromConfig


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/kaggle/working/VAE/vae_config.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to checkpoint to resume from",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="/kaggle/working/VAE/logs",
        help="directory for logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # 设置随机种子
    pl.seed_everything(args.seed)
    
    # 创建输出目录
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(os.path.join(args.logdir, "checkpoints"), exist_ok=True)
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 打印Kaggle环境信息
    print("Kaggle环境信息:")
    print(f"数据集路径: /kaggle/input/the-oxfordiiit-pet-dataset")
    print(f"工作目录: /kaggle/working")
    print(f"输出目录: {args.logdir}")
    
    # 检查数据集
    dataset_path = "/kaggle/input/the-oxfordiiit-pet-dataset"
    if os.path.exists(dataset_path):
        print(f"✓ 数据集存在: {dataset_path}")
        files = os.listdir(dataset_path)
        print(f"数据集内容: {files[:10]}...")  # 显示前10个文件
    else:
        print(f"✗ 数据集不存在: {dataset_path}")
        return
    
    # 创建模型
    model = instantiate_from_config(config.model)
    
    # 创建数据模块
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    
    # 创建检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.logdir, "checkpoints"),
        filename="vae-pet-{epoch:02d}-{val_rec_loss:.4f}",
        save_top_k=3,
        monitor="val/rec_loss",
        mode="min",
        save_last=True,
        every_n_epochs=5,
    )
    
    # 创建日志器
    logger = TensorBoardLogger(
        save_dir=args.logdir,
        name="vae_pet_training"
    )
    
    # Kaggle环境优化的训练器设置
    trainer_config = config.lightning.trainer.copy()
    trainer_config.update({
        'gpus': 1 if torch.cuda.is_available() else 0,
        'precision': 16,  # 使用半精度以节省内存
        'gradient_clip_val': 1.0,  # 梯度裁剪
        'log_every_n_steps': 50,
        'val_check_interval': 0.5,  # 每半个epoch验证一次
    })
    
    # 创建训练器
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback] + list(instantiate_from_config(config.lightning.callbacks).values()),
        **trainer_config,
        default_root_dir=args.logdir,
    )
    
    # 开始训练
    print("开始VAE训练...")
    if args.resume:
        trainer.fit(model, data, ckpt_path=args.resume)
    else:
        trainer.fit(model, data)
    
    print(f"VAE训练完成! 检查点保存在: {args.logdir}/checkpoints/")
    print(f"最佳模型: {checkpoint_callback.best_model_path}")
    
    # 保存最终模型到Kaggle输出
    final_model_path = "/kaggle/working/vae_final.ckpt"
    if checkpoint_callback.best_model_path:
        import shutil
        shutil.copy2(checkpoint_callback.best_model_path, final_model_path)
        print(f"最佳模型已复制到: {final_model_path}")


if __name__ == "__main__":
    import torch
    main() 
