#!/usr/bin/env python3
"""
训练VAE (AutoencoderKL) for Oxford-IIIT Pet Dataset
使用方法: python VAE/train_vae.py --config VAE/vae_config.yaml
"""

import os
import sys
import argparse
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# 添加latent-diffusion到路径
sys.path.insert(0, 'latent-diffusion')

from main import instantiate_from_config
from main import DataModuleFromConfig


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="VAE/vae_config.yaml",
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
        default="VAE/logs",
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
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 创建模型
    model = instantiate_from_config(config.model)
    
    # 创建数据模块
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    
    # 创建检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.logdir, "checkpoints"),
        filename="vae-pet-{epoch:02d}-{val_loss:.2f}",
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
    
    # 创建训练器
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback] + list(instantiate_from_config(config.lightning.callbacks).values()),
        **config.lightning.trainer,
        default_root_dir=args.logdir,
    )
    
    # 开始训练
    if args.resume:
        trainer.fit(model, data, ckpt_path=args.resume)
    else:
        trainer.fit(model, data)
    
    print(f"训练完成! 检查点保存在: {args.logdir}/checkpoints/")
    print(f"最佳模型: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main() 
