#!/usr/bin/env python3
"""
在Kaggle上训练VAE模型
为P100 GPU (16GB显存)优化
"""

import os
import sys
import argparse
import yaml
import time
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# 添加latent-diffusion到系统路径
LATENT_DIFFUSION_PATH = "/kaggle/working/latent-diffusion"
sys.path.append(LATENT_DIFFUSION_PATH)

# 导入VAE模块
from VAE.pet_dataset import PetDatasetTrain, PetDatasetValidation

# 从ldm模块导入
try:
    from ldm.util import instantiate_from_config
    from ldm.models.autoencoder import AutoencoderKL
except ImportError:
    print(f"❌ 无法导入ldm模块，请确认latent-diffusion路径: {LATENT_DIFFUSION_PATH}")
    sys.exit(1)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default="/kaggle/working/VAE/vae_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="/kaggle/input/the-oxfordiiit-pet-dataset/images",
        help="数据集根目录"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64,
        help="批次大小"
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        default=256,
        help="图像大小"
    )
    parser.add_argument(
        "--max_epochs", 
        type=int, 
        default=100,
        help="最大训练轮数"
    )
    parser.add_argument(
        "--precision", 
        type=str, 
        default="16-mixed",
        help="训练精度，可选: 32, 16-mixed"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="恢复训练的检查点路径"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--logdir", 
        type=str, 
        default="/kaggle/working/VAE/logs",
        help="日志目录"
    )
    return parser

def main():
    """主函数"""
    parser = get_parser()
    args = parser.parse_args()
    
    # 设置随机种子
    pl.seed_everything(args.seed)
    
    # 创建输出目录
    os.makedirs(args.logdir, exist_ok=True)
    checkpoint_dir = os.path.join(args.logdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 修改配置
    config.data.params.batch_size = args.batch_size
    if "size" in config.data.params.train.params:
        config.data.params.train.params.size = args.image_size
    if "size" in config.data.params.validation.params:
        config.data.params.validation.params.size = args.image_size
    
    # 设置数据集路径
    config.data.params.train.params.data_root = args.data_root
    config.data.params.validation.params.data_root = args.data_root
    
    # 打印训练信息
    print("=" * 60)
    print("🚀 开始训练VAE (Kaggle P100 GPU优化)")
    print("=" * 60)
    print(f"数据集路径: {args.data_root}")
    print(f"配置文件: {args.config}")
    print(f"批次大小: {args.batch_size}")
    print(f"图像大小: {args.image_size}x{args.image_size}")
    print(f"训练轮数: {args.max_epochs}")
    print(f"精度: {args.precision}")
    print(f"输出目录: {args.logdir}")
    
    # 检查GPU
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"使用GPU: {device_name} ({vram:.1f}GB)")
    else:
        print("⚠️ 未检测到GPU，将使用CPU训练（非常慢）")
    
    # 创建模型
    model = instantiate_from_config(config.model)
    print(f"模型创建成功: {type(model).__name__}")
    
    # 创建数据模块
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print(f"数据模块创建成功，训练集大小: {len(data.datasets['train'])}, 验证集大小: {len(data.datasets['validation'])}")
    
    # 创建回调
    callbacks = []
    
    # 添加检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="vae-{epoch:02d}-{val/rec_loss:.4f}",
        save_top_k=3,
        monitor="val/rec_loss",
        mode="min",
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # 添加学习率监控
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # 添加图像记录器
    if "image_logger" in config.lightning.callbacks:
        image_logger_config = config.lightning.callbacks.image_logger
        image_logger = instantiate_from_config(image_logger_config)
        callbacks.append(image_logger)
    
    # 创建日志器
    logger = TensorBoardLogger(
        save_dir=args.logdir,
        name="",
    )
    
    # 训练器配置
    trainer_config = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1 if torch.cuda.is_available() else 0,
        "precision": args.precision,
        "max_epochs": args.max_epochs,
        "log_every_n_steps": 50,
        "num_sanity_val_steps": 2,
        "check_val_every_n_epoch": 5,
        "benchmark": True,
    }
    
    # 添加配置文件中的其他配置
    if "trainer" in config.lightning:
        for k, v in config.lightning.trainer.items():
            if k not in trainer_config:
                trainer_config[k] = v
    
    # 创建训练器
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **trainer_config,
    )
    
    # 开始训练
    print(f"开始训练...")
    start_time = time.time()
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"从检查点恢复训练: {args.resume}")
            trainer.fit(model, data, ckpt_path=args.resume)
        else:
            print(f"❌ 检查点不存在: {args.resume}")
            sys.exit(1)
    else:
        trainer.fit(model, data)
    
    # 计算训练时间
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"训练完成! 耗时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    
    # 打印最佳模型路径
    if hasattr(checkpoint_callback, "best_model_path") and checkpoint_callback.best_model_path:
        print(f"最佳模型: {checkpoint_callback.best_model_path}")
        print(f"最佳性能: {checkpoint_callback.best_model_score:.4f}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.logdir, "vae_final.ckpt")
    trainer.save_checkpoint(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")

if __name__ == "__main__":
    main() 
