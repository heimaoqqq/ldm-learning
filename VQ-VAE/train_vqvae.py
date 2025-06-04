#!/usr/bin/env python3
"""
基于官方taming-transformers的VQ-VAE训练脚本
针对Kaggle P100 16GB GPU环境优化
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from pathlib import Path

# 添加taming-transformers到路径
sys.path.insert(0, str(Path("../taming-transformers-master").resolve()))

def main():
    print("🚀 开始VQ-VAE训练 (基于官方taming-transformers)")
    print("=" * 70)
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ 未检测到GPU")
        return
        
    print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # 导入taming模块
    try:
        from taming.models.vqgan import VQModel
        print("✅ 成功导入taming.models.vqgan.VQModel")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请检查taming-transformers-master路径")
        return
    
    # 导入自定义数据模块
    try:
        from kaggle_dataset import KaggleDataModule
        print("✅ 成功导入KaggleDataModule")
    except ImportError as e:
        print(f"❌ 导入kaggle_dataset失败: {e}")
        return
    
    # 加载配置
    config_path = "configs/kaggle_p100_vqgan.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
        
    config = OmegaConf.load(config_path)
    print("✅ 配置加载成功")
    
    # 创建数据模块
    try:
        data_module = KaggleDataModule(
            data_path="/kaggle/input/dataset",
            batch_size=12,  # P100优化批次大小
            num_workers=2,
            image_size=256
        )
        print("✅ 数据模块创建成功")
    except Exception as e:
        print(f"❌ 数据模块创建失败: {e}")
        return
    
    # 创建模型
    try:
        model = VQModel(**config.model.params)
        model.learning_rate = config.model.base_learning_rate
        print(f"✅ VQ-GAN模型创建成功")
        
        # 计算模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数: {total_params/1e6:.2f}M")
        print(f"可训练参数: {trainable_params/1e6:.2f}M")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return
    
    # 创建回调函数
    callbacks = []
    
    # 检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="vqvae-{epoch:02d}-{val_rec_loss:.4f}",
        monitor="val/rec_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=5,
        save_weights_only=False
    )
    callbacks.append(checkpoint_callback)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # 创建日志记录器
    logger = TensorBoardLogger(
        save_dir="logs",
        name="vqvae_training",
        version=None
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=16,  # 混合精度
        max_epochs=100,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        check_val_every_n_epoch=2,
        accumulate_grad_batches=2,  # 梯度累积
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        benchmark=True
    )
    
    print("✅ 训练器创建成功")
    
    # 开始训练
    try:
        print("\n🚀 开始训练...")
        print("📊 训练配置:")
        print(f"  - 批次大小: 12 (有效: 24)")
        print(f"  - 学习率: {config.model.base_learning_rate}")
        print(f"  - 码本大小: {config.model.params.n_embed}")
        print(f"  - 最大epochs: 100")
        print(f"  - 混合精度: 16-bit")
        
        trainer.fit(model, data_module)
        print("\n✅ 训练完成!")
        
        # 保存最终模型
        final_path = "checkpoints/final_vqvae_model.ckpt"
        trainer.save_checkpoint(final_path)
        print(f"✅ 最终模型已保存: {final_path}")
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
