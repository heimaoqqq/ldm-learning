#!/usr/bin/env python3
"""
VQ-VAE训练主脚本
专门用于256*256彩色图像数据集训练
"""

import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf
import warnings

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vqvae_model import create_vqvae_model
from dataset import create_dataloaders

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

def setup_logging():
    """设置日志记录"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_environment():
    """检查训练环境"""
    logger = setup_logging()
    
    logger.info("=== 环境检查 ===")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"PyTorch Lightning版本: {pl.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        logger.warning("CUDA不可用，将使用CPU训练 (速度会很慢)")
    
    return logger

def load_config(config_path: str = None):
    """加载配置文件"""
    if config_path is None:
        config_path = "configs/vqvae_256_config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    config = OmegaConf.load(config_path)
    return config

def setup_data(config, logger):
    """设置数据加载器"""
    logger.info("=== 设置数据加载器 ===")
    
    data_config = config.data.params
    
    # 检查数据路径
    if not os.path.exists(data_config.data_path):
        raise FileNotFoundError(f"数据路径不存在: {data_config.data_path}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(**data_config)
    
    logger.info(f"训练数据: {len(train_loader.dataset)} 张图像")
    logger.info(f"验证数据: {len(val_loader.dataset)} 张图像") 
    logger.info(f"测试数据: {len(test_loader.dataset)} 张图像")
    logger.info(f"批次大小: {data_config.batch_size}")
    
    return train_loader, val_loader, test_loader

def setup_model(config, logger):
    """设置模型"""
    logger.info("=== 设置VQ-VAE模型 ===")
    
    model_config = config.model.params
    model = create_vqvae_model(**model_config)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"模型总参数: {total_params / 1e6:.2f}M")
    logger.info(f"可训练参数: {trainable_params / 1e6:.2f}M")
    logger.info(f"码本大小: {model_config.n_embed}")
    logger.info(f"嵌入维度: {model_config.embed_dim}")
    
    return model

def setup_callbacks(config, logger):
    """设置回调函数"""
    logger.info("=== 设置训练回调 ===")
    
    callbacks = []
    
    # 模型检查点
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="vqvae-{epoch:02d}-{val_rec_loss:.4f}",
        monitor="val/rec_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=5,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    logger.info("添加模型检查点回调")
    
    # 早停
    early_stopping = EarlyStopping(
        monitor="val/rec_loss",
        mode="min", 
        patience=15,
        verbose=True
    )
    callbacks.append(early_stopping)
    logger.info("添加早停回调 (patience=15)")
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    logger.info("添加学习率监控")
    
    return callbacks

def setup_logger_and_trainer(config, callbacks, logger):
    """设置记录器和训练器"""
    logger.info("=== 设置训练器 ===")
    
    # TensorBoard记录器
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name="vqvae_256",
        version=None
    )
    logger.info("使用TensorBoard记录器")
    
    # 训练器配置
    trainer_config = config.trainer
    
    # 根据可用GPU调整配置
    if not torch.cuda.is_available():
        trainer_config.accelerator = "cpu"
        trainer_config.devices = 1
        logger.warning("使用CPU训练，建议使用GPU以获得更好性能")
    
    trainer = pl.Trainer(
        accelerator=trainer_config.accelerator,
        devices=trainer_config.devices,
        precision=trainer_config.precision,
        max_epochs=trainer_config.max_epochs,
        check_val_every_n_epoch=trainer_config.check_val_every_n_epoch,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        gradient_clip_val=1.0,  # 梯度裁剪
        log_every_n_steps=trainer_config.log_every_n_steps,
        enable_progress_bar=trainer_config.enable_progress_bar,
        enable_checkpointing=trainer_config.enable_checkpointing,
        benchmark=trainer_config.benchmark,
        deterministic=trainer_config.deterministic,
        callbacks=callbacks,
        logger=tb_logger,
        default_root_dir="logs"
    )
    
    logger.info(f"训练器配置:")
    logger.info(f"  设备: {trainer_config.accelerator}")
    logger.info(f"  精度: {trainer_config.precision}")
    logger.info(f"  最大轮数: {trainer_config.max_epochs}") 
    logger.info(f"  梯度累积: {trainer_config.accumulate_grad_batches}")
    
    return trainer, tb_logger

def train_model(
    config_path: str = None,
    resume_from_checkpoint: str = None,
    test_only: bool = False
):
    """训练VQ-VAE模型"""
    
    # 环境检查
    logger = check_environment()
    
    try:
        # 加载配置
        config = load_config(config_path)
        logger.info(f"加载配置文件: {config_path or 'configs/vqvae_256_config.yaml'}")
        
        # 设置数据
        train_loader, val_loader, test_loader = setup_data(config, logger)
        
        # 设置模型
        model = setup_model(config, logger)
        
        # 设置回调
        callbacks = setup_callbacks(config, logger) 
        
        # 设置训练器
        trainer, tb_logger = setup_logger_and_trainer(config, callbacks, logger)
        
        if test_only:
            # 仅测试模式
            logger.info("=== 开始测试 ===")
            if resume_from_checkpoint:
                logger.info(f"从检查点加载: {resume_from_checkpoint}")
            trainer.test(model, test_loader, ckpt_path=resume_from_checkpoint)
        else:
            # 训练模式
            logger.info("=== 开始训练 ===")
            
            if resume_from_checkpoint:
                logger.info(f"从检查点恢复训练: {resume_from_checkpoint}")
            
            # 开始训练
            trainer.fit(
                model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=resume_from_checkpoint
            )
            
            # 训练完成后测试
            logger.info("=== 训练完成，开始测试 ===")
            trainer.test(model, test_loader, ckpt_path="best")
        
        logger.info("✅ 训练/测试完成!")
        logger.info(f"日志文件: {tb_logger.log_dir}")
        logger.info(f"检查点: checkpoints/")
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VQ-VAE训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (默认: configs/vqvae_256_config.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="仅测试模式"
    )
    
    args = parser.parse_args()
    
    # 创建必要目录
    os.makedirs("data", exist_ok=True)
    os.makedirs("configs", exist_ok=True) 
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # 开始训练
    train_model(
        config_path=args.config,
        resume_from_checkpoint=args.resume,
        test_only=args.test_only
    )

if __name__ == "__main__":
    main() 
