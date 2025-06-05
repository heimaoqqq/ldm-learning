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
import importlib
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 添加kaggle工作目录到路径
kaggle_working = "/kaggle/working"
if os.path.exists(kaggle_working) and kaggle_working not in sys.path:
    sys.path.insert(0, kaggle_working)

# 添加latent-diffusion到系统路径
LATENT_DIFFUSION_PATH = "/kaggle/working/latent-diffusion"
sys.path.append(LATENT_DIFFUSION_PATH)

# 导入VAE模块
try:
    from pet_dataset import PetDatasetTrain, PetDatasetValidation
except ImportError as e:
    print(f"❌ 无法导入pet_dataset模块: {e}")
    print(f"搜索路径: {sys.path}")
    sys.exit(1)

# 从ldm模块导入
try:
    from ldm.util import instantiate_from_config
    from ldm.models.autoencoder import AutoencoderKL
except ImportError as e:
    print(f"❌ 无法导入ldm模块: {e}")
    print(f"尝试从 {LATENT_DIFFUSION_PATH} 导入")
    print(f"搜索路径: {sys.path}")
    
    # 尝试检查ldm目录内容
    ldm_dir = os.path.join(LATENT_DIFFUSION_PATH, "ldm")
    if os.path.exists(ldm_dir):
        print(f"ldm目录存在，内容:")
        for root, dirs, files in os.walk(ldm_dir, topdown=True, maxdepth=2):
            print(f"- {root}")
            for f in files:
                if f.endswith(".py"):
                    print(f"  - {f}")
    else:
        print(f"ldm目录不存在: {ldm_dir}")
    
    sys.exit(1)

# 确保main模块存在
try:
    import main
except ImportError:
    print("⚠️ 无法导入main模块，尝试创建默认实现...")
    
    # 添加必要的main函数到当前模块
    def instantiate_from_config(config):
        if not "target" in config:
            raise KeyError("Expected key `target` to instantiate.")
        return get_obj_from_str(config["target"])(**config.get("params", dict()))

    def get_obj_from_str(string, reload=False):
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(module, package=None), cls)

    class DataModuleFromConfig(pl.LightningDataModule):
        def __init__(self, batch_size, train=None, validation=None, test=None,
                    wrap=False, num_workers=None):
            super().__init__()
            self.batch_size = batch_size
            self.dataset_configs = dict()
            self.num_workers = num_workers if num_workers is not None else batch_size*2
            if train is not None:
                self.dataset_configs["train"] = train
                self.train_dataloader = self._train_dataloader
            if validation is not None:
                self.dataset_configs["validation"] = validation
                self.val_dataloader = self._val_dataloader

            self.wrap = wrap

        def prepare_data(self):
            self.datasets = dict(
                (k, instantiate_from_config(self.dataset_configs[k]))
                for k in self.dataset_configs)

        def setup(self, stage=None):
            if self.wrap:
                for k in self.datasets:
                    self.datasets[k] = WrappedDataset(self.datasets[k])

        def _train_dataloader(self):
            return torch.utils.data.DataLoader(
                self.datasets["train"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True
            )

        def _val_dataloader(self):
            return torch.utils.data.DataLoader(
                self.datasets["validation"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False
            )

    class ImageLogger:
        def __init__(self, batch_frequency=1000, max_images=8, increase_log_steps=True):
            self.batch_freq = batch_frequency
            self.max_images = max_images
            self.log_steps = [batch_frequency]
            self.increase_log_steps = increase_log_steps
            
        def __call__(self, pl_module, batch, batch_idx, split="train"):
            pass  # 简化版实现

    # 添加到main模块
    main = type('main', (), {})
    main.instantiate_from_config = instantiate_from_config
    main.DataModuleFromConfig = DataModuleFromConfig
    main.ImageLogger = ImageLogger
    sys.modules['main'] = main
    
    print("✓ 已创建main模块的默认实现")

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
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="启用调试模式"
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
    
    # 如果是调试模式，输出更多信息
    if args.debug:
        print("\n调试信息:")
        print(f"Python路径: {sys.path}")
        print(f"当前目录: {os.getcwd()}")
        print(f"latent-diffusion路径是否存在: {os.path.exists(LATENT_DIFFUSION_PATH)}")
        print(f"配置文件是否存在: {os.path.exists(args.config)}")
        print(f"数据目录是否存在: {os.path.exists(args.data_root)}")
        
        # 尝试列出数据目录内容
        if os.path.exists(args.data_root):
            print(f"\n数据目录内容 ({args.data_root}):")
            try:
                files = os.listdir(args.data_root)
                print(f"发现 {len(files)} 个文件/目录")
                print(f"前10个: {files[:10]}")
            except Exception as e:
                print(f"列出目录内容时出错: {e}")
        else:
            print(f"数据目录不存在: {args.data_root}")
    
    # 加载配置
    try:
        config = OmegaConf.load(args.config)
    except Exception as e:
        print(f"❌ 加载配置文件出错: {e}")
        print(f"检查配置文件路径: {args.config}")
        sys.exit(1)
    
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
    try:
        model = instantiate_from_config(config.model)
        print(f"模型创建成功: {type(model).__name__}")
    except Exception as e:
        print(f"❌ 创建模型失败: {e}")
        print("这可能是由于配置文件与ldm实现不匹配导致的")
        print("请检查latent-diffusion库的版本和配置文件的兼容性")
        sys.exit(1)
    
    # 创建数据模块
    try:
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()
        print(f"数据模块创建成功，训练集大小: {len(data.datasets['train'])}, 验证集大小: {len(data.datasets['validation'])}")
    except Exception as e:
        print(f"❌ 创建数据模块失败: {e}")
        print("这可能是由于数据路径错误或格式不正确导致的")
        sys.exit(1)
    
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
        try:
            image_logger_config = config.lightning.callbacks.image_logger
            image_logger = instantiate_from_config(image_logger_config)
            callbacks.append(image_logger)
        except Exception as e:
            print(f"⚠️ 创建图像记录器失败: {e}")
            print("将继续训练，但不会记录重建图像")
    
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
    
    try:
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
    
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 
