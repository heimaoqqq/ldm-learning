"""
基于taming-transformers的VQ-VAE模型实现
专门适配256*256彩色图像训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf
from einops import rearrange
import numpy as np
import os

try:
    from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
    from taming.modules.diffusionmodules.model import Encoder, Decoder
    from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
except ImportError:
    print("警告: taming-transformers未安装，请先运行 setup_environment.py")
    # 提供简化的替代实现
    VectorQuantizer = None
    Encoder = None 
    Decoder = None
    VQLPIPSWithDiscriminator = None

class VQVAEModel(pl.LightningModule):
    """VQ-VAE模型 - 基于taming-transformers"""
    
    def __init__(
        self,
        # 编码器参数
        n_embed: int = 16384,  # 码本大小
        embed_dim: int = 256,  # 嵌入维度
        ch: int = 128,  # 基础通道数
        ch_mult: list = [1, 1, 2, 2, 4],  # 通道倍数
        num_res_blocks: int = 2,  # 残差块数量
        attn_resolutions: list = [16],  # 注意力分辨率
        dropout: float = 0.0,
        resolution: int = 256,  # 输入分辨率
        in_channels: int = 3,  # 输入通道数
        out_ch: int = 3,  # 输出通道数
        z_channels: int = 256,  # 潜在空间通道数
        # 量化器参数
        beta: float = 0.25,  # 承诺损失权重
        # 训练参数
        learning_rate: float = 4.5e-6,
        lr_g_factor: float = 1.0,
        # 损失函数参数
        disc_conditional: bool = False,
        disc_in_channels: int = 3,
        disc_start: int = 50001,
        disc_weight: float = 0.8,
        codebook_weight: float = 1.0,
        pixelloss_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        # 其他参数
        ckpt_path: str = None,
        ignore_keys: list = [],
        image_key: str = "image",
        colorize_nlabels: int = None,
        monitor: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.image_key = image_key
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.learning_rate = learning_rate
        self.lr_g_factor = lr_g_factor
        
        # 初始化编码器和解码器
        self.encoder = Encoder(
            ch=ch,
            out_ch=z_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=True,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
            double_z=False
        )
        
        self.decoder = Decoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=True,
            in_channels=z_channels,
            resolution=resolution,
            z_channels=z_channels
        )
        
        # 初始化向量量化器
        self.quantize = VectorQuantizer(
            n_embed, embed_dim, beta=beta
        )
        
        # 量化层的卷积
        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)
        
        # 损失函数
        if VQLPIPSWithDiscriminator is not None:
            lossconfig = OmegaConf.create({
                "target": "taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
                "params": {
                    "disc_conditional": disc_conditional,
                    "disc_in_channels": disc_in_channels,
                    "disc_start": disc_start,
                    "disc_weight": disc_weight,
                    "codebook_weight": codebook_weight,
                    "pixelloss_weight": pixelloss_weight,
                    "perceptual_weight": perceptual_weight
                }
            })
            self.loss = VQLPIPSWithDiscriminator(**lossconfig.params)
        else:
            self.loss = None
        
        # 加载预训练权重
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        # 用于监控的指标
        self.monitor = monitor
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        """从检查点加载权重"""
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"从 {path} 恢复模型")
    
    def encode(self, x):
        """编码"""
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info
    
    def decode(self, quant):
        """解码"""
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
    
    def decode_code(self, code_b):
        """从码本索引解码"""
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec
    
    def forward(self, input, return_pred_indices=False):
        """前向传播"""
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff
    
    def get_input(self, batch, k):
        """获取输入"""
        x = batch[k] if isinstance(batch, dict) else batch
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        """训练步骤"""
        # 处理输入
        x = self.get_input(batch, self.image_key) if isinstance(batch, dict) else batch
        xrec, qloss, ind = self(x, return_pred_indices=True)
        
        if optimizer_idx == 0:
            # 训练生成器
            aeloss, log_dict_ae = self.loss(
                qloss, x, xrec, optimizer_idx, self.global_step,
                last_layer=self.get_last_layer(), split="train",
                predicted_indices=ind
            )
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss
        
        if optimizer_idx == 1:
            # 训练判别器
            discloss, log_dict_disc = self.loss(
                qloss, x, xrec, optimizer_idx, self.global_step,
                last_layer=self.get_last_layer(), split="train"
            )
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
            log_dict.update(log_dict_ema)
        return log_dict
    
    def _validation_step(self, batch, batch_idx, suffix=""):
        """验证步骤实现"""
        x = self.get_input(batch, self.image_key) if isinstance(batch, dict) else batch
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(
            qloss, x, xrec, 0, self.global_step,
            last_layer=self.get_last_layer(), split="val"+suffix,
            predicted_indices=ind
        )
        
        discloss, log_dict_disc = self.loss(
            qloss, x, xrec, 1, self.global_step,
            last_layer=self.get_last_layer(), split="val"+suffix,
            predicted_indices=ind
        )
        
        log_dict_ae.update(log_dict_disc)
        self.log_dict(log_dict_ae)
        return log_dict_ae
    
    def configure_optimizers(self):
        """配置优化器"""
        lr = self.learning_rate
        
        # 生成器参数
        ae_params_list = list(self.encoder.parameters()) + \
                        list(self.decoder.parameters()) + \
                        list(self.quantize.parameters()) + \
                        list(self.quant_conv.parameters()) + \
                        list(self.post_quant_conv.parameters())
        
        # 判别器参数
        disc_params = list(self.loss.discriminator.parameters())
        
        # 优化器
        opt_ae = torch.optim.Adam(ae_params_list, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(disc_params, lr=lr, betas=(0.5, 0.9))
        
        return [opt_ae, opt_disc], []
    
    def get_last_layer(self):
        """获取最后一层权重"""
        return self.decoder.conv_out.weight
    
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, log_keys=None, **kwargs):
        """记录图像用于可视化"""
        log = dict()
        x = self.get_input(batch, self.image_key) if isinstance(batch, dict) else batch
        x = x.to(self.device)
        
        if not only_inputs:
            xrec, _ = self(x)
            if x.shape[1] > 3:
                # 彩色图像
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["reconstructions"] = xrec
        
        log["inputs"] = x
        return log
    
    def to_rgb(self, x):
        """转换为RGB"""
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

# 简化版本的VQ-VAE（如果taming-transformers未安装）
class SimpleVQVAE(pl.LightningModule):
    """简化版VQ-VAE实现"""
    
    def __init__(self, **kwargs):
        super().__init__()
        print("使用简化版VQ-VAE实现")
        print("建议安装taming-transformers以获得最佳性能")
        
        # 这里可以实现一个简化版本
        # 或者提示用户安装完整版本
        
    def forward(self, x):
        return x, torch.tensor(0.0)

def create_vqvae_model(config_path: str = None, **kwargs):
    """创建VQ-VAE模型"""
    
    if VectorQuantizer is None:
        print("taming-transformers未安装，返回简化版模型")
        return SimpleVQVAE(**kwargs)
    
    # 默认配置
    default_config = {
        "n_embed": 16384,
        "embed_dim": 256,
        "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "z_channels": 256,
        "beta": 0.25,
        "learning_rate": 4.5e-6,
        "disc_start": 50001,
        "disc_weight": 0.8,
        "codebook_weight": 1.0,
        "pixelloss_weight": 1.0,
        "perceptual_weight": 1.0
    }
    
    # 合并配置
    if config_path and os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        config = OmegaConf.merge(default_config, config)
    else:
        config = OmegaConf.create(default_config)
        config.update(kwargs)
    
    return VQVAEModel(**config)

if __name__ == "__main__":
    # 测试模型创建
    model = create_vqvae_model()
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 测试前向传播
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        recon, loss = model(x)
    print(f"输入形状: {x.shape}")
    print(f"重构形状: {recon.shape}")
    print(f"量化损失: {loss.item():.4f}") 
