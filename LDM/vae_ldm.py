"""
VAE-LDM 完整模型
集成VAE编码器/解码器和扩散模型的完整潜在扩散系统
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from vae_diffusion import VAEDiffusionUNet
from gaussian_diffusion import create_diffusion
import sys
import os

# 添加VAE路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VAE'))
from advanced_vqvae import AdvancedVQVAE

class VAELatentDiffusionModel(nn.Module):
    """
    完整的VAE潜在扩散模型
    """
    def __init__(
        self,
        vae_config: Dict[str, Any],
        unet_config: Dict[str, Any],
        diffusion_config: Dict[str, Any],
        vae_checkpoint_path: str = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.device = device
        
        # 1. 初始化VAE（冻结参数）
        self.vae = self._load_vae(vae_config, vae_checkpoint_path)
        
        # 2. 初始化U-Net
        self.unet = VAEDiffusionUNet(**unet_config)
        
        # 3. 初始化扩散过程
        self.diffusion = create_diffusion(**diffusion_config)
        
        # 移动到设备
        self.to(device)
        
        print(f"✅ VAE-LDM 初始化完成")
        print(f"   VAE潜在维度: {self.vae.latent_dim}")
        print(f"   U-Net类别数: {self.unet.num_classes}")
        print(f"   扩散步数: {self.diffusion.num_timesteps}")
        
    def _load_vae(self, vae_config: Dict[str, Any], checkpoint_path: str) -> AdvancedVQVAE:
        """加载预训练的VAE"""
        vae = AdvancedVQVAE(**vae_config)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            vae.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ VAE 权重加载成功: {checkpoint_path}")
        else:
            print(f"⚠️  VAE checkpoint 未找到: {checkpoint_path}")
            
        # 冻结VAE参数
        for param in vae.parameters():
            param.requires_grad = False
        vae.eval()
        
        return vae
    
    @torch.no_grad()
    def encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """
        将图像编码到潜在空间
        
        Args:
            images: [B, 3, H, W] 输入图像 [-1, 1]
            
        Returns:
            latents: [B, latent_dim, h, w] 潜在表示
        """
        self.vae.eval()
        
        # VAE编码
        x_recon, vq_loss, perplexity, encodings, min_encoding_indices, z_q = self.vae(images)
        
        # 返回量化后的潜在表示
        return z_q
    
    @torch.no_grad() 
    def decode_from_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """
        从潜在空间解码到图像
        
        Args:
            latents: [B, latent_dim, h, w] 潜在表示
            
        Returns:
            images: [B, 3, H, W] 重建图像 [-1, 1]
        """
        self.vae.eval()
        
        # VAE解码
        decoded = self.vae.decode(latents)
        return decoded
    
    def forward(self, images: torch.Tensor, class_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        训练前向传播
        
        Args:
            images: [B, 3, H, W] 输入图像
            class_labels: [B] 类别标签
            
        Returns:
            loss_dict: 包含各种损失的字典
        """
        # 1. 编码到潜在空间
        with torch.no_grad():
            latents = self.encode_to_latent(images)
        
        # 2. 随机采样时间步
        batch_size = latents.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)
        
        # 3. 计算扩散损失
        model_kwargs = {}
        if class_labels is not None:
            model_kwargs['y'] = class_labels
            
        losses = self.diffusion.training_losses(
            model=self.unet,
            x_start=latents,
            t=t,
            model_kwargs=model_kwargs
        )
        
        return losses
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        num_inference_steps: Optional[int] = None,
        use_ddim: bool = True,
        eta: float = 0.0,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        生成样本
        
        Args:
            num_samples: 生成样本数量
            class_labels: [num_samples] 类别标签
            guidance_scale: 分类器自由引导强度
            num_inference_steps: 推理步数
            use_ddim: 是否使用DDIM采样
            eta: DDIM随机性参数
            return_intermediates: 是否返回中间结果
            
        Returns:
            images: [num_samples, 3, H, W] 生成的图像
            latents: [num_samples, latent_dim, h, w] 可选的潜在表示
        """
        self.unet.eval()
        
        # 获取潜在空间形状
        with torch.no_grad():
            # 使用正确的输入图像尺寸：256×256 (数据集的实际尺寸)
            dummy_img = torch.randn(1, 3, 256, 256, device=self.device) 
            dummy_latent = self.encode_to_latent(dummy_img)
            latent_shape = dummy_latent.shape[1:]
        
        # 采样形状
        shape = (num_samples, *latent_shape)
        
        # 模型参数
        model_kwargs = {}
        if class_labels is not None:
            assert len(class_labels) == num_samples
            model_kwargs['y'] = class_labels
        
        # 分类器自由引导
        if guidance_scale > 1.0 and class_labels is not None:
            # 需要实现CFG，这里先简化
            pass
        
        # 采样潜在表示
        if use_ddim:
            latents = self.diffusion.ddim_sample_loop(
                model=self.unet,
                shape=shape,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                device=self.device,
                progress=True,
                eta=eta
            )
        else:
            latents = self.diffusion.p_sample_loop(
                model=self.unet,
                shape=shape,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                device=self.device,
                progress=True
            )
        
        # 解码到图像空间
        images = self.decode_from_latent(latents)
        
        if return_intermediates:
            return images, latents
        else:
            return images, None
    
    def configure_optimizers(self, lr: float = 1e-4, weight_decay: float = 0.0):
        """配置优化器"""
        # 只优化U-Net参数，VAE参数被冻结
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def save_checkpoint(self, filepath: str, epoch: int, optimizer_state: Dict = None):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'unet_state_dict': self.unet.state_dict(),
            'model_config': {
                'unet_config': {
                    'image_size': self.unet.image_size,
                    'in_channels': self.unet.in_channels,
                    'model_channels': self.unet.model_channels,
                    'out_channels': self.unet.out_channels,
                    'num_res_blocks': self.unet.num_res_blocks,
                    'attention_resolutions': self.unet.attention_resolutions,
                    'dropout': self.unet.dropout,
                    'channel_mult': self.unet.channel_mult,
                    'num_classes': self.unet.num_classes,
                    'use_checkpoint': self.unet.use_checkpoint,
                    'num_heads': self.unet.num_heads,
                    'num_heads_upsample': self.unet.num_heads_upsample,
                },
                'diffusion_config': {
                    'timestep_respacing': "",
                    'noise_schedule': "cosine",
                    'diffusion_steps': self.diffusion.num_timesteps,
                }
            }
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        torch.save(checkpoint, filepath)
        print(f"✅ Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = False):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 加载U-Net权重
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        optimizer_state = checkpoint.get('optimizer_state_dict', None) if load_optimizer else None
        
        print(f"✅ Checkpoint loaded: {filepath} (epoch {epoch})")
        
        return epoch, optimizer_state

def create_vae_ldm(
    vae_checkpoint_path: str,
    image_size: int = 32,
    num_classes: int = 31,
    diffusion_steps: int = 1000,
    noise_schedule: str = "cosine",
    device: str = None
) -> VAELatentDiffusionModel:
    """
    创建VAE-LDM模型的便捷函数
    
    Args:
        vae_checkpoint_path: VAE检查点路径
        image_size: 图像尺寸
        num_classes: 类别数
        diffusion_steps: 扩散步数
        noise_schedule: 噪声调度
        device: 设备
        
    Returns:
        VAE-LDM模型
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # VAE配置（需要与预训练模型匹配）
    vae_config = {
        'in_channels': 3,
        'latent_dim': 256,
        'num_embeddings': 512,
        'beta': 0.25,
        'groups': 1,
    }
    
    # U-Net配置
    unet_config = {
        'image_size': image_size,
        'in_channels': 256,  # VAE潜在维度
        'model_channels': 128,
        'out_channels': 256,  # VAE潜在维度
        'num_res_blocks': 2,
        'attention_resolutions': (8, 16),
        'dropout': 0.1,
        'channel_mult': (1, 2, 4, 8),
        'conv_resample': True,
        'dims': 2,
        'num_classes': num_classes,
        'use_checkpoint': False,
        'num_heads': 4,
        'num_heads_upsample': -1,
        'use_scale_shift_norm': True,
    }
    
    # 扩散配置
    diffusion_config = {
        'timestep_respacing': "",
        'noise_schedule': noise_schedule,
        'use_kl': False,
        'sigma_small': False,
        'predict_xstart': False,
        'learn_sigma': False,
        'rescale_learned_sigmas': False,
        'diffusion_steps': diffusion_steps,
        'rescale_timesteps': True,
    }
    
    model = VAELatentDiffusionModel(
        vae_config=vae_config,
        unet_config=unet_config,
        diffusion_config=diffusion_config,
        vae_checkpoint_path=vae_checkpoint_path,
        device=device
    )
    
    return model 
