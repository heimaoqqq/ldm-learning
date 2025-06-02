"""
标准VAE-LDM实现 - 使用AutoencoderKL替换VQ-VAE
基于CompVis/latent-diffusion的标准架构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from vae_diffusion import VAEDiffusionUNet
from gaussian_diffusion import create_diffusion

class StandardVAELatentDiffusionModel(nn.Module):
    """
    标准VAE潜在扩散模型 - 使用AutoencoderKL
    """
    def __init__(
        self,
        unet_config: Dict[str, Any],
        diffusion_config: Dict[str, Any],
        vae_model_name: str = "runwayml/stable-diffusion-v1-5",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.device = device
        
        # 1. 加载标准AutoencoderKL
        self.vae = self._load_standard_vae(vae_model_name)
        
        # 2. 初始化U-Net (适配4通道潜在空间)
        unet_config['in_channels'] = 4  # AutoencoderKL的潜在通道数
        unet_config['out_channels'] = 4
        self.unet = VAEDiffusionUNet(**unet_config)
        
        # 3. 初始化扩散过程
        self.diffusion = create_diffusion(**diffusion_config)
        
        # 移动到设备
        self.to(device)
        
        print(f"✅ 标准VAE-LDM 初始化完成")
        print(f"   VAE潜在通道数: {self.vae.config.latent_channels}")
        print(f"   VAE缩放因子: {self.vae.config.scaling_factor}")
        print(f"   U-Net类别数: {self.unet.num_classes}")
        print(f"   扩散步数: {self.diffusion.num_timesteps}")
        
    def _load_standard_vae(self, model_name: str):
        """加载标准AutoencoderKL"""
        try:
            from diffusers import AutoencoderKL
            
            vae = AutoencoderKL.from_pretrained(
                model_name,
                subfolder="vae",
                torch_dtype=torch.float32
            )
            
            # 冻结VAE参数
            for param in vae.parameters():
                param.requires_grad = False
            vae.eval()
            
            print(f"✅ 标准AutoencoderKL加载成功: {model_name}")
            print(f"🔒 VAE参数已冻结（{sum(p.numel() for p in vae.parameters())} 参数）")
            
            return vae
            
        except Exception as e:
            print(f"❌ 无法加载标准AutoencoderKL: {e}")
            print("💡 请安装必要库: pip install diffusers transformers accelerate")
            raise
    
    @torch.no_grad()
    def encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """
        使用标准方式编码到潜在空间
        
        Args:
            images: [B, 3, H, W] 输入图像 [-1, 1]
            
        Returns:
            latents: [B, 4, h, w] 潜在表示
        """
        self.vae.eval()
        
        # 标准编码方式
        latent_dist = self.vae.encode(images).latent_dist
        latents = latent_dist.sample() * self.vae.config.scaling_factor
        
        return latents
    
    @torch.no_grad() 
    def decode_from_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """
        从潜在空间解码到图像
        
        Args:
            latents: [B, 4, h, w] 潜在表示
            
        Returns:
            images: [B, 3, H, W] 重建图像 [-1, 1]
        """
        self.vae.eval()
        
        # 标准解码方式
        decoded = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        
        return decoded
    
    def forward(self, images: torch.Tensor, class_labels: Optional[torch.Tensor] = None, current_epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        训练前向传播
        
        Args:
            images: [B, 3, H, W] 输入图像
            class_labels: [B] 类别标签
            current_epoch: 当前训练的epoch号
            
        Returns:
            loss_dict: 包含各种损失的字典
        """
        # 1. 编码到潜在空间
        with torch.no_grad():
            latents = self.encode_to_latent(images)
        
        # 调试输出：前5个epoch显示潜在表示统计信息
        if current_epoch is not None and current_epoch < 5:
            if not hasattr(self, '_debug_batch_count'):
                self._debug_batch_count = {}
            
            if current_epoch not in self._debug_batch_count:
                self._debug_batch_count[current_epoch] = 0
            
            if self._debug_batch_count[current_epoch] < 3:  # 每个epoch只显示前3个batch
                print(f"📊 Epoch {current_epoch+1} Batch {self._debug_batch_count[current_epoch]+1}:")
                print(f"   标准VAE输出: 均值={latents.mean().item():.4f}, 标准差={latents.std().item():.4f}")
                print(f"   潜在表示范围: [{latents.min().item():.4f}, {latents.max().item():.4f}]")
                print(f"   潜在空间形状: {latents.shape}")
                
                self._debug_batch_count[current_epoch] += 1
        
        # 2. 随机采样时间步
        batch_size = latents.shape[0]
        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device, dtype=torch.long)
        
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
        """
        self.unet.eval()
        
        # 获取潜在空间形状 (4通道, 32x32 for 256x256图像)
        latent_shape = (4, 32, 32)  # 标准AutoencoderKL的潜在形状
        shape = (num_samples, *latent_shape)
        
        # 模型参数
        model_kwargs = {}
        if class_labels is not None:
            assert len(class_labels) == num_samples
            model_kwargs['y'] = class_labels
        
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
        """配置优化器 - 只优化U-Net参数"""
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

def create_standard_vae_ldm(
    image_size: int = 32,
    num_classes: int = 31,
    diffusion_steps: int = 1000,
    noise_schedule: str = "cosine",
    vae_model_name: str = "runwayml/stable-diffusion-v1-5",
    device: str = None
) -> StandardVAELatentDiffusionModel:
    """
    创建标准VAE-LDM模型的便捷函数
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # U-Net配置 (适配4通道潜在空间)
    unet_config = {
        'image_size': image_size,
        'in_channels': 4,                    # 标准AutoencoderKL使用4通道
        'model_channels': 192,
        'out_channels': 4,                   # 输出也是4通道
        'num_res_blocks': 3,
        'attention_resolutions': (4, 8, 16),
        'dropout': 0.1,
        'channel_mult': (1, 2, 3, 4),
        'num_classes': num_classes,
        'use_checkpoint': False,
        'num_heads': 4,
        'num_heads_upsample': -1,
        'use_scale_shift_norm': True
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

    model = StandardVAELatentDiffusionModel(
        unet_config=unet_config,
        diffusion_config=diffusion_config,
        vae_model_name=vae_model_name,
        device=device
    )
    
    return model 
