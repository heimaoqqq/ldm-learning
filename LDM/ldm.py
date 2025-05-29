import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加VAE模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VAE'))

from adv_vq_vae import AdvVQVAE
from unet import UNetModel
from scheduler import DDPMScheduler, DDIMScheduler
from typing import Optional, Union, Tuple, Dict, Any
import numpy as np

class LatentDiffusionModel(nn.Module):
    """
    潜在扩散模型 (LDM)
    结合预训练的VAE和U-Net进行潜在空间中的扩散生成
    """
    def __init__(
        self,
        # VAE配置
        vae_config: Dict[str, Any],
        # U-Net配置
        unet_config: Dict[str, Any],
        # 扩散配置
        scheduler_config: Dict[str, Any],
        # 可选参数
        vae_path: Optional[str] = None,
        freeze_vae: bool = True,
        # 训练配置
        use_cfg: bool = True,
        cfg_dropout_prob: float = 0.1,
    ):
        super().__init__()
        
        self.use_cfg = use_cfg
        self.cfg_dropout_prob = cfg_dropout_prob
        
        # 初始化VAE
        self.vae = AdvVQVAE(
            in_channels=vae_config['in_channels'],
            latent_dim=vae_config['latent_dim'],
            num_embeddings=vae_config['num_embeddings'],
            beta=vae_config['beta'],
            decay=vae_config['vq_ema_decay'],
            groups=vae_config['groups'],
            disc_ndf=64
        )
        
        # 加载预训练的VAE权重
        if vae_path and os.path.exists(vae_path):
            print(f"加载预训练VAE: {vae_path}")
            checkpoint = torch.load(vae_path, map_location='cpu')
            self.vae.load_state_dict(checkpoint)
        else:
            print(f"警告: VAE权重文件不存在: {vae_path}")
        
        # 冻结VAE参数
        if freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()
            print("VAE参数已冻结")
        
        # 初始化U-Net
        self.unet = UNetModel(**unet_config)
        
        # 初始化调度器
        self.scheduler = DDPMScheduler(
            num_train_timesteps=scheduler_config['timesteps'],
            beta_start=scheduler_config['beta_start'],
            beta_end=scheduler_config['beta_end'],
            beta_schedule=scheduler_config['noise_schedule'],
            clip_denoised=scheduler_config['clip_denoised'],
        )
        
        print(f"LDM模型初始化完成:")
        print(f"  - VAE: {vae_config['latent_dim']}维潜在空间")
        print(f"  - U-Net: {unet_config['in_channels']}→{unet_config['out_channels']}通道")
        print(f"  - 扩散步数: {scheduler_config['timesteps']}")
        print(f"  - 噪声调度: {scheduler_config['noise_schedule']}")
    
    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        将图像编码到潜在空间
        Args:
            x: [batch_size, 3, height, width] 输入图像 (已归一化到[-1,1])
        Returns:
            z: [batch_size, latent_dim, h, w] 潜在表示
        """
        z = self.vae.encoder(x)
        z_q, _, _ = self.vae.vq(z)
        return z_q
    
    @torch.no_grad()
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        从潜在空间解码到图像
        Args:
            z: [batch_size, latent_dim, h, w] 潜在表示
        Returns:
            x: [batch_size, 3, height, width] 重建图像
        """
        return self.vae.decoder(z)
    
    def forward(
        self,
        x: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        训练时的前向传播
        Args:
            x: [batch_size, 3, height, width] 输入图像
            class_labels: [batch_size] 类别标签
            noise: [batch_size, latent_dim, h, w] 可选的预定义噪声
        Returns:
            包含损失和预测的字典
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 1. 编码到潜在空间
        with torch.no_grad():
            latents = self.encode_to_latent(x)  # [B, latent_dim, h, w]
        
        # 2. 采样噪声和时间步
        if noise is None:
            noise = torch.randn_like(latents)
        
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (batch_size,), device=device
        ).long()
        
        # 3. 添加噪声 (前向扩散过程)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # 4. Classifier-Free Guidance训练
        if self.use_cfg and class_labels is not None:
            # 随机丢弃一部分条件
            if self.training:
                dropout_mask = torch.rand(batch_size, device=device) < self.cfg_dropout_prob
                # 将被丢弃的标签设为特殊值 (如-1或num_classes)
                unconditional_labels = torch.full_like(class_labels, -1)
                class_labels = torch.where(dropout_mask, unconditional_labels, class_labels)
        
        # 5. 预测噪声
        predicted_noise = self.unet(
            x=noisy_latents,
            timesteps=timesteps,
            class_labels=class_labels,
        )
        
        # 6. 计算损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return {
            'loss': loss,
            'predicted_noise': predicted_noise,
            'target_noise': noise,
            'timesteps': timesteps,
            'noisy_latents': noisy_latents,
        }
    
    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        class_labels: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        生成图像
        Args:
            batch_size: 批量大小
            class_labels: [batch_size] 类别标签
            num_inference_steps: 推理步数
            guidance_scale: CFG引导强度
            eta: DDIM参数
            generator: 随机数生成器
            latents: 初始潜在表示 (可选)
            return_intermediates: 是否返回中间结果
        Returns:
            生成的图像 [batch_size, 3, height, width]
        """
        device = next(self.parameters()).device
        
        # 设置推理调度器
        if eta > 0:
            scheduler = DDIMScheduler(
                eta=eta,
                num_train_timesteps=self.scheduler.num_train_timesteps,
                beta_start=self.scheduler.beta_start,
                beta_end=self.scheduler.beta_end,
                beta_schedule=self.scheduler.beta_schedule,
                clip_denoised=self.scheduler.clip_denoised,
            )
        else:
            scheduler = self.scheduler
            
        scheduler.set_timesteps(num_inference_steps, device=device)
        
        # 初始化潜在表示
        if latents is None:
            # 从配置推断潜在空间形状
            latent_shape = (batch_size, self.unet.in_channels, 32, 32)  # 32x32潜在空间
            latents = torch.randn(latent_shape, generator=generator, device=device)
        
        latents = latents * scheduler.init_noise_sigma if hasattr(scheduler, 'init_noise_sigma') else latents
        
        intermediates = []
        
        # 去噪循环
        for i, t in enumerate(scheduler.timesteps):
            # 扩展潜在表示以用于CFG
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            # 预测噪声
            if guidance_scale > 1.0 and class_labels is not None:
                # Classifier-Free Guidance
                # 无条件预测
                unconditional_labels = torch.full_like(class_labels, -1)
                combined_labels = torch.cat([unconditional_labels, class_labels])
                
                noise_pred = self.unet(
                    x=latent_model_input,
                    timesteps=torch.cat([t.unsqueeze(0)] * latent_model_input.shape[0]),
                    class_labels=combined_labels,
                )
                
                # 分离有条件和无条件的预测
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.unet(
                    x=latents,
                    timesteps=t.unsqueeze(0).repeat(batch_size),
                    class_labels=class_labels,
                )
            
            # 计算前一时间步的潜在表示
            latents = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=latents,
                generator=generator,
            ).prev_sample
            
            if return_intermediates:
                intermediates.append(latents.cpu().clone())
        
        # 解码到图像空间
        images = self.decode_from_latent(latents)
        
        if return_intermediates:
            return images, intermediates
        else:
            return images
    
    @torch.no_grad()
    def interpolate(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
        num_steps: int = 10,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        在潜在空间中进行图像插值
        Args:
            images1: [batch_size, 3, height, width] 起始图像
            images2: [batch_size, 3, height, width] 结束图像
            num_steps: 插值步数
            class_labels: 类别标签
        Returns:
            插值图像序列 [num_steps, batch_size, 3, height, width]
        """
        # 编码到潜在空间
        latents1 = self.encode_to_latent(images1)
        latents2 = self.encode_to_latent(images2)
        
        # 在潜在空间中插值
        interpolated_images = []
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            interpolated_latents = (1 - alpha) * latents1 + alpha * latents2
            interpolated_image = self.decode_from_latent(interpolated_latents)
            interpolated_images.append(interpolated_image)
        
        return torch.stack(interpolated_images)
    
    def get_latent_shape(self, image_size: int = 256) -> Tuple[int, int, int]:
        """
        获取给定图像尺寸对应的潜在空间形状
        Args:
            image_size: 图像尺寸
        Returns:
            (channels, height, width) 潜在空间形状
        """
        # 根据VAE的下采样比例计算
        # 我们的VAE下采样了8倍 (256 -> 32)
        latent_size = image_size // 8
        return (self.unet.in_channels, latent_size, latent_size)
    
    def save_pretrained(self, save_path: str):
        """保存模型"""
        os.makedirs(save_path, exist_ok=True)
        
        # 只保存U-Net权重 (VAE保持冻结)
        torch.save(self.unet.state_dict(), os.path.join(save_path, 'unet.pth'))
        print(f"LDM模型已保存到: {save_path}")
    
    def load_pretrained(self, load_path: str):
        """加载模型"""
        unet_path = os.path.join(load_path, 'unet.pth')
        if os.path.exists(unet_path):
            self.unet.load_state_dict(torch.load(unet_path, map_location='cpu'))
            print(f"LDM模型已从 {load_path} 加载")
        else:
            raise FileNotFoundError(f"找不到模型文件: {unet_path}")


class ClassifierFreeGuidance:
    """
    无分类器引导 (Classifier-Free Guidance) 工具类
    """
    def __init__(self, guidance_scale: float = 7.5, unconditional_token: int = -1):
        self.guidance_scale = guidance_scale
        self.unconditional_token = unconditional_token
    
    def apply_cfg(
        self,
        noise_pred_uncond: torch.Tensor,
        noise_pred_cond: torch.Tensor,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        应用无分类器引导
        Args:
            noise_pred_uncond: 无条件噪声预测
            noise_pred_cond: 有条件噪声预测
            guidance_scale: 引导强度
        Returns:
            引导后的噪声预测
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        
        return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    
    def prepare_labels_for_cfg(
        self, 
        labels: torch.Tensor,
        dropout_prob: float = 0.1,
        training: bool = True,
    ) -> torch.Tensor:
        """
        为CFG训练准备标签
        Args:
            labels: 原始标签
            dropout_prob: 条件丢弃概率
            training: 是否为训练模式
        Returns:
            处理后的标签
        """
        if training and dropout_prob > 0:
            dropout_mask = torch.rand(labels.shape[0], device=labels.device) < dropout_prob
            unconditional_labels = torch.full_like(labels, self.unconditional_token)
            return torch.where(dropout_mask, unconditional_labels, labels)
        return labels 