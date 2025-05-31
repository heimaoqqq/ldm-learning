#!/usr/bin/env python3
"""
潜在扩散模型 (Latent Diffusion Model) 主要实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import math
from typing import Dict, Tuple, Optional, Any

# 添加VAE模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VAE'))

from adv_vq_vae import AdvVQVAE
from unet import UNetModel  # 使用修复后的原始U-Net
from scheduler import DDPMScheduler, DDIMScheduler  # 🆕 导入DDIM调度器

class LatentDiffusionModel(nn.Module):
    """
    潜在扩散模型
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化VAE (预训练，冻结参数)
        self.vae = self._init_vae(config['vae'])
        
        # 初始化扩散调度器 - 🆕 支持DDIM和改进eta
        if config['diffusion'].get('scheduler_type', 'ddpm') == 'ddim':
            self.scheduler = DDIMScheduler(
                num_train_timesteps=config['diffusion']['timesteps'],
                beta_start=config['diffusion']['beta_start'],
                beta_end=config['diffusion']['beta_end'],
                beta_schedule=config['diffusion']['noise_schedule'],
                eta=config['inference'].get('eta', 0.0),
                adaptive_eta=config['inference'].get('adaptive_eta', False)
            )
        else:
            self.scheduler = DDPMScheduler(
                num_train_timesteps=config['diffusion']['timesteps'],
                beta_start=config['diffusion']['beta_start'],
                beta_end=config['diffusion']['beta_end'],
                beta_schedule=config['diffusion']['noise_schedule']
            )
        
        # 初始化U-Net
        self.unet = UNetModel(**config['unet'])
        
        # 其他配置
        self.use_cfg = config['training'].get('use_cfg', False)
        self.cfg_dropout_prob = config['training'].get('cfg_dropout_prob', 0.1)
        
    def _init_vae(self, vae_config: Dict[str, Any]) -> AdvVQVAE:
        """初始化VAE并加载预训练权重"""
        vae = AdvVQVAE(
            in_channels=vae_config['in_channels'],
            latent_dim=vae_config['latent_dim'],
            num_embeddings=vae_config['num_embeddings'],
            beta=vae_config['beta'],
            decay=vae_config['vq_ema_decay'],
            groups=vae_config['groups']
        )
        
        # 加载预训练权重
        if 'model_path' in vae_config and vae_config['model_path']:
            try:
                print(f"🔄 加载VAE预训练模型: {vae_config['model_path']}")
                state_dict = torch.load(vae_config['model_path'], map_location='cpu')
                vae.load_state_dict(state_dict)
                print("✅ VAE模型加载成功")
            except Exception as e:
                print(f"⚠️ VAE模型加载失败: {e}")
        
        # 冻结VAE参数
        if vae_config.get('freeze', True):
            for param in vae.parameters():
                param.requires_grad = False
            vae.eval()
            print("❄️ VAE参数已冻结")
        
        return vae
    
    def encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """将图像编码到潜在空间"""
        with torch.no_grad():
            # VAE编码
            encoded = self.vae.encoder(images)
            # VQ量化
            quantized, _, _ = self.vae.vq(encoded)
            return quantized
    
    def decode_from_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """从潜在空间解码到图像"""
        with torch.no_grad():
            return self.vae.decoder(latents)
    
    def forward(self, images: torch.Tensor, class_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: [B, 3, H, W] 输入图像
            class_labels: [B] 类别标签
            
        Returns:
            包含损失信息的字典
        """
        batch_size = images.shape[0]
        
        # 编码到潜在空间
        latents = self.encode_to_latent(images)  # [B, C, H', W']
        
        # 采样时间步
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, 
            (batch_size,), device=images.device
        )
        
        # 添加噪声
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # CFG训练 (随机丢弃类别标签)
        if self.use_cfg and class_labels is not None:
            # 随机mask掉一些类别标签用于CFG训练
            cfg_mask = torch.rand(batch_size, device=images.device) < self.cfg_dropout_prob
            class_labels = class_labels.clone()
            class_labels[cfg_mask] = -1  # 使用-1表示无条件
        
        # U-Net预测噪声
        predicted_noise = self.unet(noisy_latents, timesteps, class_labels)
        
        # 计算损失
        noise_loss = F.mse_loss(predicted_noise, noise, reduction='mean')
        
        return {
            'noise_loss': noise_loss,
            'total_loss': noise_loss,
            'predicted_noise': predicted_noise,
            'target_noise': noise
        }
    
    @torch.no_grad()
    def sample(
        self, 
        batch_size: int, 
        class_labels: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        DDIM采样生成图像
        
        Args:
            batch_size: 批次大小
            class_labels: [batch_size] 类别标签
            num_inference_steps: 推理步数
            guidance_scale: CFG引导强度
            eta: DDIM参数 (0为确定性采样)
            
        Returns:
            生成的图像 [batch_size, 3, H, W]
        """
        device = self.device
        
        # 获取潜在空间尺寸
        # 假设输入图像为256x256，VAE下采样8倍，得到32x32
        latent_height = 32
        latent_width = 32
        latent_channels = self.config['unet']['in_channels']
        
        # 初始化随机噪声
        latents = torch.randn(
            batch_size, latent_channels, latent_height, latent_width,
            device=device
        )
        
        # 设置推理调度器
        self.scheduler.set_timesteps(num_inference_steps)
        
        for t in self.scheduler.timesteps:
            # 扩展时间步
            t_batch = t.unsqueeze(0).repeat(batch_size).to(device)
            
            if self.use_cfg and class_labels is not None and guidance_scale > 1.0:
                # CFG: 同时预测有条件和无条件噪声
                latent_model_input = torch.cat([latents] * 2)
                t_input = torch.cat([t_batch] * 2)
                
                # 创建条件标签 (有条件 + 无条件)
                class_input = torch.cat([
                    class_labels,
                    torch.full_like(class_labels, -1)  # 无条件标签
                ])
                
                # U-Net预测
                noise_pred = self.unet(latent_model_input, t_input, class_input)
                
                # 分离有条件和无条件预测
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                
                # CFG引导
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # 无CFG的标准预测
                noise_pred = self.unet(latents, t_batch, class_labels)
            
            # DDIM步骤
            result = self.scheduler.step(noise_pred, t, latents)
            latents = result.prev_sample if hasattr(result, 'prev_sample') else result[0]
        
        # 解码到图像空间
        images = self.decode_from_latent(latents)
        
        # 归一化到[0,1]
        images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)
        
        return images
    
    def get_loss_dict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """获取损失字典 (兼容训练循环)"""
        images = batch['image']
        class_labels = batch.get('label', None)
        
        return self(images, class_labels)


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
