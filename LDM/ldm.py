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
        scheduler_config = config['diffusion']
        scheduler_type = scheduler_config.get('scheduler_type', 'ddpm').lower()
        
        if scheduler_type == 'ddim':
            # 🔧 从推理配置获取eta参数，确保一致性
            inference_config = config.get('inference', {})
            default_eta = inference_config.get('eta', 0.0)
            adaptive_eta = inference_config.get('adaptive_eta', False)
            
            self.scheduler = DDIMScheduler(
                num_train_timesteps=scheduler_config['timesteps'],
                beta_start=scheduler_config['beta_start'],
                beta_end=scheduler_config['beta_end'],
                beta_schedule=scheduler_config['noise_schedule'],
                eta=default_eta,
                adaptive_eta=adaptive_eta
            )
            print(f"✅ 使用DDIM调度器，默认eta={default_eta}, 自适应eta={adaptive_eta}")
        else:
            # 标准DDPM调度器
            self.scheduler = DDPMScheduler(
                num_train_timesteps=scheduler_config['timesteps'],
                beta_start=scheduler_config['beta_start'],
                beta_end=scheduler_config['beta_end'],
                beta_schedule=scheduler_config['noise_schedule']
            )
            print(f"✅ 使用DDPM调度器")
        
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
            # 🔧 添加潜在表示范围检查和裁剪
            # 监控输入范围
            input_min, input_max = latents.min().item(), latents.max().item()
            input_std = latents.std().item()
            
            # 如果范围超出合理范围，进行裁剪
            if abs(input_min) > 10 or abs(input_max) > 10:
                print(f"⚠️ 潜在表示范围异常: [{input_min:.3f}, {input_max:.3f}], 进行裁剪")
                latents = torch.clamp(latents, min=-5.0, max=5.0)
            
            # 如果标准差过大，进行缩放
            if input_std > 3.0:
                print(f"⚠️ 潜在表示标准差过大: {input_std:.3f}, 进行缩放")
                latents = latents / (input_std / 2.0)  # 缩放到合理范围
            
            # VAE解码
            decoded = self.vae.decoder(latents)
            
            # 🔧 检查解码输出范围
            output_min, output_max = decoded.min().item(), decoded.max().item()
            if output_min < -1.1 or output_max > 1.1:
                print(f"⚠️ VAE解码输出超出tanh范围: [{output_min:.3f}, {output_max:.3f}]")
                decoded = torch.clamp(decoded, min=-1.0, max=1.0)
            
            return decoded
    
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
        eta: Optional[float] = None  # 🔧 改为可选参数，默认使用调度器配置
    ) -> torch.Tensor:
        """
        DDIM/DDPM采样生成图像
        
        Args:
            batch_size: 批次大小
            class_labels: [batch_size] 类别标签
            num_inference_steps: 推理步数
            guidance_scale: CFG引导强度
            eta: DDIM参数 (None=使用调度器默认值, 0=确定性采样)
            
        Returns:
            生成的图像 [batch_size, 3, H, W]
        """
        device = self.device
        
        # 🔧 确定实际使用的eta值
        if eta is None:
            # 使用调度器的默认eta（如果是DDIM）
            actual_eta = getattr(self.scheduler, 'eta', 0.0)
        else:
            actual_eta = eta
        
        # 🔧 检查调度器类型和eta兼容性
        is_ddim = isinstance(self.scheduler, DDIMScheduler)
        if not is_ddim and actual_eta != 0.0:
            print(f"⚠️ 当前使用DDPM调度器，eta参数({actual_eta})将被忽略")
            actual_eta = 0.0
        
        print(f"🔄 开始采样: 调度器={'DDIM' if is_ddim else 'DDPM'}, 步数={num_inference_steps}, eta={actual_eta}, CFG={guidance_scale}")
        
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
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        for i, t in enumerate(self.scheduler.timesteps):
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
            
            # 🔧 统一的调度器步骤调用
            try:
                if is_ddim:
                    # DDIM调度器 - 传递eta参数
                    result = self.scheduler.step(
                        model_output=noise_pred, 
                        timestep=t, 
                        sample=latents, 
                        eta=actual_eta
                    )
                else:
                    # DDPM调度器 - 传递eta以保持兼容性（但会被忽略）
                    result = self.scheduler.step(
                        model_output=noise_pred, 
                        timestep=t, 
                        sample=latents,
                        eta=actual_eta  # DDPM会忽略这个参数
                    )
                
                latents = result.prev_sample if hasattr(result, 'prev_sample') else result[0]
                
                # 🔧 定期打印进度
                if i % (len(self.scheduler.timesteps) // 4) == 0:
                    print(f"  📊 采样进度: {i+1}/{len(self.scheduler.timesteps)}")
                    
            except Exception as e:
                print(f"⚠️ 采样步骤 {i} 出错: {e}")
                # 继续采样，但使用前一个latents
                continue
        
        # 解码到图像空间
        try:
            images = self.decode_from_latent(latents)
            
            # 🔧 改进的归一化处理
            # 记录解码后的原始范围
            raw_min, raw_max = images.min().item(), images.max().item()
            
            # 归一化到[0,1]，更鲁棒的处理
            if raw_min >= -1.1 and raw_max <= 1.1:
                # 标准情况：VAE输出在[-1,1]范围内
                images = (images + 1.0) / 2.0
            else:
                # 异常情况：使用min-max归一化
                print(f"⚠️ VAE输出范围异常: [{raw_min:.3f}, {raw_max:.3f}], 使用自适应归一化")
                images = (images - raw_min) / (raw_max - raw_min + 1e-8)
            
            # 最终裁剪到[0,1]
            images = torch.clamp(images, 0.0, 1.0)
            
            # 🔧 质量检查
            final_min, final_max = images.min().item(), images.max().item()
            final_mean = images.mean().item()
            
            if final_mean < 0.1 or final_mean > 0.9:
                print(f"⚠️ 生成图像亮度异常: 均值={final_mean:.3f}")
            
            print(f"✅ 采样完成，生成 {batch_size} 张图像")
            print(f"   📊 最终图像范围: [{final_min:.3f}, {final_max:.3f}], 均值: {final_mean:.3f}")
            return images
            
        except Exception as e:
            print(f"⚠️ 图像解码失败: {e}")
            # 返回一个默认的图像张量
            return torch.zeros(batch_size, 3, 256, 256, device=device)
    
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
