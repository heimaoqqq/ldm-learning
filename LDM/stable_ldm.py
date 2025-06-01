"""
基于Stable Diffusion的轻量化潜在扩散模型
专门适配您的VQ-VAE和31个类别条件生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Dict, Tuple, Optional, Any, Union
from tqdm import tqdm

# 添加VAE模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VAE'))

from adv_vq_vae import AdvVQVAE
from simple_enhanced_unet import SimpleEnhancedUNet, create_simple_enhanced_unet  # 🚀 增强版UNet
from scheduler import DDPMScheduler, DDIMScheduler

class EMA:
    """指数移动平均"""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # 确保shadow参数在正确的设备上
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                # 确保shadow参数在正确的设备上
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class StableLDM(nn.Module):
    """
    基于Stable Diffusion的轻量化潜在扩散模型
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化VAE (预训练，冻结参数)
        self.vae = self._init_vae(config['vae'])
        
        # 初始化U-Net - 🚀 支持简化增强版
        unet_type = config['unet'].get('type', 'simple')  # 默认使用简单版本
        
        if unet_type == 'simple_enhanced':
            print("🚀 使用简化增强版U-Net (包含多层Transformer)")
            self.unet = create_simple_enhanced_unet(config)
        else:
            print("📋 使用简化增强版U-Net (包含多层Transformer)")
            # 所有情况下都使用增强版UNet
            self.unet = create_simple_enhanced_unet(config)
        
        # 初始化调度器
        scheduler_config = config['diffusion']
        scheduler_type = scheduler_config.get('scheduler_type', 'ddpm').lower()
        
        if scheduler_type == 'ddim':
            self.scheduler = DDIMScheduler(
                num_train_timesteps=scheduler_config['timesteps'],
                beta_start=scheduler_config['beta_start'],
                beta_end=scheduler_config['beta_end'],
                beta_schedule=scheduler_config['noise_schedule'],
            )
            print(f"✅ 使用DDIM调度器")
        else:
            self.scheduler = DDPMScheduler(
                num_train_timesteps=scheduler_config['timesteps'],
                beta_start=scheduler_config['beta_start'],
                beta_end=scheduler_config['beta_end'],
                beta_schedule=scheduler_config['noise_schedule']
            )
            print(f"✅ 使用DDPM调度器")
        
        # CFG配置
        self.use_cfg = config['training'].get('use_cfg', False)
        self.cfg_dropout_prob = config['training'].get('cfg_dropout_prob', 0.15)
        
        # EMA
        if config.get('use_ema', False):
            self.ema = EMA(self.unet, config.get('ema_decay', 0.9999))
            print(f"✅ 启用EMA，衰减率: {config.get('ema_decay', 0.9999)}")
        else:
            self.ema = None
        
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
            # 输入预处理：确保在[0,1]范围，转换到[-1,1]
            images = torch.clamp(images, 0.0, 1.0)
            images = images * 2.0 - 1.0
            
            try:
                # VAE编码
                encoded = self.vae.encoder(images)
                encoded = torch.clamp(encoded, -5.0, 5.0)
                
                # VQ量化
                quantized, _, _ = self.vae.vq(encoded)
                quantized = torch.clamp(quantized, -3.0, 3.0)
                
                return quantized
                
            except Exception as e:
                print(f"⚠️ VAE编码失败: {e}")
                # 返回安全的随机潜在表示
                batch_size = images.shape[0]
                latent_shape = (batch_size, 256, 32, 32)
                return torch.randn(latent_shape, device=images.device) * 0.5
    
    def decode_from_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """从潜在空间解码到图像"""
        with torch.no_grad():
            # 范围检查和裁剪
            latents = torch.clamp(latents, -5.0, 5.0)
            
            # VAE解码
            decoded = self.vae.decoder(latents)
            decoded = torch.clamp(decoded, -1.0, 1.0)
            
            # 转换到[0,1]范围
            images = (decoded + 1.0) / 2.0
            images = torch.clamp(images, 0.0, 1.0)
            
            return images
    
    def forward(self, images: torch.Tensor, class_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        训练前向传播
        
        Args:
            images: [B, 3, H, W] 输入图像
            class_labels: [B] 类别标签
            
        Returns:
            包含损失信息的字典
        """
        batch_size = images.shape[0]
        
        # 输入检查和预处理
        images = torch.clamp(images, 0.0, 1.0)
        
        # 编码到潜在空间
        latents = self.encode_to_latent(images)
        
        # 采样时间步
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, 
            (batch_size,), device=images.device
        )
        
        # 添加噪声
        noise = torch.randn_like(latents)
        noise = torch.clamp(noise, -2.0, 2.0)
        
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # CFG训练：随机丢弃类别标签
        if self.use_cfg and class_labels is not None:
            cfg_mask = torch.rand(batch_size, device=images.device) < self.cfg_dropout_prob
            class_labels = class_labels.clone()
            class_labels[cfg_mask] = -1  # 无条件标签
        
        # U-Net预测噪声
        predicted_noise = self.unet(noisy_latents, timesteps, class_labels)
        
        # 计算损失
        loss = F.mse_loss(predicted_noise, noise, reduction='mean')
        
        # 检查损失健康状况
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ 检测到异常损失，使用安全值")
            loss = torch.tensor(1.0, device=images.device, requires_grad=True)
        
        # 更新EMA
        if self.ema is not None and self.training:
            self.ema.update()
        
        return {
            'loss': loss,
            'predicted_noise': predicted_noise.detach(),
            'target_noise': noise.detach()
        }
    
    @torch.no_grad()
    def sample(
        self, 
        batch_size: int, 
        class_labels: Optional[torch.Tensor] = None,
        num_inference_steps: int = 75,
        guidance_scale: float = 8.5,
        eta: float = 0.3,
        use_ema: bool = False,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        DDIM/DDPM采样生成图像
        
        Args:
            batch_size: 批次大小
            class_labels: [batch_size] 类别标签
            num_inference_steps: 推理步数
            guidance_scale: CFG引导强度
            eta: DDIM随机性参数
            use_ema: 是否使用EMA权重
            verbose: 是否显示详细日志
            
        Returns:
            生成的图像 [batch_size, 3, H, W]
        """
        device = self.device
        
        # 使用EMA权重（如果可用且请求）
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        
        try:
            if verbose:
                print(f"🔄 开始采样: 步数={num_inference_steps}, CFG={guidance_scale}, EMA={'✓' if use_ema and self.ema else '✗'}")
            
            # 初始化随机噪声
            latents = torch.randn(batch_size, 256, 32, 32, device=device)
            
            # 设置推理调度器
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            
            # DDIM采样循环
            for i, t in enumerate(self.scheduler.timesteps):
                t_batch = t.unsqueeze(0).repeat(batch_size).to(device)
                
                if self.use_cfg and class_labels is not None and guidance_scale > 1.0:
                    # CFG: 预测有条件和无条件噪声
                    latent_model_input = torch.cat([latents] * 2)
                    t_input = torch.cat([t_batch] * 2)
                    
                    class_input = torch.cat([
                        class_labels,
                        torch.full_like(class_labels, -1)  # 无条件
                    ])
                    
                    noise_pred = self.unet(latent_model_input, t_input, class_input)
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    
                    # CFG引导
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    # 标准预测
                    noise_pred = self.unet(latents, t_batch, class_labels)
                
                # 调度器步骤
                if hasattr(self.scheduler, 'step'):
                    if isinstance(self.scheduler, DDIMScheduler):
                        result = self.scheduler.step(noise_pred, t, latents, eta=eta)
                    else:
                        result = self.scheduler.step(noise_pred, t, latents)
                    
                    latents = result.prev_sample if hasattr(result, 'prev_sample') else result[0]
                else:
                    # 简单的DDIM步骤
                    alpha_t = self.scheduler.alphas_cumprod[t].to(device)
                    alpha_prev = self.scheduler.alphas_cumprod[t-1].to(device) if t > 0 else torch.tensor(1.0, device=device)
                    
                    pred_x0 = (latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                    pred_x0 = torch.clamp(pred_x0, -3.0, 3.0)
                    
                    latents = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_pred
            
            # 解码到图像空间
            images = self.decode_from_latent(latents)
            
            if verbose:
                final_min, final_max = images.min().item(), images.max().item()
                final_mean = images.mean().item()
                print(f"✅ 采样完成: 范围[{final_min:.3f}, {final_max:.3f}], 均值{final_mean:.3f}")
            
            return images
            
        finally:
            # 恢复原始权重
            if use_ema and self.ema is not None:
                self.ema.restore()
    
    def get_loss_dict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """获取损失字典 (兼容训练循环)"""
        images = batch['image']
        class_labels = batch.get('label', None)
        
        return self(images, class_labels)

def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_stable_ldm(config: Dict[str, Any]) -> StableLDM:
    """创建StableLDM模型"""
    model = StableLDM(config)
    
    # 打印模型信息
    total_params = count_parameters(model.unet)
    print(f"🔥 轻量化U-Net参数量: {total_params:,} ({total_params/1e6:.1f}M)")
    
    vae_params = count_parameters(model.vae)
    print(f"❄️ VAE参数量: {vae_params:,} ({vae_params/1e6:.1f}M) [冻结]")
    
    print(f"🚀 总训练参数: {total_params:,} ({total_params/1e6:.1f}M)")
    
    return model 
