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
from adv_vq_vae import AdvVQVAE

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
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        scale_factor: float = 1.0,
        vae_scale_factor: float = 0.18215,
        latent_scale_factor: float = 1.0
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
        
        # 存储配置信息用于输出
        self.vae_latent_dim = vae_config.get('latent_dim', 256)
        
        self.scale_factor = scale_factor
        self.vae_scale_factor = vae_scale_factor
        self.latent_scale_factor = latent_scale_factor
        
        # 注释掉调试相关变量，因为调试打印已被注释
        # 新增：用于控制epoch级别调试打印的属性
        # self._last_printed_epoch_debug = -1
        # self._prints_this_epoch_debug = 0
        
        print(f"✅ VAE-LDM 初始化完成")
        print(f"   VAE潜在维度: {self.vae_latent_dim}")
        print(f"   U-Net类别数: {self.unet.num_classes}")
        print(f"   扩散步数: {self.diffusion.num_timesteps}")
        
    def _load_vae(self, vae_config: Dict[str, Any], checkpoint_path: str) -> AdvVQVAE:
        """加载预训练的VAE"""
        vae = AdvVQVAE(**vae_config)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                print(f"🔍 检查点文件包含 {len(checkpoint)} 个键")
                
                # 尝试不同的键名来加载模型状态
                state_dict = None
                
                # 尝试常见的键名
                possible_keys = [
                    'model_state_dict',  # 标准格式
                    'state_dict',        # 简化格式
                    'model',             # 另一种格式
                    'vae_state_dict',    # VAE特定格式
                ]
                
                for key in possible_keys:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        print(f"✅ 使用键 '{key}' 加载VAE权重")
                        break
                
                # 如果没有找到合适的键，检查是否直接是状态字典
                if state_dict is None:
                    # 检查是否整个checkpoint就是状态字典
                    if isinstance(checkpoint, dict) and any(k.startswith(('encoder', 'decoder', 'quantize', 'quant_conv', 'post_quant_conv')) for k in checkpoint.keys()):
                        state_dict = checkpoint
                        print("✅ 检查点直接是状态字典格式")
                    else:
                        print(f"❌ 无法找到VAE状态字典，可用键: {list(checkpoint.keys())}")
                        raise KeyError(f"无法在检查点中找到模型状态字典。可用键: {list(checkpoint.keys())}")
                
                # 加载状态字典
                vae.load_state_dict(state_dict)
                print(f"✅ VAE 权重加载成功: {checkpoint_path}")
                
            except Exception as e:
                print(f"❌ VAE加载失败: {e}")
                print(f"🔄 将使用随机初始化的VAE权重")
                print("   警告：这可能导致训练性能下降")
        else:
            print(f"⚠️  VAE checkpoint 未找到: {checkpoint_path}")
            print("🔄 将使用随机初始化的VAE权重")
            
        # 冻结VAE参数
        for param in vae.parameters():
            param.requires_grad = False
        vae.eval()
        
        print(f"🔒 VAE参数已冻结（{sum(p.numel() for p in vae.parameters())} 参数）")
        
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
        
        # 尝试使用VAE的编码方法
        try:
            # 方法1: 使用encode方法 (最直接)
            if hasattr(self.vae, 'encode'):
                latents_raw = self.vae.encode(images)
            # 方法2: 使用编码器 + 量化器
            elif hasattr(self.vae, 'encoder') and hasattr(self.vae, 'vq'):
                encoded = self.vae.encoder(images)
                quantized, vq_loss, perplexity = self.vae.vq(encoded)
                latents_raw = quantized
            # 方法3: 使用forward，解析返回值（最后备用）
            else:
                vae_output = self.vae(images)
                if isinstance(vae_output, tuple) and len(vae_output) >= 3:
                    # VAE forward通常返回 (reconstruction, commitment_loss, codebook_loss)
                    # 我们需要用编码器+量化获取潜在表示
                    encoded = self.vae.encoder(images)
                    quantized, _, _ = self.vae.vq(encoded)
                    latents_raw = quantized
                else:
                    raise ValueError("Unexpected VAE output format")
            
            # 应用潜变量缩放因子
            scaled_latents = latents_raw * self.latent_scale_factor
            return scaled_latents
                     
        except Exception as e:
            print(f"❌ VAE编码失败: {e}")
            # 返回一个合适形状的随机张量作为备用
            batch_size = images.shape[0]
            latent_h, latent_w = images.shape[2] // 8, images.shape[3] // 8  # 假设8倍下采样
            backup_latents = torch.randn(batch_size, 256, latent_h, latent_w, device=images.device)
            print(f"🔄 使用备用随机潜在表示，形状: {backup_latents.shape}")
            return backup_latents
    
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
    
    def forward(self, images: torch.Tensor, class_labels: Optional[torch.Tensor] = None, current_epoch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        训练前向传播
        
        Args:
            images: [B, 3, H, W] 输入图像
            class_labels: [B] 类别标签
            current_epoch: 当前训练的epoch号（用于控制调试打印）
            
        Returns:
            loss_dict: 包含各种损失的字典
        """
        # 1. 编码到潜在空间
        with torch.no_grad():
            latents = self.encode_to_latent(images)
        
        # <--- 注释掉调试打印逻辑，使日志更简洁 --->
        # if current_epoch is not None and current_epoch < 10: # 检查是否在前10个epoch
        #     if self._last_printed_epoch_debug != current_epoch: # 如果是新的epoch（在前10个中）
        #         self._last_printed_epoch_debug = current_epoch
        #         self._prints_this_epoch_debug = 0 # 重置当前epoch的打印计数器
        #     
        #     if self._prints_this_epoch_debug < 5: # 打印当前epoch的前5个批次
        #         # 使用 current_epoch + 1 使其从1开始计数epoch，_prints_this_epoch_debug 从0开始，所以也+1
        #         print(f"Debug (Epoch {current_epoch + 1}, Batch In Epoch {self._prints_this_epoch_debug + 1}): Latents fed to U-Net - Mean: {latents.mean().item():.4f}, Std: {latents.std().item():.4f}, Min: {latents.min().item():.4f}, Max: {latents.max().item():.4f}")
        #         self._prints_this_epoch_debug += 1
        # <--- 调试打印结束 --->
        
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
    unet_params_for_model = {
        'image_size': image_size,             # 来自函数参数
        'in_channels': 256,                  # 示例值，可以配置
        'model_channels': 192,                # 示例值，可以配置
        'out_channels': 256,                  # 示例值，可以配置
        'num_res_blocks': 3,                  # 示例值
        'attention_resolutions': (4, 8, 16),  # 示例值，基于32x32潜在空间
        'dropout': 0.1,
        'channel_mult': (1, 2, 3, 4),       # 示例值
        'num_classes': num_classes,           # 来自函数参数
        'use_checkpoint': False,
        'num_heads': 4,
        'num_heads_upsample': -1,
        'use_scale_shift_norm': True  # <--- 确保这里是 True
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
    
    # <<<< 修改：根据训练中的观察调整缩放因子 >>>>
    # 原始计算 (基于单批次): 0.0616 (来自 1/16.2432)
    # 训练中观察到的 scaled_std ≈ 0.54 时，表示 actual_raw_std ≈ 0.54 / 0.0616 ≈ 8.77
    # 新的缩放因子 = 1 / actual_raw_std = 1 / 8.77 ≈ 0.1140
    calculated_latent_scale_factor = 0.114 
    # <<<< 结束修改 >>>>

    model = VAELatentDiffusionModel(
        vae_config=vae_config,
        unet_config=unet_params_for_model,
        diffusion_config=diffusion_config,
        vae_checkpoint_path=vae_checkpoint_path,
        device=device,
        latent_scale_factor=calculated_latent_scale_factor # <--- 传递更新后的缩放因子
    )
    
    return model 
