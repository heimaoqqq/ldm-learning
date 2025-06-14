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
            
            # 1. 先应用基础归一化：确保潜在表示接近标准正态分布
            latents_normalized = self._normalize_latents(latents_raw)
            
            # 2. 直接返回归一化后的结果，不再应用额外的缩放因子
            # 因为缩放因子应该在扩散训练中自动学习，而不是预先固定
            return latents_normalized
                     
        except Exception as e:
            print(f"❌ VAE编码失败: {e}")
            # 返回一个合适形状的随机张量作为备用
            batch_size = images.shape[0]
            latent_h, latent_w = images.shape[2] // 8, images.shape[3] // 8  # 假设8倍下采样
            backup_latents = torch.randn(batch_size, 256, latent_h, latent_w, device=images.device)
            print(f"🔄 使用备用随机潜在表示，形状: {backup_latents.shape}")
            return backup_latents
    
    def _normalize_latents(self, latents_raw: torch.Tensor) -> torch.Tensor:
        """
        归一化潜在表示，使其接近标准正态分布
        专门处理严重偏斜和异常分布的情况
        
        Args:
            latents_raw: 原始潜在表示
            
        Returns:
            normalized_latents: 归一化后的潜在表示
        """
        # 对于严重偏斜的分布，使用Box-Cox变换 + 鲁棒归一化
        
        # 1. 处理非负约束（VAE输出可能全为非负）
        # 添加小的偏移确保所有值为正，以便进行对数变换
        epsilon = 1e-6
        shifted = latents_raw + epsilon
        
        # 2. 对数变换减少右偏度
        # 这可以有效处理指数分布或伽马分布的数据
        log_transformed = torch.log(shifted + 1.0)  # log(1+x)更稳定
        
        # 3. 计算鲁棒统计量（使用分位数而非均值/方差）
        flat_data = log_transformed.flatten()
        
        # 使用更宽的分位数范围处理极值
        q01 = torch.quantile(flat_data, 0.01)
        q99 = torch.quantile(flat_data, 0.99)
        q_median = torch.quantile(flat_data, 0.5)
        
        # 使用分位数距离作为尺度参数
        scale = (q99 - q01) / 4.65  # 4.65 ≈ 2 * 2.326 (99%分位数对应的z分数)
        
        # 4. 鲁棒中心化和缩放
        centered = log_transformed - q_median
        scaled = centered / (scale + 1e-8)
        
        # 5. 使用erf函数进行软归一化，将任意分布映射到近似正态分布
        # erf函数具有S形曲线，可以有效压缩极值
        normalized = torch.erf(scaled / 1.4142) * 2.0  # 1.4142 = sqrt(2)
        
        # 6. 最终标准化确保严格的均值=0，标准差=1
        final_mean = normalized.mean()
        final_std = normalized.std()
        normalized = (normalized - final_mean) / (final_std + 1e-8)
        
        return normalized
    
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
        
        # 1. 直接使用输入的潜在表示，不再进行缩放因子的逆操作
        # 因为编码时已经不再应用缩放因子
        
        # 2. VAE解码（直接使用归一化后的潜在表示）
        decoded = self.vae.decode(latents)
        
        # 3. 确保输出在正确范围 [-1, 1]
        # VAE decode输出已经通过tanh，但为了安全起见再次clamp
        decoded = torch.clamp(decoded, -1.0, 1.0)
        
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
        # 1. 编码到潜在空间 - 分步进行以便调试
        with torch.no_grad():
            # 首先获取原始VAE输出
            try:
                if hasattr(self.vae, 'encode'):
                    raw_latents = self.vae.encode(images)
                elif hasattr(self.vae, 'encoder') and hasattr(self.vae, 'vq'):
                    encoded = self.vae.encoder(images)
                    quantized, vq_loss, perplexity = self.vae.vq(encoded)
                    raw_latents = quantized
                else:
                    encoded = self.vae.encoder(images)
                    quantized, _, _ = self.vae.vq(encoded)
                    raw_latents = quantized
                
                # 然后进行归一化
                latents = self._normalize_latents(raw_latents)
                
            except Exception as e:
                print(f"❌ VAE编码失败: {e}")
                batch_size = images.shape[0]
                latent_h, latent_w = images.shape[2] // 8, images.shape[3] // 8
                raw_latents = torch.randn(batch_size, 256, latent_h, latent_w, device=images.device)
                latents = self._normalize_latents(raw_latents)
        
        # 调试输出：前10个epoch显示潜在表示统计信息
        if current_epoch is not None and current_epoch < 10:
            if not hasattr(self, '_debug_batch_count'):
                self._debug_batch_count = {}
            
            if current_epoch not in self._debug_batch_count:
                self._debug_batch_count[current_epoch] = 0
            
            if self._debug_batch_count[current_epoch] < 3:  # 每个epoch只显示前3个batch
                print(f"📊 Epoch {current_epoch+1} Batch {self._debug_batch_count[current_epoch]+1}:")
                print(f"   原始VAE输出: 均值={raw_latents.mean().item():.4f}, 标准差={raw_latents.std().item():.4f}")
                print(f"   原始VAE范围: [{raw_latents.min().item():.4f}, {raw_latents.max().item():.4f}]")
                print(f"   归一化后: 均值={latents.mean().item():.4f}, 标准差={latents.std().item():.4f}")
                print(f"   归一化范围: [{latents.min().item():.4f}, {latents.max().item():.4f}]")
                
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
        'decay': 0.99,  # 添加缺失的decay参数
        'groups': 32,   # 修正：从1改为32，与VAE训练配置匹配
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
