"""
Complete LDM Model - 基于CompVis/latent-diffusion
集成AutoencoderKL、U-Net和扩散过程的完整潜在扩散模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Dict, Any, Optional, Tuple, Union
from diffusers import AutoencoderKL
from ldm_unet import create_ldm_unet, LDMUNet
from ldm_diffusion import create_ldm_diffusion, LDMDiffusion


class LatentDiffusionModel(nn.Module):
    """
    完整的潜在扩散模型
    集成VAE编码器/解码器和扩散U-Net
    """
    def __init__(
        self,
        vae_checkpoint_path: str = None,
        unet_config: Dict[str, Any] = None,
        diffusion_config: Dict[str, Any] = None,
        device: str = 'cuda',
        scaling_factor: float = 0.18215,  # AutoencoderKL的标准缩放因子
    ):
        super().__init__()
        
        self.device = device
        self.scaling_factor = scaling_factor
        
        # 默认配置
        if unet_config is None:
            unet_config = {
                'in_channels': 4,
                'out_channels': 4,
                'model_channels': 320,
                'num_res_blocks': 2,
                'attention_resolutions': (4, 2, 1),
                'channel_mult': (1, 2, 4, 4),
                'num_classes': 31,
                'dropout': 0.0,
                'num_heads': 8,
                'use_scale_shift_norm': True,
                'resblock_updown': True,
            }
        
        if diffusion_config is None:
            diffusion_config = {
                'timesteps': 1000,
                'beta_schedule': 'cosine',
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'device': device
            }
        
        # 1. 初始化VAE（冻结参数）
        print("📦 加载AutoencoderKL...")
        self.vae = self._load_vae(vae_checkpoint_path)
        
        # 2. 初始化U-Net
        print("🔧 初始化U-Net...")
        self.unet = create_ldm_unet(**unet_config)
        
        # 3. 初始化扩散过程
        print("🌊 初始化扩散过程...")
        self.diffusion = create_ldm_diffusion(**diffusion_config)
        
        # 移动到设备
        self.to(device)
        
        # 存储配置
        self.unet_config = unet_config
        self.diffusion_config = diffusion_config
        
        print(f"✅ LDM初始化完成")
        print(f"   VAE缩放因子: {self.scaling_factor}")
        print(f"   U-Net输入通道: {self.unet.in_channels}")
        print(f"   扩散时间步数: {self.diffusion.timesteps}")

    def _load_vae(self, checkpoint_path: str = None) -> AutoencoderKL:
        """加载预训练的AutoencoderKL"""
        # 首先加载基础的Stable Diffusion VAE
        vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="vae"
        )
        
        # 如果提供了微调的检查点，加载微调权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                print(f"🔄 加载微调VAE权重: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # 尝试不同的键名来加载VAE状态字典
                vae_state_dict = None
                possible_keys = [
                    'vae_state_dict',    # 来自VAE微调脚本
                    'model_state_dict',  # 通用格式
                    'state_dict',        # 简化格式
                ]
                
                for key in possible_keys:
                    if key in checkpoint:
                        vae_state_dict = checkpoint[key]
                        print(f"✅ 使用键 '{key}' 加载VAE权重")
                        break
                
                if vae_state_dict is None:
                    # 检查是否整个checkpoint就是状态字典
                    if isinstance(checkpoint, dict) and any(
                        k.startswith(('encoder', 'decoder', 'quant_conv', 'post_quant_conv')) 
                        for k in checkpoint.keys()
                    ):
                        vae_state_dict = checkpoint
                        print("✅ 检查点直接是VAE状态字典格式")
                    else:
                        print(f"❌ 无法找到VAE状态字典，可用键: {list(checkpoint.keys())}")
                        raise KeyError("无法在检查点中找到VAE状态字典")
                
                # 加载状态字典
                vae.load_state_dict(vae_state_dict)
                print(f"✅ 微调VAE权重加载成功")
                
                # 如果检查点中有训练历史，打印一些信息
                if 'train_history' in checkpoint:
                    history = checkpoint['train_history']
                    if 'epoch' in history and len(history['epoch']) > 0:
                        last_epoch = history['epoch'][-1]
                        if 'recon_loss' in history and len(history['recon_loss']) > 0:
                            last_recon_loss = history['recon_loss'][-1]
                            print(f"   微调信息: Epoch {last_epoch}, 重建损失: {last_recon_loss:.4f}")
                        
            except Exception as e:
                print(f"❌ 微调VAE权重加载失败: {e}")
                print(f"🔄 使用预训练VAE权重")
        else:
            print(f"⚠️  未提供VAE checkpoint路径或文件不存在，使用预训练VAE权重")
        
        # 冻结VAE参数
        for param in vae.parameters():
            param.requires_grad = False
        vae.eval()
        
        print(f"🔒 VAE参数已冻结 ({sum(p.numel() for p in vae.parameters()):,} 参数)")
        
        return vae

    @torch.no_grad()
    def encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """
        将图像编码到潜在空间
        
        Args:
            images: [B, 3, H, W] 输入图像，范围[-1, 1]
            
        Returns:
            latents: [B, 4, h, w] 潜在表示，已经按scaling_factor缩放
        """
        self.vae.eval()
        
        # VAE编码
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample()  # [B, 4, h, w]
        
        # ⚠️ 重要修复：diffusers.AutoencoderKL的sample()方法并不会自动应用scaling_factor
        # 需要手动乘以scaling_factor来将潜变量缩放到合适的范围供扩散模型使用
        latents = latents * self.scaling_factor
        
        return latents

    @torch.no_grad()
    def decode_from_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """
        从潜在空间解码到图像
        
        Args:
            latents: [B, 4, h, w] 潜在表示（已经过scaling_factor缩放）
            
        Returns:
            images: [B, 3, H, W] 重建图像，范围[-1, 1]
        """
        self.vae.eval()
        
        # ⚠️ 重要修复：VAE解码前需要先除以scaling_factor，恢复原始潜变量范围
        # 因为VAE期望接收未缩放的潜变量
        unscaled_latents = latents / self.scaling_factor
        
        # VAE解码
        images = self.vae.decode(unscaled_latents).sample
        
        # 确保输出在正确范围
        images = torch.clamp(images, -1.0, 1.0)
        
        return images

    def forward(self, images: torch.Tensor, class_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        训练时的前向传播
        
        Args:
            images: [B, 3, H, W] 输入图像
            class_labels: [B] 类别标签（可选）
            
        Returns:
            损失字典
        """
        batch_size = images.shape[0]
        device = images.device
        
        # 1. 编码到潜在空间
        with torch.no_grad():
            latents = self.encode_to_latent(images)  # [B, 4, h, w]
        
        # 2. 随机采样时间步
        timesteps = torch.randint(
            0, self.diffusion.timesteps, (batch_size,), device=device, dtype=torch.long
        )
        
        # 3. 计算扩散损失
        losses = self.diffusion.training_losses(
            model=self.unet,
            x_start=latents,
            t=timesteps,
            class_labels=class_labels
        )
        
        return {
            'loss': losses['loss'].mean(),
            'pred_noise': losses['pred_noise'],
            'target_noise': losses['target_noise'],
            'latents': latents,
            'timesteps': timesteps
        }

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        class_labels: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_latents: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        从噪音采样生成图像
        
        Args:
            num_samples: 生成样本数量
            class_labels: [num_samples] 类别标签（可选）
            num_inference_steps: 推理步数
            eta: DDIM的随机性参数（0=确定性，1=随机）
            guidance_scale: 分类器自由引导强度
            generator: 随机数生成器
            return_latents: 是否返回潜在表示
            
        Returns:
            images: [num_samples, 3, H, W] 生成的图像
            latents: [num_samples, 4, h, w] 潜在表示（如果return_latents=True）
        """
        device = next(self.parameters()).device
        
        # 计算潜在空间的形状
        # 假设输入图像是256x256，VAE下采样8倍，得到32x32
        latent_shape = (num_samples, 4, 32, 32)
        
        # 如果使用分类器自由引导
        if guidance_scale > 1.0 and class_labels is not None:
            # 修复：使用num_classes作为无条件标记，而不是-1
            # 因为Embedding层无法处理负数索引
            unconditional_label = self.unet.num_classes  # 使用类别数作为无条件标记
            
            # 定义引导版本的模型
            def guided_unet(x, t, class_labels):
                batch_size = x.shape[0]
                
                # 创建无条件标签
                unconditional_labels = torch.full(
                    (batch_size,), unconditional_label, 
                    device=device, dtype=class_labels.dtype
                )
                
                # 条件预测
                cond_pred = self.unet(x, t, class_labels)
                
                # 无条件预测
                uncond_pred = self.unet(x, t, unconditional_labels)
                
                # 分类器自由引导
                guided_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
                
                return guided_pred
            
            model_fn = guided_unet
        else:
            model_fn = self.unet
        
        # 采样潜在表示
        latents = self.diffusion.sample(
            model=model_fn,
            shape=latent_shape,
            class_labels=class_labels,
            num_inference_steps=num_inference_steps,
            eta=eta,
            generator=generator
        )
        
        # 解码到图像
        images = self.decode_from_latent(latents)
        
        if return_latents:
            return images, latents
        else:
            return images

    def configure_optimizers(self, lr: float = 1e-4, weight_decay: float = 0.01):
        """配置优化器"""
        # 只优化U-Net参数，VAE保持冻结
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        return optimizer

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        optimizer_state: Optional[Dict] = None,
        scheduler_state: Optional[Dict] = None,
        metrics: Optional[Dict] = None
    ):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'unet_state_dict': self.unet.state_dict(),
            'unet_config': self.unet_config,
            'diffusion_config': self.diffusion_config,
            'scaling_factor': self.scaling_factor,
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if scheduler_state:
            checkpoint['scheduler_state_dict'] = scheduler_state
        if metrics:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, filepath)
        print(f"💾 检查点已保存: {filepath}")

    def load_checkpoint(self, filepath: str, load_optimizer: bool = False):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 加载U-Net权重
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        
        print(f"✅ LDM检查点加载成功: {filepath}")
        print(f"   训练轮次: {checkpoint.get('epoch', 'Unknown')}")
        
        if load_optimizer:
            return checkpoint.get('optimizer_state_dict'), checkpoint.get('scheduler_state_dict')
        
        return None, None


def create_ldm_model(
    vae_checkpoint_path: str = None,
    unet_config: Dict[str, Any] = None,
    diffusion_config: Dict[str, Any] = None,
    device: str = 'cuda'
) -> LatentDiffusionModel:
    """
    创建完整LDM模型的工厂函数
    """
    return LatentDiffusionModel(
        vae_checkpoint_path=vae_checkpoint_path,
        unet_config=unet_config,
        diffusion_config=diffusion_config,
        device=device
    )


if __name__ == "__main__":
    # 测试LDM模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🧪 测试LDM模型...")
    
    # 创建模型
    model = create_ldm_model(device=device)
    
    # 测试编码-解码
    print("\n📊 测试VAE编码-解码...")
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256, device=device)  # 随机图像
    
    with torch.no_grad():
        # 编码
        latents = model.encode_to_latent(images)
        print(f"原始图像形状: {images.shape}")
        print(f"潜在表示形状: {latents.shape}")
        
        # 解码
        reconstructed = model.decode_from_latent(latents)
        print(f"重建图像形状: {reconstructed.shape}")
        
        # 计算重建误差
        mse = F.mse_loss(images, reconstructed)
        print(f"重建MSE: {mse.item():.4f}")
    
    # 测试训练前向传播
    print("\n🎯 测试训练前向传播...")
    class_labels = torch.randint(0, 31, (batch_size,), device=device)
    
    loss_dict = model(images, class_labels)
    print(f"训练损失: {loss_dict['loss'].item():.4f}")
    print(f"预测噪音形状: {loss_dict['pred_noise'].shape}")
    
    # 测试采样（少量步数）
    print("\n🎨 测试图像生成...")
    with torch.no_grad():
        generated_images = model.sample(
            num_samples=1,
            class_labels=torch.tensor([0], device=device),
            num_inference_steps=10  # 快速测试
        )
        print(f"生成图像形状: {generated_images.shape}")
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📈 模型统计:")
    print(f"   总参数数: {total_params:,}")
    print(f"   可训练参数数: {trainable_params:,}")
    print(f"   VAE参数数: {sum(p.numel() for p in model.vae.parameters()):,} (冻结)")
    print(f"   U-Net参数数: {sum(p.numel() for p in model.unet.parameters()):,} (可训练)")
    
    print("\n✅ LDM模型测试成功!") 
