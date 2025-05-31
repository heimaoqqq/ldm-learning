"""
简化的轻量化U-Net实现
基于Stable Diffusion但更简单、更可靠
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class TimestepEmbedding(nn.Module):
    """时间步嵌入层"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.cos(embeddings), torch.sin(embeddings)], dim=1)
        return embeddings

class ClassEmbedding(nn.Module):
    """类别嵌入层"""
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, embed_dim)  # +1 for unconditional
        
    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        # 将-1映射到最后一个索引（无条件）
        class_ids = torch.where(class_ids == -1, self.embedding.num_embeddings - 1, class_ids)
        return self.embedding(class_ids)

class ResBlock(nn.Module):
    """简化的ResNet块"""
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # 第一层
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        
        # 时间嵌入
        time_emb = F.silu(time_emb)
        time_emb = self.time_proj(time_emb)[:, :, None, None]
        x = x + time_emb
        
        # 第二层
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        
        return x + self.skip_conv(residual)

class AttentionBlock(nn.Module):
    """简化的注意力块"""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.norm(x)
        qkv = self.to_qkv(x)
        
        b, c, h, w = x.shape
        qkv = qkv.view(b, 3, c, h * w)
        q, k, v = qkv.unbind(1)
        
        # 简化的自注意力
        attn = torch.matmul(q.transpose(-2, -1), k) / (c ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.view(b, c, h, w)
        
        out = self.to_out(out)
        return out + residual

class SimpleUNet(nn.Module):
    """
    简化的轻量化U-Net
    适配256通道潜在空间
    """
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        model_channels: int = 128,
        num_classes: int = 31,
        time_embed_dim: int = 512,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        
        # 时间嵌入
        self.time_embedding = TimestepEmbedding(model_channels)
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # 类别嵌入
        self.class_embedding = ClassEmbedding(num_classes, time_embed_dim)
        
        # 输入投影
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # 编码器（下采样）
        self.down1 = ResBlock(model_channels, model_channels * 2, time_embed_dim)
        self.down1_pool = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
        
        self.down2 = ResBlock(model_channels * 2, model_channels * 4, time_embed_dim)
        self.down2_pool = nn.Conv2d(model_channels * 4, model_channels * 4, 3, stride=2, padding=1)
        
        # 中间层
        self.middle1 = ResBlock(model_channels * 4, model_channels * 4, time_embed_dim)
        self.middle_attn = AttentionBlock(model_channels * 4)
        self.middle2 = ResBlock(model_channels * 4, model_channels * 4, time_embed_dim)
        
        # 解码器（上采样）
        self.up2 = nn.ConvTranspose2d(model_channels * 4, model_channels * 4, 4, stride=2, padding=1)
        self.up2_res = ResBlock(model_channels * 8, model_channels * 2, time_embed_dim)  # 8=4+4 (concat)
        
        self.up1 = nn.ConvTranspose2d(model_channels * 2, model_channels * 2, 4, stride=2, padding=1)
        self.up1_res = ResBlock(model_channels * 4, model_channels, time_embed_dim)  # 4=2+2 (concat)
        
        # 输出层
        self.output = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GroupNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, 
                class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, height, width] 潜在表示
            timesteps: [batch_size] 时间步
            class_labels: [batch_size] 类别标签，可选
        
        Returns:
            noise_pred: [batch_size, out_channels, height, width] 预测的噪声
        """
        
        # 时间嵌入
        time_emb = self.time_embedding(timesteps)
        time_emb = self.time_embed(time_emb)
        
        # 类别嵌入（如果有）
        if class_labels is not None:
            class_emb = self.class_embedding(class_labels)
            time_emb = time_emb + class_emb
        
        # 输入投影
        h = self.input_conv(x)  # [B, 128, 32, 32]
        h0 = h
        
        # 下采样路径
        h1 = self.down1(h, time_emb)  # [B, 256, 32, 32]
        h1_pooled = self.down1_pool(h1)  # [B, 256, 16, 16]
        
        h2 = self.down2(h1_pooled, time_emb)  # [B, 512, 16, 16]
        h2_pooled = self.down2_pool(h2)  # [B, 512, 8, 8]
        
        # 中间层
        h = self.middle1(h2_pooled, time_emb)  # [B, 512, 8, 8]
        h = self.middle_attn(h)  # [B, 512, 8, 8]
        h = self.middle2(h, time_emb)  # [B, 512, 8, 8]
        
        # 上采样路径
        h = self.up2(h)  # [B, 512, 16, 16]
        h = torch.cat([h, h2], dim=1)  # [B, 1024, 16, 16]
        h = self.up2_res(h, time_emb)  # [B, 256, 16, 16]
        
        h = self.up1(h)  # [B, 256, 32, 32]
        h = torch.cat([h, h1], dim=1)  # [B, 512, 32, 32]
        h = self.up1_res(h, time_emb)  # [B, 128, 32, 32]
        
        # 输出层
        h = self.output(h)  # [B, 256, 32, 32]
        
        return h

def create_simple_unet(config: dict) -> SimpleUNet:
    """根据配置创建简化U-Net"""
    unet_config = config['unet']
    
    return SimpleUNet(
        in_channels=unet_config['in_channels'],
        out_channels=unet_config['out_channels'],
        model_channels=unet_config['model_channels'],
        num_classes=unet_config['num_classes'],
        time_embed_dim=unet_config['time_embed_dim'],
    ) 
