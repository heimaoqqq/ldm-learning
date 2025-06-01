"""
简化的增强版U-Net实现
包含多层注意力机制，但结构更简单可靠
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class TimestepEmbedding(nn.Module):
    """时间步嵌入"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class ClassEmbedding(nn.Module):
    """类别嵌入"""
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, embed_dim)

    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        class_ids = torch.where(class_ids == -1, self.embedding.num_embeddings - 1, class_ids)
        return self.embedding(class_ids)

class MultiLayerAttention(nn.Module):
    """多层注意力块 - 🚀 真正的Transformer层"""
    def __init__(self, channels: int, num_heads: int = 8, transformer_depth: int = 2, dropout: float = 0.0):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.transformer_depth = transformer_depth
        
        # 输入规范化
        self.norm = nn.GroupNorm(32, channels)
        
        # 多层Transformer
        self.layers = nn.ModuleList()
        for _ in range(transformer_depth):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(channels),
                'attn': nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True),
                'norm2': nn.LayerNorm(channels),
                'mlp': nn.Sequential(
                    nn.Linear(channels, channels * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(channels * 4, channels),
                    nn.Dropout(dropout)
                )
            }))
        
        # 输出投影
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        residual = x
        B, C, H, W = x.shape
        
        # 规范化
        x = self.norm(x)
        
        # 重塑为序列格式 [B, H*W, C]
        x = x.view(B, C, H * W).transpose(1, 2)
        
        # 多层Transformer
        for layer in self.layers:
            # 自注意力
            x_norm = layer['norm1'](x)
            attn_out, _ = layer['attn'](x_norm, x_norm, x_norm)
            x = x + attn_out
            
            # MLP
            x_norm = layer['norm2'](x)
            mlp_out = layer['mlp'](x_norm)
            x = x + mlp_out
        
        # 重塑回空间格式 [B, C, H, W]
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # 输出投影
        x = self.proj_out(x)
        
        # 残差连接
        return x + residual

class EnhancedResBlock(nn.Module):
    """增强的ResNet块"""
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, 
                 use_attention: bool = False, num_heads: int = 8, transformer_depth: int = 2):
        super().__init__()
        
        # 第一个卷积层
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # 时间嵌入
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        
        # 第二个卷积层
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 跳跃连接
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # 可选的多层注意力
        self.use_attention = use_attention
        if use_attention:
            self.attention = MultiLayerAttention(out_channels, num_heads, transformer_depth)
        
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
        
        # 跳跃连接
        x = x + self.skip_conv(residual)
        
        # 多层注意力（如果启用）
        if self.use_attention:
            x = self.attention(x)
        
        return x

class SimpleEnhancedUNet(nn.Module):
    """
    简化的增强版U-Net
    包含真正的多层Transformer，但结构简单可靠
    """
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        model_channels: int = 192,
        num_classes: int = 31,
        time_embed_dim: int = 768,
        transformer_depth: int = 2,  # 🚀 真正的Transformer深度
        num_heads: int = 8,
        use_attention_layers: list = [1, 2],  # 在哪些层使用注意力
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.transformer_depth = transformer_depth
        
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
        
        # 编码器（下采样）- 🚀 多层注意力
        self.down1 = EnhancedResBlock(
            model_channels, model_channels * 2, time_embed_dim,
            use_attention=1 in use_attention_layers,
            num_heads=num_heads, transformer_depth=transformer_depth
        )
        self.down1_pool = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
        
        self.down2 = EnhancedResBlock(
            model_channels * 2, model_channels * 4, time_embed_dim,
            use_attention=2 in use_attention_layers,
            num_heads=num_heads, transformer_depth=transformer_depth
        )
        self.down2_pool = nn.Conv2d(model_channels * 4, model_channels * 4, 3, stride=2, padding=1)
        
        # 中间层 - 🚀 强制使用多层注意力
        self.middle1 = EnhancedResBlock(
            model_channels * 4, model_channels * 4, time_embed_dim,
            use_attention=True, num_heads=num_heads, transformer_depth=transformer_depth
        )
        self.middle2 = EnhancedResBlock(
            model_channels * 4, model_channels * 4, time_embed_dim,
            use_attention=True, num_heads=num_heads, transformer_depth=transformer_depth
        )
        
        # 解码器（上采样）- 🚀 多层注意力
        self.up2 = nn.ConvTranspose2d(model_channels * 4, model_channels * 4, 4, stride=2, padding=1)
        self.up2_res = EnhancedResBlock(
            model_channels * 8, model_channels * 2, time_embed_dim,  # 8=4+4 (concat)
            use_attention=2 in use_attention_layers,
            num_heads=num_heads, transformer_depth=transformer_depth
        )
        
        self.up1 = nn.ConvTranspose2d(model_channels * 2, model_channels * 2, 4, stride=2, padding=1)
        self.up1_res = EnhancedResBlock(
            model_channels * 4, model_channels, time_embed_dim,  # 4=2+2 (concat)
            use_attention=1 in use_attention_layers,
            num_heads=num_heads, transformer_depth=transformer_depth
        )
        
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
        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
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
        h = self.input_conv(x)  # [B, 192, 32, 32]
        h0 = h
        
        # 下采样路径
        h1 = self.down1(h, time_emb)  # [B, 384, 32, 32] 🚀 可能包含Transformer
        h1_pooled = self.down1_pool(h1)  # [B, 384, 16, 16]
        
        h2 = self.down2(h1_pooled, time_emb)  # [B, 768, 16, 16] 🚀 可能包含Transformer
        h2_pooled = self.down2_pool(h2)  # [B, 768, 8, 8]
        
        # 中间层 - 🚀 强制使用多层Transformer
        h = self.middle1(h2_pooled, time_emb)  # [B, 768, 8, 8]
        h = self.middle2(h, time_emb)  # [B, 768, 8, 8]
        
        # 上采样路径
        h = self.up2(h)  # [B, 768, 16, 16]
        h = torch.cat([h, h2], dim=1)  # [B, 1536, 16, 16]
        h = self.up2_res(h, time_emb)  # [B, 384, 16, 16] 🚀 可能包含Transformer
        
        h = self.up1(h)  # [B, 384, 32, 32]
        h = torch.cat([h, h1], dim=1)  # [B, 768, 32, 32]
        h = self.up1_res(h, time_emb)  # [B, 192, 32, 32] 🚀 可能包含Transformer
        
        # 输出层
        h = self.output(h)  # [B, 256, 32, 32]
        
        return h

def create_simple_enhanced_unet(config: dict) -> SimpleEnhancedUNet:
    """根据配置创建简化的增强U-Net"""
    unet_config = config['unet']
    
    return SimpleEnhancedUNet(
        in_channels=unet_config['in_channels'],
        out_channels=unet_config['out_channels'],
        model_channels=unet_config['model_channels'],
        num_classes=unet_config['num_classes'],
        time_embed_dim=unet_config['time_embed_dim'],
        transformer_depth=unet_config.get('transformer_depth', 2),  # 🚀 真正使用！
        num_heads=unet_config.get('num_heads', 8),
        use_attention_layers=unet_config.get('use_attention_layers', [1, 2]),
    ) 
