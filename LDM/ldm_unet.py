"""
LDM U-Net Model - 基于CompVis/latent-diffusion
兼容AutoencoderKL的4通道潜变量输入
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    创建时间步嵌入，用于在U-Net中注入时间信息
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Downsample(nn.Module):
    """下采样层"""
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.use_conv:
            return self.conv(x)
        else:
            return self.pool(x)


class Upsample(nn.Module):
    """上采样层"""
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        dropout=0.0,
        use_scale_shift_norm=True,
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * out_channels if use_scale_shift_norm else out_channels),
        )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, time_emb):
        h = self.in_layers(x)
        
        time_emb = self.time_emb_proj(time_emb)
        while len(time_emb.shape) < len(h.shape):
            time_emb = time_emb[..., None]
        
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(time_emb, 2, dim=1)
            h = self.out_layers[0](h) * (1 + scale) + shift
            h = self.out_layers[1:](h)
        else:
            h = h + time_emb
            h = self.out_layers(h)
        
        return h + self.skip_connection(x)


class AttentionBlock(nn.Module):
    """自注意力块"""
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        residual = x
        
        x = self.norm(x)
        x = x.view(batch_size, channels, height * width)
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, self.num_heads, channels // self.num_heads, height * width)
        k = k.view(batch_size, self.num_heads, channels // self.num_heads, height * width)
        v = v.view(batch_size, self.num_heads, channels // self.num_heads, height * width)
        
        # Attention
        scale = (channels // self.num_heads) ** -0.5
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * scale
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.contiguous().view(batch_size, channels, height * width)
        
        out = self.proj_out(out)
        out = out.view(batch_size, channels, height, width)
        
        return out + residual


class LDMUNet(nn.Module):
    """
    基于CompVis/latent-diffusion的U-Net模型
    专门设计用于处理AutoencoderKL的4通道潜变量
    """
    def __init__(
        self,
        in_channels=4,           # AutoencoderKL输出4通道
        out_channels=4,          # 预测4通道噪音
        model_channels=320,      # 基础通道数，参考Stable Diffusion
        num_res_blocks=2,
        attention_resolutions=(4, 2, 1),  # 在哪些分辨率使用注意力
        channel_mult=(1, 2, 4, 4),       # 每层的通道倍数
        num_classes=31,          # 类别数，用于条件生成
        dropout=0.0,
        num_heads=8,
        use_scale_shift_norm=True,
        resblock_updown=True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_heads = num_heads
        
        # 时间嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # 类别嵌入（如果需要条件生成）
        if num_classes is not None:
            # 修复：添加一个额外的嵌入用于无条件生成
            # 索引0到num_classes-1为正常类别，索引num_classes为无条件标记
            self.class_embed = nn.Embedding(num_classes + 1, time_embed_dim)
        
        # 输入投影
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        # 下采样路径
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(
                        ch, mult * model_channels, time_embed_dim,
                        dropout=dropout, use_scale_shift_norm=use_scale_shift_norm
                    )
                ]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(Downsample(ch, use_conv=resblock_updown))
                input_block_chans.append(ch)
                ds *= 2
        
        # 中间块
        self.middle_block = nn.Sequential(
            ResidualBlock(
                ch, ch, time_embed_dim,
                dropout=dropout, use_scale_shift_norm=use_scale_shift_norm
            ),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(
                ch, ch, time_embed_dim,
                dropout=dropout, use_scale_shift_norm=use_scale_shift_norm
            ),
        )
        
        # 上采样路径
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResidualBlock(
                        ch + ich, mult * model_channels, time_embed_dim,
                        dropout=dropout, use_scale_shift_norm=use_scale_shift_norm
                    )
                ]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, use_conv=resblock_updown))
                    ds //= 2
                
                self.output_blocks.append(nn.Sequential(*layers))
        
        # 输出投影
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, class_labels=None):
        """
        前向传播
        
        Args:
            x: [B, 4, H, W] 潜变量
            timesteps: [B] 时间步
            class_labels: [B] 类别标签（可选）
        
        Returns:
            [B, 4, H, W] 预测的噪音
        """
        # 时间嵌入
        time_emb = timestep_embedding(timesteps, self.model_channels)
        time_emb = self.time_embed(time_emb)
        
        # 类别嵌入
        if class_labels is not None and self.num_classes is not None:
            class_emb = self.class_embed(class_labels)
            time_emb = time_emb + class_emb
        
        # 下采样
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, ResidualBlock):
                        h = layer(h, time_emb)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)
        
        # 中间处理
        for layer in self.middle_block:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)
        
        # 上采样
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, time_emb)
                else:
                    h = layer(h)
        
        return self.out(h)


def create_ldm_unet(
    in_channels=4,
    out_channels=4,
    model_channels=320,
    num_res_blocks=2,
    attention_resolutions=(4, 2, 1),
    channel_mult=(1, 2, 4, 4),
    num_classes=31,
    dropout=0.0,
    num_heads=8,
    use_scale_shift_norm=True,
    resblock_updown=True,
):
    """
    创建LDM U-Net模型的工厂函数
    """
    return LDMUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        channel_mult=channel_mult,
        num_classes=num_classes,
        dropout=dropout,
        num_heads=num_heads,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
    )


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_ldm_unet().to(device)
    
    # 测试输入
    batch_size = 2
    x = torch.randn(batch_size, 4, 32, 32).to(device)  # 4通道，32x32潜变量
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    class_labels = torch.randint(0, 31, (batch_size,)).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = model(x, timesteps, class_labels)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("✅ LDM U-Net模型测试成功!") 
