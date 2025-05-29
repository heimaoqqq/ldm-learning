import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000):
    """
    创建正弦时间步嵌入
    Args:
        timesteps: 1-D Tensor of N 整数
        embedding_dim: 嵌入维度
        max_period: 控制最小频率
    """
    half = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TimestepEmbedding(nn.Module):
    """时间步嵌入层"""
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
        
        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "mish":
            self.act = nn.Mish()
        else:
            self.act = nn.ReLU()

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample

class Downsample(nn.Module):
    """下采样层"""
    def __init__(self, channels: int, use_conv: bool = True, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=1)
        else:
            self.conv = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels
        return self.conv(hidden_states)

class Upsample(nn.Module):
    """上采样层"""
    def __init__(self, channels: int, use_conv: bool = True, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states

class ResnetBlock(nn.Module):
    """残差块"""
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        time_embedding_norm: str = "default",
        time_embedding_dim: Optional[int] = None,
        groups: int = 32,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if time_embedding_dim is not None:
            if time_embedding_norm == "default":
                time_embedding_norm = "scale_shift" if use_scale_shift_norm else "default"

            if time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(time_embedding_dim, 2 * out_channels)
            else:
                self.time_emb_proj = nn.Linear(time_embedding_dim, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = nn.SiLU()

        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, input_tensor: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if time_emb is not None:
            time_emb = self.nonlinearity(time_emb)
            time_emb = self.time_emb_proj(time_emb)[:, :, None, None]

            if self.use_scale_shift_norm:
                scale, shift = time_emb.chunk(2, dim=1)
                hidden_states = self.norm2(hidden_states)
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = self.nonlinearity(hidden_states)
            else:
                hidden_states = self.norm2(hidden_states)
                hidden_states = hidden_states + time_emb
                hidden_states = self.nonlinearity(hidden_states)
        else:
            hidden_states = self.norm2(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        return self.skip_connection(input_tensor) + hidden_states

class AttentionBlock(nn.Module):
    """注意力块"""
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_groups: int = 32,
        encoder_hidden_states_channels: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)

        self.num_heads = num_heads
        self.head_size = channels // num_heads

        # 交叉注意力
        encoder_hidden_states_channels = encoder_hidden_states_channels or channels
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(encoder_hidden_states_channels, channels, bias=False)
        self.to_v = nn.Linear(encoder_hidden_states_channels, channels, bias=False)

        self.to_out = nn.Sequential(nn.Linear(channels, channels), nn.Dropout(0.0))

    def reshape_heads_to_batch_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = tensor.shape
        head_size = dim // self.num_heads
        tensor = tensor.reshape(batch_size, seq_len, self.num_heads, head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = tensor.shape
        batch_size = batch_size // self.num_heads
        tensor = tensor.reshape(batch_size, self.num_heads, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, dim * self.num_heads)
        return tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, channel, height, width = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # 计算注意力
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # 应用注意力权重
        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        hidden_states = self.to_out(hidden_states)

        # 重新reshape为原始形状
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)
        return hidden_states + residual

class UNetModel(nn.Module):
    """
    U-Net模型用于扩散过程中的噪声预测
    """
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [8, 16],
        dropout: float = 0.0,
        channel_mult: List[int] = [1, 2, 4, 4],
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = True,
        num_heads: int = 8,
        time_embed_dim: Optional[int] = None,
        class_embed_dim: Optional[int] = None,
        use_cross_attention: bool = True,
        use_self_attention: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.use_cross_attention = use_cross_attention
        self.use_self_attention = use_self_attention

        # 时间嵌入
        time_embed_dim = time_embed_dim or model_channels * 4
        self.time_embed = TimestepEmbedding(model_channels, time_embed_dim)

        # 类别嵌入 (条件)
        self.num_classes = num_classes
        if num_classes is not None:
            class_embed_dim = class_embed_dim or model_channels * 4
            self.class_embed = nn.Embedding(num_classes, class_embed_dim)
            self.class_proj = nn.Linear(class_embed_dim, time_embed_dim)
        else:
            self.class_embed = None
            self.class_proj = None

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
                    ResnetBlock(
                        in_channels=ch,
                        out_channels=mult * model_channels,
                        dropout=dropout,
                        time_embedding_dim=time_embed_dim,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels

                # 添加注意力层
                if ds in attention_resolutions and use_self_attention:
                    layers.append(
                        AttentionBlock(
                            channels=ch,
                            num_heads=num_heads,
                        )
                    )

                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                self.input_blocks.append(Downsample(ch, use_conv=True))
                input_block_chans.append(ch)
                ds *= 2

        # 中间块 (bottleneck)
        self.middle_block = nn.ModuleList([
            ResnetBlock(
                in_channels=ch,
                out_channels=ch,
                dropout=dropout,
                time_embedding_dim=time_embed_dim,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                channels=ch,
                num_heads=num_heads,
            ),
            ResnetBlock(
                in_channels=ch,
                out_channels=ch,
                dropout=dropout,
                time_embedding_dim=time_embed_dim,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        ])

        # 上采样路径
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResnetBlock(
                        in_channels=ch + input_block_chans.pop(),
                        out_channels=model_channels * mult,
                        dropout=dropout,
                        time_embedding_dim=time_embed_dim,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult

                # 添加注意力层
                if ds in attention_resolutions and use_self_attention:
                    layers.append(
                        AttentionBlock(
                            channels=ch,
                            num_heads=num_heads,
                        )
                    )

                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, use_conv=True))
                    ds //= 2

                self.output_blocks.append(nn.ModuleList(layers))

        # 输出层
        self.out_norm = nn.GroupNorm(32, model_channels)
        self.out_conv = nn.Conv2d(model_channels, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, in_channels, height, width] 噪声潜在表示
            timesteps: [batch_size] 时间步
            class_labels: [batch_size] 类别标签 (可选)
            encoder_hidden_states: 编码器隐藏状态 (用于交叉注意力)
        """
        # 时间嵌入
        time_emb = get_timestep_embedding(timesteps, self.model_channels)
        time_emb = self.time_embed(time_emb)

        # 类别嵌入
        if class_labels is not None and self.class_embed is not None:
            class_emb = self.class_embed(class_labels)
            class_emb = self.class_proj(class_emb)
            time_emb = time_emb + class_emb

        # 前向过程
        h = x
        hs = []

        # 下采样
        for module in self.input_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ResnetBlock):
                        h = layer(h, time_emb)
                    elif isinstance(layer, AttentionBlock):
                        h = layer(h, encoder_hidden_states)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)

        # Bottleneck
        for layer in self.middle_block:
            if isinstance(layer, ResnetBlock):
                h = layer(h, time_emb)
            elif isinstance(layer, AttentionBlock):
                h = layer(h, encoder_hidden_states)
            else:
                h = layer(h)

        # 上采样
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResnetBlock):
                    h = layer(h, time_emb)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h, encoder_hidden_states)
                else:
                    h = layer(h)

        # 输出
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)

        return h 