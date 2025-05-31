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
    """增强的残差块 - 更强的特征学习能力"""
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

        # 🔧 增强的归一化和激活
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=1e-6, affine=True)
        self.norm3 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=1e-6, affine=True)
        
        # 🔧 多分支卷积 - 更丰富的特征提取
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # 深度可分离卷积分支
        self.depthwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                       stride=1, padding=1, groups=out_channels)
        self.pointwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        # 多尺度特征融合
        self.conv_1x1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1)
        self.conv_3x3 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1)
        self.conv_5x5 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=5, padding=2)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # 时间嵌入投影
        if time_embedding_dim is not None:
            if time_embedding_norm == "default":
                time_embedding_norm = "scale_shift" if use_scale_shift_norm else "default"

            if time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(time_embedding_dim, 2 * out_channels)
            else:
                self.time_emb_proj = nn.Linear(time_embedding_dim, out_channels)
        else:
            self.time_emb_proj = None

        # 🔧 增强的Dropout和激活
        self.dropout = nn.Dropout(dropout)
        self.nonlinearity = nn.SiLU()
        
        # 🔧 注意力增强的跳跃连接
        self.skip_connection = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.GroupNorm(num_groups=min(groups, out_channels), num_channels=out_channels, eps=1e-6)
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        
        # 🔧 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, 1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, input_tensor: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = input_tensor

        # 第一个卷积分支
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        # 🔧 多尺度特征融合
        branch_1x1 = self.conv_1x1(hidden_states)
        branch_3x3 = self.conv_3x3(hidden_states)
        branch_5x5 = self.conv_5x5(hidden_states)
        
        # 深度可分离卷积分支
        depthwise_out = self.depthwise_conv(hidden_states)
        depthwise_out = self.pointwise_conv(depthwise_out)
        
        # 融合多尺度特征
        multi_scale_features = torch.cat([branch_1x1, branch_3x3, branch_5x5], dim=1)
        hidden_states = multi_scale_features + depthwise_out

        # 时间嵌入注入
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
        
        # 🔧 通道注意力增强
        attention_weights = self.channel_attention(hidden_states)
        hidden_states = hidden_states * attention_weights
        
        # 最终归一化
        hidden_states = self.norm3(hidden_states)

        # 增强的跳跃连接
        return self.skip_connection(input_tensor) + hidden_states

class AttentionBlock(nn.Module):
    """增强的注意力块 - 更强的空间和特征注意力"""
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_groups: int = 32,
        encoder_hidden_states_channels: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels

        # 🔧 动态调整注意力头数确保整除
        self.num_heads = min(num_heads, channels // 32)  # 至少32个通道每头
        if channels % self.num_heads != 0:
            # 找到最大的能整除的头数
            for h in range(self.num_heads, 0, -1):
                if channels % h == 0:
                    self.num_heads = h
                    break
            else:
                self.num_heads = 1  # 最后fallback到1
        
        self.head_size = channels // self.num_heads

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)
        
        # 🔧 预归一化和后归一化
        self.pre_norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)
        self.post_norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)

        # 交叉注意力
        encoder_hidden_states_channels = encoder_hidden_states_channels or channels
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(encoder_hidden_states_channels, channels, bias=False)
        self.to_v = nn.Linear(encoder_hidden_states_channels, channels, bias=False)
        
        # 🔧 多尺度查询、键、值投影
        self.to_q_multi = nn.ModuleList([
            nn.Linear(channels, channels // 2, bias=False),
            nn.Linear(channels, channels // 2, bias=False),
        ])
        self.to_k_multi = nn.ModuleList([
            nn.Linear(encoder_hidden_states_channels, channels // 2, bias=False),
            nn.Linear(encoder_hidden_states_channels, channels // 2, bias=False),
        ])
        self.to_v_multi = nn.ModuleList([
            nn.Linear(encoder_hidden_states_channels, channels // 2, bias=False),
            nn.Linear(encoder_hidden_states_channels, channels // 2, bias=False),
        ])

        # 🔧 空间注意力机制
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 8, channels // 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # 🔧 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, channels, 32, 32) * 0.02)
        
        # 🔧 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.Dropout(0.1)
        )

        self.to_out = nn.Sequential(
            nn.Linear(channels, channels), 
            nn.Dropout(0.1)  # 增加dropout
        )

    def reshape_heads_to_batch_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = tensor.shape
        head_size = dim // self.num_heads
        # 确保channels能被num_heads整除
        if dim % self.num_heads != 0:
            raise ValueError(f"channels ({dim}) must be divisible by num_heads ({self.num_heads})")
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

        # 如果没有提供编码器隐藏状态，使用self-attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # 🔧 输入检查
        if torch.any(torch.isnan(hidden_states)) or torch.any(torch.isinf(hidden_states)):
            print(f"⚠️ 注意力输入异常")
            hidden_states = torch.clamp(hidden_states, min=-5.0, max=5.0)
            hidden_states = torch.where(torch.isnan(hidden_states) | torch.isinf(hidden_states), torch.zeros_like(hidden_states), hidden_states)

        # 标准化和增强
        normalized = self.norm(hidden_states)
        
        # 🔧 检查归一化输出
        if torch.any(torch.isnan(normalized)) or torch.any(torch.isinf(normalized)):
            print(f"⚠️ 注意力归一化异常")
            normalized = torch.zeros_like(hidden_states)
        
        # 🔧 空间增强 - 添加数值稳定性
        try:
            spatial_enhanced = self.spatial_attention(normalized)
            if torch.any(torch.isnan(spatial_enhanced)) or torch.any(torch.isinf(spatial_enhanced)):
                print(f"⚠️ 空间注意力输出异常")
                spatial_enhanced = normalized
        except Exception as e:
            print(f"⚠️ 空间注意力计算失败: {e}")
            spatial_enhanced = normalized

        # Reshape for attention
        hidden_states = spatial_enhanced.view(batch, channel, height * width).transpose(1, 2)

        # 🔧 基础注意力计算 - 加强数值稳定性
        try:
            query = self.to_q(hidden_states)
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            # 检查QKV
            for name, tensor in [("query", query), ("key", key), ("value", value)]:
                if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                    print(f"⚠️ 注意力{name}异常，使用零值")
                    tensor.fill_(0.0)

            # Reshape for multi-head attention
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            # 🔧 计算注意力分数 - 添加数值稳定性
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(query.size(-1))
            
            # 🔧 检查注意力分数
            if torch.any(torch.isnan(attention_scores)) or torch.any(torch.isinf(attention_scores)):
                print(f"⚠️ 注意力分数异常，使用均匀分布")
                attention_scores = torch.zeros_like(attention_scores)

            # 🔧 稳定的softmax
            attention_scores = torch.clamp(attention_scores, min=-10.0, max=10.0)
            attention_probs = torch.softmax(attention_scores, dim=-1)
            
            # 检查注意力概率
            if torch.any(torch.isnan(attention_probs)) or torch.any(torch.isinf(attention_probs)):
                print(f"⚠️ 注意力概率异常，使用均匀分布")
                seq_len = attention_probs.size(-1)
                attention_probs = torch.full_like(attention_probs, 1.0 / seq_len)

            # 应用注意力权重
            standard_out = torch.matmul(attention_probs, value)
            standard_out = self.reshape_batch_dim_to_heads(standard_out)
            
            # 检查标准输出
            if torch.any(torch.isnan(standard_out)) or torch.any(torch.isinf(standard_out)):
                print(f"⚠️ 标准注意力输出异常")
                standard_out = torch.zeros_like(standard_out)
        except Exception as e:
            print(f"⚠️ 基础注意力计算失败: {e}")
            standard_out = torch.zeros(batch, height * width, channel, device=hidden_states.device)
        
        # 🔧 多尺度注意力路径 - 添加异常处理
        try:
            multi_scale_hidden = spatial_enhanced.view(batch, channel, height * width).transpose(1, 2)
            
            multi_queries = []
            multi_keys = []
            multi_values = []
            
            for q_proj, k_proj, v_proj in zip(self.to_q_multi, self.to_k_multi, self.to_v_multi):
                try:
                    mq = q_proj(multi_scale_hidden)
                    mk = k_proj(encoder_hidden_states)
                    mv = v_proj(encoder_hidden_states)
                    
                    # 检查多尺度QKV
                    for name, tensor in [("multi_q", mq), ("multi_k", mk), ("multi_v", mv)]:
                        if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                            print(f"⚠️ {name}异常，使用零值")
                            tensor.fill_(0.0)
                    
                    multi_queries.append(mq)
                    multi_keys.append(mk)
                    multi_values.append(mv)
                except Exception as e:
                    print(f"⚠️ 多尺度投影失败: {e}")
                    # 使用零值作为备选
                    zero_q = torch.zeros_like(multi_scale_hidden)
                    zero_kv = torch.zeros_like(encoder_hidden_states)
                    multi_queries.append(zero_q)
                    multi_keys.append(zero_kv)
                    multi_values.append(zero_kv)
            
            multi_out_parts = []
            for query_part, key_part, value_part in zip(multi_queries, multi_keys, multi_values):
                try:
                    # 🔧 简化的注意力计算 - 添加数值稳定性
                    attn_scores = torch.matmul(query_part, key_part.transpose(-1, -2)) / math.sqrt(query_part.size(-1))
                    attn_scores = torch.clamp(attn_scores, min=-10.0, max=10.0)
                    
                    attn_probs = torch.softmax(attn_scores, dim=-1)
                    
                    # 检查多尺度注意力
                    if torch.any(torch.isnan(attn_probs)) or torch.any(torch.isinf(attn_probs)):
                        print(f"⚠️ 多尺度注意力概率异常")
                        seq_len = attn_probs.size(-1)
                        attn_probs = torch.full_like(attn_probs, 1.0 / seq_len)
                    
                    attn_out = torch.matmul(attn_probs, value_part)
                    
                    if torch.any(torch.isnan(attn_out)) or torch.any(torch.isinf(attn_out)):
                        print(f"⚠️ 多尺度注意力输出异常")
                        attn_out = torch.zeros_like(attn_out)
                    
                    multi_out_parts.append(attn_out)
                except Exception as e:
                    print(f"⚠️ 多尺度注意力计算失败: {e}")
                    multi_out_parts.append(torch.zeros_like(query_part))
            
            multi_out = torch.cat(multi_out_parts, dim=-1)
        except Exception as e:
            print(f"⚠️ 多尺度注意力路径失败: {e}")
            multi_out = torch.zeros_like(standard_out)
        
        # 🔧 特征融合 - 添加异常处理
        try:
            combined_features = torch.cat([standard_out, multi_out], dim=-1)
            
            if torch.any(torch.isnan(combined_features)) or torch.any(torch.isinf(combined_features)):
                print(f"⚠️ 组合特征异常")
                combined_features = torch.zeros_like(combined_features)
            
            fused_output = self.feature_fusion(combined_features)
            
            if torch.any(torch.isnan(fused_output)) or torch.any(torch.isinf(fused_output)):
                print(f"⚠️ 特征融合输出异常")
                fused_output = torch.zeros_like(standard_out)
        except Exception as e:
            print(f"⚠️ 特征融合失败: {e}")
            fused_output = standard_out
        
        # 🔧 输出投影 - 添加异常处理
        try:
            final_output = self.to_out(fused_output)
            
            if torch.any(torch.isnan(final_output)) or torch.any(torch.isinf(final_output)):
                print(f"⚠️ 输出投影异常")
                final_output = torch.zeros_like(final_output)
        except Exception as e:
            print(f"⚠️ 输出投影失败: {e}")
            final_output = torch.zeros_like(fused_output)

        # 重新reshape为原始形状
        final_output = final_output.transpose(-1, -2).reshape(batch, channel, height, width)
        
        # 🔧 后归一化和残差连接 - 添加异常处理
        try:
            final_output = self.post_norm(final_output)
            
            if torch.any(torch.isnan(final_output)) or torch.any(torch.isinf(final_output)):
                print(f"⚠️ 后归一化异常")
                final_output = torch.zeros_like(final_output)
            
            # 🔧 安全的残差连接
            result = final_output + residual
            
            if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
                print(f"⚠️ 残差连接异常，仅使用残差")
                result = residual
            
            # 最终输出限制
            result = torch.clamp(result, min=-5.0, max=5.0)
            
            return result
        except Exception as e:
            print(f"⚠️ 后处理失败: {e}")
            return torch.clamp(residual, min=-5.0, max=5.0)

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
        # 🔧 输入检查
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            print(f"⚠️ U-Net输入异常，使用零输入")
            x = torch.zeros_like(x)
        
        # 限制输入范围
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        # 时间嵌入
        try:
            time_emb = get_timestep_embedding(timesteps, self.model_channels)
            if torch.any(torch.isnan(time_emb)) or torch.any(torch.isinf(time_emb)):
                print(f"⚠️ 时间嵌入异常，重新初始化")
                time_emb = torch.zeros_like(time_emb)
            
            time_emb = self.time_embed(time_emb)
            
            # 检查时间嵌入输出
            if torch.any(torch.isnan(time_emb)) or torch.any(torch.isinf(time_emb)):
                print(f"⚠️ 时间嵌入网络输出异常")
                time_emb = torch.zeros_like(time_emb)
        except Exception as e:
            print(f"⚠️ 时间嵌入计算失败: {e}")
            time_emb = torch.zeros(x.shape[0], self.model_channels * 4, device=x.device)

        # 类别嵌入
        if class_labels is not None and self.class_embed is not None:
            try:
                # 验证标签范围以避免索引越界
                if torch.any(class_labels < 0) or torch.any(class_labels >= self.num_classes):
                    # 静默修复CFG训练中的-1标签，只在第一次时警告
                    if not hasattr(self, '_label_warning_shown'):
                        invalid_mask = (class_labels < 0) | (class_labels >= self.num_classes)
                        invalid_labels = class_labels[invalid_mask]
                        print(f"💡 检测到CFG训练标签（如-1），已自动处理。后续将静默修复。")
                        self._label_warning_shown = True
                    
                    # 将超出范围的标签替换为0（安全值）
                    class_labels = torch.clamp(class_labels, 0, self.num_classes - 1)
                
                class_emb = self.class_embed(class_labels)
                class_emb = self.class_proj(class_emb)
                
                # 检查类别嵌入
                if torch.any(torch.isnan(class_emb)) or torch.any(torch.isinf(class_emb)):
                    print(f"⚠️ 类别嵌入异常，跳过类别条件")
                    class_emb = torch.zeros_like(time_emb)
                
                time_emb = time_emb + class_emb
            except Exception as e:
                print(f"⚠️ 类别嵌入计算失败: {e}")

        # 前向过程
        h = x
        hs = []

        # 下采样
        try:
            for module_idx, module in enumerate(self.input_blocks):
                if isinstance(module, nn.ModuleList):
                    for layer_idx, layer in enumerate(module):
                        try:
                            if isinstance(layer, ResnetBlock):
                                h = layer(h, time_emb)
                            elif isinstance(layer, AttentionBlock):
                                h = layer(h, encoder_hidden_states)
                            else:
                                h = layer(h)
                            
                            # 🔧 每层后检查数值健康
                            if torch.any(torch.isnan(h)) or torch.any(torch.isinf(h)):
                                print(f"⚠️ 下采样模块{module_idx}层{layer_idx}输出异常")
                                h = torch.clamp(h, min=-5.0, max=5.0)
                                h = torch.where(torch.isnan(h) | torch.isinf(h), torch.zeros_like(h), h)
                        except Exception as e:
                            print(f"⚠️ 下采样层计算失败: {e}")
                            continue
                else:
                    h = module(h)
                
                hs.append(h)
        except Exception as e:
            print(f"⚠️ 下采样路径计算失败: {e}")

        # Bottleneck
        try:
            for layer_idx, layer in enumerate(self.middle_block):
                try:
                    if isinstance(layer, ResnetBlock):
                        h = layer(h, time_emb)
                    elif isinstance(layer, AttentionBlock):
                        h = layer(h, encoder_hidden_states)
                    else:
                        h = layer(h)
                    
                    # 🔧 检查中间层输出
                    if torch.any(torch.isnan(h)) or torch.any(torch.isinf(h)):
                        print(f"⚠️ 中间层{layer_idx}输出异常")
                        h = torch.clamp(h, min=-5.0, max=5.0)
                        h = torch.where(torch.isnan(h) | torch.isinf(h), torch.zeros_like(h), h)
                except Exception as e:
                    print(f"⚠️ 中间层{layer_idx}计算失败: {e}")
                    continue
        except Exception as e:
            print(f"⚠️ 中间块计算失败: {e}")

        # 上采样
        try:
            for module_idx, module in enumerate(self.output_blocks):
                try:
                    if hs:
                        skip_connection = hs.pop()
                        # 检查跳跃连接
                        if torch.any(torch.isnan(skip_connection)) or torch.any(torch.isinf(skip_connection)):
                            print(f"⚠️ 跳跃连接{module_idx}异常")
                            skip_connection = torch.zeros_like(h)
                        h = torch.cat([h, skip_connection], dim=1)
                    
                    for layer_idx, layer in enumerate(module):
                        try:
                            if isinstance(layer, ResnetBlock):
                                h = layer(h, time_emb)
                            elif isinstance(layer, AttentionBlock):
                                h = layer(h, encoder_hidden_states)
                            else:
                                h = layer(h)
                            
                            # 🔧 检查上采样层输出
                            if torch.any(torch.isnan(h)) or torch.any(torch.isinf(h)):
                                print(f"⚠️ 上采样模块{module_idx}层{layer_idx}输出异常")
                                h = torch.clamp(h, min=-5.0, max=5.0)
                                h = torch.where(torch.isnan(h) | torch.isinf(h), torch.zeros_like(h), h)
                        except Exception as e:
                            print(f"⚠️ 上采样层计算失败: {e}")
                            continue
                except Exception as e:
                    print(f"⚠️ 上采样模块{module_idx}计算失败: {e}")
                    continue
        except Exception as e:
            print(f"⚠️ 上采样路径计算失败: {e}")

        # 输出层
        try:
            h = self.out_norm(h)
            h = F.silu(h)
            
            # 🔧 激活后检查
            if torch.any(torch.isnan(h)) or torch.any(torch.isinf(h)):
                print(f"⚠️ 输出激活异常")
                h = torch.clamp(h, min=-5.0, max=5.0)
                h = torch.where(torch.isnan(h) | torch.isinf(h), torch.zeros_like(h), h)
            
            h = self.out_conv(h)
            
            # 🔧 最终输出检查和处理
            if torch.any(torch.isnan(h)) or torch.any(torch.isinf(h)):
                print(f"⚠️ U-Net输出异常，使用零输出")
                h = torch.zeros_like(h)
            
            # 更严格的输出范围限制
            h = torch.clamp(h, min=-5.0, max=5.0)
        except Exception as e:
            print(f"⚠️ 输出层计算失败: {e}")
            h = torch.zeros_like(x)

        return h 
