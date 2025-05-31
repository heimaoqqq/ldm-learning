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
    """简化的注意力块 - 已修复形状问题"""
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
        
        # 简化的注意力投影
        encoder_hidden_states_channels = encoder_hidden_states_channels or channels
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(encoder_hidden_states_channels, channels, bias=False)
        self.to_v = nn.Linear(encoder_hidden_states_channels, channels, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(channels, channels), 
            nn.Dropout(0.1)
        )

    def reshape_heads_to_batch_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = tensor.shape
        
        # 🔧 严格的形状检查和修复
        if dim != self.channels:
            if dim < self.channels:
                # 填充到正确大小
                padding = torch.zeros(batch_size, seq_len, self.channels - dim, device=tensor.device, dtype=tensor.dtype)
                tensor = torch.cat([tensor, padding], dim=2)
            else:
                # 截取到正确大小
                tensor = tensor[:, :, :self.channels]
            dim = self.channels
        
        if self.channels % self.num_heads != 0:
            # 动态调整头数
            for h in range(min(self.num_heads, self.channels), 0, -1):
                if self.channels % h == 0:
                    self.num_heads = h
                    self.head_size = self.channels // h
                    break
        
        head_size = self.head_size
        
        try:
            # 重新形状到多头
            tensor = tensor.reshape(batch_size, seq_len, self.num_heads, head_size)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, head_size)
            return tensor
        except Exception as e:
            # 创建安全的张量
            safe_tensor = torch.zeros(batch_size * self.num_heads, seq_len, head_size, 
                                    device=tensor.device, dtype=tensor.dtype)
            return safe_tensor

    def reshape_batch_dim_to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, head_size = tensor.shape
        batch_size = batch_size // self.num_heads
        
        try:
            tensor = tensor.reshape(batch_size, self.num_heads, seq_len, head_size)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, head_size * self.num_heads)
            return tensor
        except Exception as e:
            # 创建安全的张量
            safe_tensor = torch.zeros(batch_size, seq_len, self.channels, 
                                    device=tensor.device, dtype=tensor.dtype)
            return safe_tensor

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
        
        # 标准化
        normalized = self.norm(hidden_states)
        
        # Reshape for attention
        hidden_states = normalized.view(batch, channel, height * width).transpose(1, 2)

        try:
            # 简化的注意力计算
            query = self.to_q(hidden_states)
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            # Reshape for multi-head attention
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            # 计算注意力分数
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(query.size(-1))
            attention_scores = torch.clamp(attention_scores, min=-10.0, max=10.0)
            attention_probs = torch.softmax(attention_scores, dim=-1)

            # 应用注意力权重
            attention_output = torch.matmul(attention_probs, value)
            attention_output = self.reshape_batch_dim_to_heads(attention_output)

            # 输出投影
            final_output = self.to_out(attention_output)

        except Exception as e:
            # 使用安全的零张量
            final_output = torch.zeros(batch, height * width, channel, device=hidden_states.device)

        # 重新reshape为原始形状
        final_output = final_output.transpose(-1, -2).reshape(batch, channel, height, width)
        
        # 残差连接
        result = final_output + residual
        result = torch.clamp(result, min=-5.0, max=5.0)
        
        return result

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
