"""
严格按照论文参数的CLDM实现
基于ControlNet和Stable Diffusion论文
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """线性beta调度 - 论文标准"""
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """余弦beta调度"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    时间步嵌入 - 使用正弦位置编码
    论文中提到使用正弦函数位置编码
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

class ResidualBlock(nn.Module):
    """
    ResNet残差块 - 论文标准
    每个分辨率级别使用2个残差块
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super().__init__()
        
        # 第一个卷积组
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # 时间嵌入投影
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        
        # 第二个卷积组
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 跳跃连接
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x, time_emb):
        identity = self.skip_connection(x)
        
        # 第一个卷积
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 时间嵌入
        time_proj = F.silu(self.time_emb_proj(time_emb))
        h = h + time_proj[:, :, None, None]
        
        # 第二个卷积
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + identity

class SelfAttentionBlock(nn.Module):
    """
    自注意力块 - 论文中在U-Net中间层使用
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        
        # 多头注意力
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 归一化
        residual = x
        x = self.norm(x)
        
        # Reshape for attention
        x = x.view(b, c, h * w).permute(0, 2, 1)  # (b, hw, c)
        
        # 计算注意力
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # 多头注意力
        scale = (self.channels // self.num_heads) ** -0.5
        q = q.view(b, h * w, self.num_heads, self.channels // self.num_heads).transpose(1, 2)
        k = k.view(b, h * w, self.num_heads, self.channels // self.num_heads).transpose(1, 2)
        v = v.view(b, h * w, self.num_heads, self.channels // self.num_heads).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, h * w, self.channels)
        out = self.to_out(out)
        
        # Reshape back
        out = out.permute(0, 2, 1).view(b, c, h, w)
        
        return out + residual

class UNetEncoder(nn.Module):
    """U-Net编码器部分"""
    def __init__(self, in_channels, model_channels, time_emb_dim, 
                 channel_mult=(1, 2, 4), num_res_blocks=2):
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks
        
        # 输入卷积
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # 下采样块
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = model_channels
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            # 残差块
            layers = []
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(ch, out_ch, time_emb_dim))
                ch = out_ch
            
            self.down_blocks.append(nn.ModuleList(layers))
            
            # 下采样
            if i < len(channel_mult) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            else:
                self.down_samples.append(nn.Identity())
    
    def forward(self, x, time_emb):
        h = self.input_conv(x)
        features = []
        
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            for block in blocks:
                h = block(h, time_emb)
            features.append(h)
            h = downsample(h)
        
        return h, features

class UNetDecoder(nn.Module):
    """U-Net解码器部分"""
    def __init__(self, out_channels, model_channels, time_emb_dim,
                 channel_mult=(4, 2, 1), num_res_blocks=2):
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks
        
        # 上采样块
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        ch = model_channels * channel_mult[0]
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            # 残差块
            layers = []
            for j in range(num_res_blocks + 1):  # +1 for skip connection
                if j == 0 and i > 0:
                    # 第一个块需要处理跳跃连接
                    in_ch = ch + model_channels * channel_mult[i-1]
                else:
                    in_ch = ch
                layers.append(ResidualBlock(in_ch, out_ch, time_emb_dim))
                ch = out_ch
            
            self.up_blocks.append(nn.ModuleList(layers))
            
            # 上采样
            if i < len(channel_mult) - 1:
                self.up_samples.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            else:
                self.up_samples.append(nn.Identity())
        
        # 输出卷积
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(self, x, time_emb, features):
        h = x
        
        for i, (blocks, upsample) in enumerate(zip(self.up_blocks, self.up_samples)):
            # 跳跃连接
            if i > 0:
                skip = features[-(i+1)]
                h = torch.cat([h, skip], dim=1)
            
            # 残差块
            for block in blocks:
                h = block(h, time_emb)
            
            # 上采样
            h = upsample(h)
        
        return self.output_conv(h)

class PaperStandardUNet(nn.Module):
    """
    严格按照论文的U-Net架构
    - U-Net + ResNet块 + 自注意力机制
    - 每个分辨率级别2个残差块
    - 在中间层使用自注意力
    """
    def __init__(self, 
                 in_channels=256,      # VQ-VAE潜在空间维度
                 out_channels=256,
                 model_channels=128,   # 论文建议的基础通道数
                 time_emb_dim=256,     # 时间嵌入维度
                 num_classes=31,
                 class_emb_dim=256,    # 类别嵌入维度
                 channel_mult=(1, 2, 4),  # 通道倍数
                 num_res_blocks=2,     # 每层残差块数量
                 attention_resolutions=(8,),  # 在8x8分辨率使用注意力
                 dropout=0.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.time_emb_dim = time_emb_dim
        
        # 时间嵌入网络
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 类别嵌入
        self.class_embed = nn.Embedding(num_classes, class_emb_dim)
        self.class_proj = nn.Sequential(
            nn.Linear(class_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 输入投影
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # 构建U-Net
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        # 下采样路径
        ch = model_channels
        input_block_chans = [ch]
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_channels, time_emb_dim, dropout)]
                ch = mult * model_channels
                
                # 在指定分辨率添加注意力
                if ds in attention_resolutions:
                    layers.append(SelfAttentionBlock(ch))
                
                self.downs.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.downs.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                input_block_chans.append(ch)
                ds *= 2
        
        # 中间块
        self.middle = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim, dropout),
            SelfAttentionBlock(ch),
            ResidualBlock(ch, ch, time_emb_dim, dropout),
        ])
        
        # 上采样路径
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResidualBlock(ch + ich, model_channels * mult, time_emb_dim, dropout)]
                ch = model_channels * mult
                
                # 在指定分辨率添加注意力
                if ds in attention_resolutions:
                    layers.append(SelfAttentionBlock(ch))
                
                if level and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                    ds //= 2
                
                self.ups.append(nn.ModuleList(layers))
        
        # 输出
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )
    
    def forward(self, x, timesteps, class_labels):
        """
        Args:
            x: 潜在空间噪声图像 [B, 256, 16, 16]
            timesteps: 时间步 [B]
            class_labels: 类别标签 [B]
        """
        # 时间嵌入
        t_emb = timestep_embedding(timesteps, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # 类别嵌入
        c_emb = self.class_embed(class_labels)
        c_emb = self.class_proj(c_emb)
        
        # 条件嵌入
        emb = t_emb + c_emb
        
        # 输入处理
        h = self.input_conv(x)
        hs = [h]
        
        # 下采样
        for module in self.downs:
            for layer in module:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)
            hs.append(h)
        
        # 中间层
        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                h = layer(h, emb)
            else:
                h = layer(h)
        
        # 上采样
        for module in self.ups:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)
        
        return self.out(h)

class PaperStandardDDPM:
    """严格按照论文的DDPM扩散过程"""
    def __init__(self, num_timesteps=1000, beta_schedule="linear"):
        self.num_timesteps = num_timesteps
        
        # Beta调度
        if beta_schedule == "linear":
            betas = linear_beta_schedule(num_timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.register_buffer('betas', betas)
        
        # 预计算常数
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # DDIM采样用
        self.register_buffer('posterior_variance', 
                           betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
    
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def ddim_sample(self, model, shape, class_labels, device, 
                   num_inference_steps=50, eta=0.0):
        """DDIM采样 - 论文中使用"""
        # 创建时间步
        timesteps = torch.linspace(self.num_timesteps-1, 0, num_inference_steps, dtype=torch.long)
        
        # 初始噪声
        x = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # 预测噪声
            pred_noise = model(x, t_batch, class_labels)
            
            # DDIM更新
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i+1]] if i < len(timesteps)-1 else torch.tensor(1.0)
            
            beta_t = 1 - alpha_t
            sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * beta_t)
            
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            direction = torch.sqrt(1 - alpha_prev - sigma_t**2) * pred_noise
            noise = torch.randn_like(x) if sigma_t > 0 else 0
            
            x = torch.sqrt(alpha_prev) * pred_x0 + direction + sigma_t * noise
        
        return x

class PaperStandardCLDM(nn.Module):
    """
    严格按照论文标准的CLDM
    - 使用标准U-Net架构
    - 标准DDPM扩散过程
    - 符合论文的所有参数设置
    """
    def __init__(self, 
                 latent_dim=256,
                 num_classes=31,
                 num_timesteps=1000,
                 beta_schedule="linear",
                 **unet_kwargs):
        super().__init__()
        
        # 扩散模型
        self.diffusion = PaperStandardDDPM(num_timesteps, beta_schedule)
        
        # U-Net模型
        self.unet = PaperStandardUNet(
            in_channels=latent_dim,
            out_channels=latent_dim,
            num_classes=num_classes,
            **unet_kwargs
        )
        
        # 将diffusion的buffer注册到模型
        for name, param in self.diffusion.__dict__.items():
            if isinstance(param, torch.Tensor):
                self.register_buffer(name, param)
    
    def forward(self, x_0, class_labels):
        """训练时的前向传播"""
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # 随机采样时间步
        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=device)
        
        # 添加噪声
        noise = torch.randn_like(x_0)
        x_t = self.diffusion.q_sample(x_0, t, noise)
        
        # 预测噪声
        pred_noise = self.unet(x_t, t, class_labels)
        
        # 计算损失
        loss = F.mse_loss(pred_noise, noise)
        
        return loss, pred_noise, noise
    
    def sample(self, class_labels, device, shape=None, num_inference_steps=50, eta=0.0):
        """生成样本"""
        if shape is None:
            batch_size = len(class_labels)
            shape = (batch_size, 256, 16, 16)  # 根据VQ-VAE潜在空间
        
        self.unet.eval()
        with torch.no_grad():
            samples = self.diffusion.ddim_sample(
                self.unet, shape, class_labels, device, num_inference_steps, eta
            )
        self.unet.train()
        
        return samples 