import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_timestep_embedding(timesteps, embedding_dim):
    """基于正弦位置编码的时间步嵌入"""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def cosine_beta_schedule(timesteps, s=0.008):
    """cosine调度"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class EnhancedResidualBlock(nn.Module):
    """增强的残差块"""
    def __init__(self, in_channels, out_channels, time_emb_dim=None, dropout=0.0, groups=32):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.gn1 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.gn2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
            
        if time_emb_dim is not None:
            self.time_emb_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels * 2)
            )
        else:
            self.time_emb_proj = None
    
    def forward(self, x, time_emb=None):
        identity = self.skip(x)
        
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.act(h)
        
        if time_emb is not None and self.time_emb_proj is not None:
            time_proj = self.time_emb_proj(time_emb)
            time_scale, time_shift = torch.chunk(time_proj, 2, dim=1)
            h = h * (1 + time_scale[:, :, None, None]) + time_shift[:, :, None, None]
        
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h += identity
        
        return self.act(h)

class SelfAttention(nn.Module):
    """自注意力模块"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.out_proj = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        h = self.norm(x)
        h = h.view(batch_size, channels, height * width)
        
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        scale = 1.0 / math.sqrt(channels)
        attn = torch.bmm(q.transpose(1, 2), k) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(v, attn.transpose(1, 2))
        out = self.out_proj(out)
        out = out.view(batch_size, channels, height, width)
        
        return x + out

class ImprovedUNet(nn.Module):
    """改进的U-Net架构"""
    def __init__(self, 
                 latent_dim=256,
                 num_classes=31,
                 time_emb_dim=512,
                 class_emb_dim=256,
                 base_channels=128,  # 降低基础通道数
                 dropout=0.1,
                 groups=32):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 类别嵌入
        self.class_embed = nn.Sequential(
            nn.Embedding(num_classes, class_emb_dim),
            nn.Linear(class_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        
        # 输入投影
        self.input_proj = nn.Conv2d(latent_dim, base_channels, 3, 1, 1)
        
        # 编码器 - 16x16 -> 8x8 -> 4x4
        self.down1 = nn.ModuleList([
            EnhancedResidualBlock(base_channels, base_channels * 2, time_emb_dim, dropout, groups),
            EnhancedResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim, dropout, groups),
        ])
        self.down_conv1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, 2, 1)
        
        self.down2 = nn.ModuleList([
            EnhancedResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim, dropout, groups),
            EnhancedResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, dropout, groups),
        ])
        self.down_conv2 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, 2, 1)
        
        # 瓶颈层 - 4x4
        self.mid = nn.ModuleList([
            EnhancedResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, dropout, groups),
            SelfAttention(base_channels * 4),
            EnhancedResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, dropout, groups),
        ])
        
        # 解码器 - 4x4 -> 8x8 -> 16x16
        self.up_conv1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.up1 = nn.ModuleList([
            EnhancedResidualBlock(base_channels * 4 * 2, base_channels * 2, time_emb_dim, dropout, groups),  # 跳跃连接
            EnhancedResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim, dropout, groups),
        ])
        
        self.up_conv2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, 2, 1)
        self.up2 = nn.ModuleList([
            EnhancedResidualBlock(base_channels * 2 * 2, base_channels, time_emb_dim, dropout, groups),  # 跳跃连接
            EnhancedResidualBlock(base_channels, base_channels, time_emb_dim, dropout, groups),
        ])
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(groups, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_dim, 3, 1, 1)
        )
    
    def forward(self, z_t, timesteps, class_labels):
        # 嵌入处理
        time_emb = get_timestep_embedding(timesteps, self.time_emb_dim)
        time_emb = self.time_embed(time_emb)
        
        class_emb = self.class_embed(class_labels)
        cond_emb = time_emb + class_emb
        
        # 输入投影
        h = self.input_proj(z_t)
        
        # 编码器
        # 16x16
        for block in self.down1:
            h = block(h, cond_emb)
        skip1 = h
        h = self.down_conv1(h)  # -> 8x8
        
        # 8x8
        for block in self.down2:
            h = block(h, cond_emb)
        skip2 = h
        h = self.down_conv2(h)  # -> 4x4
        
        # 瓶颈层 4x4
        for block in self.mid:
            if isinstance(block, EnhancedResidualBlock):
                h = block(h, cond_emb)
            else:  # SelfAttention
                h = block(h)
        
        # 解码器
        # 4x4 -> 8x8
        h = self.up_conv1(h)
        h = torch.cat([h, skip2], dim=1)
        for block in self.up1:
            h = block(h, cond_emb)
        
        # 8x8 -> 16x16
        h = self.up_conv2(h)
        h = torch.cat([h, skip1], dim=1)
        for block in self.up2:
            h = block(h, cond_emb)
        
        return self.output_proj(h)

class EnhancedDDPM:
    """增强的DDPM"""
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, noise_schedule="cosine"):
        self.num_timesteps = num_timesteps
        
        if noise_schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer = lambda name, val: setattr(self, name, val)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
    
    def to(self, device):
        for name, tensor in self.__dict__.items():
            if isinstance(tensor, torch.Tensor):
                setattr(self, name, tensor.to(device))
        return self
    
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def ddim_sample_step(self, model, z_t, t, t_prev, class_labels, eta=0.0):
        with torch.no_grad():
            pred_noise = model(z_t, t, class_labels)
            
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            alpha_t_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1) if t_prev >= 0 else torch.ones_like(alpha_t)
            
            pred_x0 = (z_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            direction = torch.sqrt(1 - alpha_t_prev - eta ** 2 * (1 - alpha_t_prev)) * pred_noise
            
            if eta > 0:
                noise = torch.randn_like(z_t)
                sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                direction += sigma * noise
            
            z_t_prev = torch.sqrt(alpha_t_prev) * pred_x0 + direction
            return z_t_prev
    
    def p_sample_loop(self, model, shape, class_labels, device, sampling_steps=None, eta=0.0):
        if sampling_steps is None:
            sampling_steps = self.num_timesteps
        
        timesteps = torch.linspace(self.num_timesteps - 1, 0, sampling_steps, dtype=torch.long, device=device)
        z_t = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1, device=device)
            
            z_t = self.ddim_sample_step(model, z_t, t_batch, t_prev, class_labels, eta)
        
        return z_t

class ImprovedCLDM(nn.Module):
    """改进的条件潜扩散模型"""
    def __init__(self, 
                 latent_dim=256,
                 num_classes=31,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 noise_schedule="cosine",
                 **unet_kwargs):
        super().__init__()
        
        self.unet = ImprovedUNet(
            latent_dim=latent_dim,
            num_classes=num_classes,
            **unet_kwargs
        )
        
        self.ddpm = EnhancedDDPM(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            noise_schedule=noise_schedule
        )
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
    
    def forward(self, z_0, class_labels):
        batch_size = z_0.shape[0]
        device = z_0.device
        
        t = torch.randint(0, self.ddpm.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(z_0)
        z_t = self.ddpm.q_sample(z_0, t, noise)
        
        pred_noise = self.unet(z_t, t, class_labels)
        loss = F.mse_loss(pred_noise, noise)
        
        return loss
    
    def sample(self, class_labels, device, sampling_steps=None, eta=0.0):
        batch_size = len(class_labels)
        shape = (batch_size, self.latent_dim, 16, 16)
        
        return self.ddpm.p_sample_loop(
            self.unet, shape, class_labels, device, sampling_steps, eta
        )
    
    def to(self, device):
        super().to(device)
        self.ddpm.to(device)
        return self 
