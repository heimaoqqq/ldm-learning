import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_timestep_embedding(timesteps, embedding_dim):
    """
    基于正弦位置编码的时间步嵌入，参考DDPM论文
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # 奇数维度补零
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class ResidualBlock(nn.Module):
    """
    残差块，与VQ-VAE保持一致的结构，但添加条件信息注入
    使用ReLU激活函数以与VQ-VAE中的ResidualBlock保持一致
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.relu = nn.ReLU(inplace=True)  # 与VQ-VAE保持一致，使用ReLU
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        
        # 跳跃连接
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
            
        # 时间嵌入投影层（用于条件信息注入）
        if time_emb_dim is not None:
            self.time_emb_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_emb_proj = None
    
    def forward(self, x, time_emb=None):
        identity = self.skip(x)
        
        # 第一个卷积层
        h = self.relu(self.gn1(self.conv1(x)))
        
        # 注入时间嵌入（条件信息）
        if time_emb is not None and self.time_emb_proj is not None:
            time_proj = self.time_emb_proj(time_emb)[:, :, None, None]
            h = h + time_proj
        
        # 第二个卷积层
        h = self.gn2(self.conv2(h))
        
        # 残差连接
        h += identity
        
        # 最终激活
        return self.relu(h)

class SelfAttention(nn.Module):
    """
    自注意力模块
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.out_proj = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # 归一化
        h = self.norm(x)
        
        # 重塑为序列格式
        h = h.view(batch_size, channels, height * width)
        
        # 计算QKV
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # 重塑为多头格式
        q = q.view(batch_size, self.num_heads, self.head_dim, height * width)
        k = k.view(batch_size, self.num_heads, self.head_dim, height * width)
        v = v.view(batch_size, self.num_heads, self.head_dim, height * width)
        
        # 计算注意力
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum('bhdk,bhdl->bhkl', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.einsum('bhkl,bhdl->bhdk', attn, v)
        out = out.contiguous().view(batch_size, channels, height * width)
        
        # 输出投影
        out = self.out_proj(out)
        out = out.view(batch_size, channels, height, width)
        
        # 残差连接
        return x + out

class UNet(nn.Module):
    """
    U-Net噪声预测器 ε_θ (论文中的噪声预测模型)
    
    与论文的一致性：
    - 采用U-Net架构，集成自注意力机制
    - 使用ResNet Blocks作为骨干（与VQ-VAE中的ResidualBlock一致）
    - 时间步嵌入：正弦位置编码 → MLP
    - 类别嵌入：nn.Embedding → MLP
    - 条件信息通过ResNet块集成（时间嵌入注入）
    - 自注意力在较低分辨率（4×4）添加，捕捉全局依赖
    
    输入：
    - z_t: [B, latent_dim, 16, 16] 加噪潜向量
    - t: 时间步（整数）
    - c: 类别标签（整数）
    
    输出：
    - ε_pred: [B, latent_dim, 16, 16] 预测的噪声
    """
    def __init__(self, 
                 latent_dim=256,
                 num_classes=31,
                 time_emb_dim=128,
                 class_emb_dim=64,
                 base_channels=128,
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
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 输入投影
        self.input_proj = nn.Conv2d(latent_dim, base_channels, 3, 1, 1)
        
        # 编码器部分
        self.down1 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim, groups)
        self.down_sample1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, 2, 1)  # 16->8
        
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim, groups)
        self.down_sample2 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, 2, 1)  # 8->4
        
        # 瓶颈层
        self.mid1 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, groups)
        self.mid_attn = SelfAttention(base_channels * 4)
        self.mid2 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, groups)
        
        # 解码器部分
        self.up_sample1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, 2, 1)  # 4->8
        self.up1 = ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim, groups)
        
        self.up_sample2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, 2, 1)  # 8->16
        self.up2 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim, groups)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.GroupNorm(groups, base_channels),
            nn.ReLU(inplace=True),  # 与ResidualBlock保持一致，使用ReLU
            nn.Conv2d(base_channels, latent_dim, 3, 1, 1)
        )
    
    def forward(self, z_t, timesteps, class_labels):
        """
        前向传播
        """
        # 计算嵌入
        time_emb = get_timestep_embedding(timesteps, self.time_emb_dim)
        time_emb = self.time_embed(time_emb)
        
        class_emb = self.class_embed(class_labels)
        
        # 合并条件信息
        cond_emb = time_emb + class_emb
        
        # 输入投影
        h = self.input_proj(z_t)  # [B, 128, 16, 16]
        
        # 下采样
        h = self.down1(h, cond_emb)  # [B, 256, 16, 16]
        h = self.down_sample1(h)     # [B, 256, 8, 8]
        
        h = self.down2(h, cond_emb)  # [B, 512, 8, 8]
        h = self.down_sample2(h)     # [B, 512, 4, 4]
        
        # 瓶颈层
        h = self.mid1(h, cond_emb)   # [B, 512, 4, 4]
        h = self.mid_attn(h)         # [B, 512, 4, 4]
        h = self.mid2(h, cond_emb)   # [B, 512, 4, 4]
        
        # 上采样
        h = self.up_sample1(h)       # [B, 512, 8, 8]
        h = self.up1(h, cond_emb)    # [B, 256, 8, 8]
        
        h = self.up_sample2(h)       # [B, 256, 16, 16]
        h = self.up2(h, cond_emb)    # [B, 128, 16, 16]
        
        # 输出投影
        return self.output_proj(h)   # [B, 256, 16, 16]

class DDPM:
    """
    去噪扩散概率模型
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # 线性噪声调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def to(self, device):
        """将所有张量移动到指定设备"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
        
    def q_sample(self, x_0, t, noise=None):
        """前向过程：q(z_t | z_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, z_t, t, class_labels, clip_denoised=True):
        """反向过程单步：p(z_{t-1} | z_t)"""
        pred_noise = model(z_t, t, class_labels)
        
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # 预测的 z_0
        pred_z_0 = (z_t - sqrt_one_minus_alpha_cumprod_t * pred_noise) / torch.sqrt(alpha_cumprod_t)
        
        if clip_denoised:
            pred_z_0 = torch.clamp(pred_z_0, -1., 1.)
        
        # 计算后验均值
        posterior_mean = (
            torch.sqrt(self.alphas_cumprod_prev[t]).view(-1, 1, 1, 1) * beta_t * pred_z_0 +
            torch.sqrt(alpha_t) * (1. - self.alphas_cumprod_prev[t]).view(-1, 1, 1, 1) * z_t
        ) / (1. - alpha_cumprod_t)
        
        if t[0] == 0:
            return posterior_mean
        else:
            posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(z_t)
            return posterior_mean + torch.sqrt(posterior_variance) * noise
    
    def p_sample_loop(self, model, shape, class_labels, device):
        """完整的反向采样过程"""
        z = torch.randn(shape, device=device)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            z = self.p_sample(model, z, t, class_labels)
            
        return z

class CLDM(nn.Module):
    """
    条件潜扩散模型 (Conditional Latent Diffusion Model)
    
    DiffGait论文实现：第二阶段模型
    
    与论文的完全一致性：
    1. 网络架构：
       - U-Net骨干 + 自注意力机制
       - ResNet Blocks（与VQ-VAE完全一致）
       - 条件信息集成（时间 + 类别嵌入）
    
    2. 扩散过程：
       - 标准DDPM前向过程：q(z_t | z_0)
       - 反向去噪过程：p(z_{t-1} | z_t)
       - 线性噪声调度：β_start=1e-4, β_end=0.02
    
    3. 损失函数：
       - MSE损失：||ε - ε_θ(z_t, t, c)||²_2
       - 与论文公式(16)完全一致
    
    4. 操作空间：
       - 在VQ-VAE潜空间操作 [B, 256, 16, 16]
       - 而非像素空间，提高效率
    
    5. 条件生成：
       - 根据类别标签c（用户ID）生成潜向量z
       - 支持31个用户（ID_1到ID_31）
    """
    def __init__(self, 
                 latent_dim=256,
                 num_classes=31,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 **unet_kwargs):
        super().__init__()
        
        self.unet = UNet(latent_dim=latent_dim, 
                        num_classes=num_classes,
                        **unet_kwargs)
        
        self.ddpm = DDPM(num_timesteps=num_timesteps,
                        beta_start=beta_start,
                        beta_end=beta_end)
        
        self.num_timesteps = num_timesteps
        
    def forward(self, z_0, class_labels):
        """训练前向传播"""
        batch_size = z_0.shape[0]
        device = z_0.device
        
        # 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # 添加噪声
        noise = torch.randn_like(z_0)
        z_t = self.ddpm.q_sample(z_0, t, noise)
        
        # 预测噪声
        pred_noise = self.unet(z_t, t, class_labels)
        
        # 计算损失
        loss = F.mse_loss(pred_noise, noise)
        
        return loss
    
    def sample(self, class_labels, device):
        """生成新的潜向量"""
        batch_size = len(class_labels)
        shape = (batch_size, self.unet.latent_dim, 16, 16)
        
        with torch.no_grad():
            return self.ddpm.p_sample_loop(self.unet, shape, class_labels, device)
    
    def to(self, device):
        """重写to方法，确保DDPM也移动到正确设备"""
        self.ddpm.to(device)
        return super().to(device)

if __name__ == "__main__":
    # 测试CLDM模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = CLDM(
        latent_dim=256,      # 与VQ-VAE一致
        num_classes=31,      # ID_1 到 ID_31
        num_timesteps=1000,  # 标准DDPM设置
        base_channels=128,   # 适合16GB GPU
    ).to(device)
    
    # 测试前向传播
    batch_size = 4
    z_0 = torch.randn(batch_size, 256, 16, 16).to(device)  # 模拟VQ-VAE输出
    class_labels = torch.randint(0, 31, (batch_size,)).to(device)  # 随机类别
    
    # 训练前向传播
    loss = model(z_0, class_labels)
    print(f"训练损失: {loss.item():.6f}")
    
    # 测试采样
    generated_z = model.sample(class_labels, device)
    print(f"生成的潜向量形状: {generated_z.shape}")
    print(f"生成的潜向量范围: [{generated_z.min().item():.3f}, {generated_z.max().item():.3f}]")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小约: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    print("CLDM模型测试成功！") 