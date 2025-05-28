import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += identity
        return self.relu(out)

class PaperEncoder(nn.Module):
    """论文风格的编码器，使用7×7初始卷积和步长2的3×3下采样卷积"""
    def __init__(self, in_channels, latent_dim, groups=32):
        super().__init__()
        # 使用7×7卷积替代4×4卷积作为stem层
        self.stem = nn.Conv2d(in_channels, 32, 7, 2, 3)  # 128x128
        self.gn1 = nn.GroupNorm(groups, 32)
        self.relu = nn.ReLU(inplace=True)
        
        self.res1 = ResidualBlock(32, 64, groups)
        # 使用步长2的3×3卷积代替4×4卷积
        self.down1 = nn.Conv2d(64, 64, 3, 2, 1)  # 64x64
        
        self.res2 = ResidualBlock(64, 128, groups)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)  # 32x32
        
        self.res3 = ResidualBlock(128, 256, groups)
        self.down3 = nn.Conv2d(256, 256, 3, 2, 1)  # 16x16
        
        self.res4 = ResidualBlock(256, latent_dim, groups)
    
    def forward(self, x):
        x = self.relu(self.gn1(self.stem(x)))
        x = self.res1(x)
        x = self.down1(x)
        x = self.res2(x)
        x = self.down2(x)
        x = self.res3(x)
        x = self.down3(x)
        x = self.res4(x)
        return x

class PaperDecoder(nn.Module):
    """论文风格的解码器，使用双线性上采样+卷积替代转置卷积"""
    def __init__(self, latent_dim, out_channels, groups=32):
        super().__init__()
        self.res1 = ResidualBlock(latent_dim, 256, groups)
        
        # 使用双线性上采样+卷积替代转置卷积
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.gn1 = nn.GroupNorm(groups, 256)
        
        self.res2 = ResidualBlock(256, 128, groups)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.gn2 = nn.GroupNorm(groups, 128)
        
        self.res3 = ResidualBlock(128, 64, groups)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.gn3 = nn.GroupNorm(groups, 64)
        
        self.res4 = ResidualBlock(64, 32, groups)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(32, out_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.res1(x)
        
        x = self.up1(x)
        x = self.relu(self.gn1(self.conv1(x)))
        
        x = self.res2(x)
        x = self.up2(x)
        x = self.relu(self.gn2(self.conv2(x)))
        
        x = self.res3(x)
        x = self.up3(x)
        x = self.relu(self.gn3(self.conv3(x)))
        
        x = self.res4(x)
        x = self.up4(x)
        x = self.conv4(x)
        
        return torch.tanh(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, decay=0.99):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.decay = decay  # EMA 衰减系数，论文参考值为 0.99
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
        # 注册 EMA 相关缓冲区
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embedding_avg', self.embedding.weight.data.clone())
        # 新增：用于跟踪整个epoch的累积码本使用情况
        self.register_buffer('cumulative_usage_vector', torch.zeros(num_embeddings))
        self._is_first_batch_in_epoch = True # 辅助判断是否是epoch的第一个batch
        
    def reset_cumulative_usage(self):
        """在每个epoch开始时调用，重置累积使用计数器"""
        self.cumulative_usage_vector.fill_(0)
        self._is_first_batch_in_epoch = True

    def forward(self, z):
        # 在epoch的第一个batch（非严格意义，但用于近似重置）重置累积使用
        # 更稳健的做法是在训练循环的epoch开始处调用reset_cumulative_usage
        if self.training and self._is_first_batch_in_epoch:
             # self.reset_cumulative_usage() # 实际重置应由外部训练循环控制
             self._is_first_batch_in_epoch = False

        z_flattened = z.permute(0,2,3,1).contiguous().view(-1, self.embedding_dim)
        dist = (z_flattened.pow(2).sum(1, keepdim=True) -
                2 * torch.matmul(z_flattened, self.embedding.weight.t()) +
                self.embedding.weight.pow(2).sum(1))
        encoding_indices = torch.argmin(dist, dim=1)
        
        batch_usage_percent = 0.0
        cumulative_epoch_usage_percent = 0.0

        if self.training:
            with torch.no_grad():
                # 当前批次的码本使用情况
                batch_active_indices = encoding_indices.unique()
                batch_usage_percent = batch_active_indices.numel() / self.num_embeddings
                
                # 更新整个epoch的累积码本使用情况
                self.cumulative_usage_vector.scatter_add_(0, batch_active_indices, torch.ones_like(batch_active_indices, dtype=self.cumulative_usage_vector.dtype))
                cumulative_epoch_usage_percent = (self.cumulative_usage_vector > 0).float().sum() / self.num_embeddings
        
        # 获取量化后的嵌入向量
        quantized_embeddings = self.embedding(encoding_indices)  # [N, D]
        # quantized = quantized_embeddings.view_as(z) # view_as(z) 可能在 z 的形状与预期不符时出错
        # 改为使用 z.shape 来确保形状一致
        quantized = quantized_embeddings.view(z.shape)


        # 计算 Codebook loss: ||sg(E(x)) - z_q||^2_2
        codebook_loss = F.mse_loss(quantized_embeddings, z_flattened.detach())
        
        # 计算 Commitment loss: beta * ||E(x) - sg(z_q)||^2_2
        commitment_loss = self.beta * F.mse_loss(z_flattened, quantized_embeddings.detach())
        
        # 使用 EMA 更新码本 (仅在训练模式下)
        if self.training:
            # 转换为 one-hot 编码
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            
            # EMA 更新簇大小
            self.cluster_size.data = self.decay * self.cluster_size + (1 - self.decay) * encodings.sum(0)
        
            # 避免除零问题
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            
            # 更新嵌入向量的移动平均
            dw = torch.matmul(encodings.t(), z_flattened)
            self.embedding_avg.data = self.decay * self.embedding_avg + (1 - self.decay) * dw
            
            # 更新码本权重 (嵌入向量)
            self.embedding.weight.data = self.embedding_avg / cluster_size.unsqueeze(1)
        
        # 直通估计器 (Straight-Through Estimator)
        quantized_straight_through = z + (quantized - z).detach()
        # 返回新增的使用率指标
        return quantized_straight_through, commitment_loss, codebook_loss, batch_usage_percent, cumulative_epoch_usage_percent

# 新增：Patch-based Discriminator（来自论文描述）
class PatchDiscriminator(nn.Module):
    """
    基于PatchGAN的判别器，使用卷积网络逐块判断图像真假
    70×70感受野 PatchGAN
    """
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        
        # 初始层 (Conv1)
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False)  # 禁用原地操作
        ]
        
        # 第二层卷积 (Conv2)：64 -> 128
        sequence += [
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, ndf * 2),
            nn.LeakyReLU(0.2, inplace=False)  # 禁用原地操作
        ]
        
        # 第三层卷积 (Conv3)：128 -> 256
        sequence += [
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, ndf * 4),
            nn.LeakyReLU(0.2, inplace=False)  # 禁用原地操作
        ]
        
        # 第四层卷积 (Conv4)：256 -> 512
        sequence += [
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, ndf * 8),
            nn.LeakyReLU(0.2, inplace=False)  # 禁用原地操作
        ]
        
        # 输出层：512 -> 1，输出真假概率图
        sequence += [nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*sequence)
        
    def forward(self, x):
        """
        返回每个patch的判别结果
        """
        return self.model(x)

class AdvVQVAE(nn.Module):
    """
    基于论文描述的对抗训练VQ-VAE，添加了patch-based discriminator
    """
    def __init__(self, in_channels=3, latent_dim=256, num_embeddings=512, beta=0.25, decay=0.99, groups=32, 
                 disc_ndf=64):
        super().__init__()
        self.encoder = PaperEncoder(in_channels, latent_dim, groups)
        self.vq = VectorQuantizer(num_embeddings, latent_dim, beta, decay)
        self.decoder = PaperDecoder(latent_dim, in_channels, groups)
        self.discriminator = PatchDiscriminator(in_channels, disc_ndf)
    
    def encode(self, x):
        """编码函数，返回量化后的潜在表示和使用率（如果需要）"""
        z = self.encoder(x)
        # 修改vq的调用以接收新的返回值
        z_q, _, _, batch_usage, epoch_usage = self.vq(z) 
        return z_q # 通常encode只关心z_q，使用率在forward中处理
    
    def decode(self, z_q):
        """解码函数，从潜在表示重建图像"""
        return self.decoder(z_q)
    
    def forward(self, x):
        z = self.encoder(x)
        # 修改vq的调用以接收新的返回值
        z_q_straight_through, commitment_loss, codebook_loss, batch_usage, cumulative_usage = self.vq(z)
        x_recon = self.decoder(z_q_straight_through)
        # 返回新增的使用率指标
        return x_recon, commitment_loss, codebook_loss, batch_usage, cumulative_usage
    
    def calculate_losses(self, x, rec_weight=1.0, perceptual_weight=0.1):
        """
        计算完整的损失函数，包括重建损失、VQ损失和对抗损失
        根据论文公式：L_Stage1 = min_{E,D} max_D (L_rec(x, D(E(x))) - L_adv(D(E(x))) + log(D(x)) + L_reg(x; E, D))
        这个方法在训练脚本中被覆盖了，主要参考训练脚本中的损失计算逻辑
        """
        # 获取重建结果和VQ损失
        # 修改forward的调用以接收新的返回值
        x_recon, commitment_loss, codebook_loss, _, _ = self.forward(x) # _ placeholder for usage metrics if not used here
        
        # 重建损失（MSE）
        rec_loss = F.mse_loss(x_recon, x)
        
        # 对抗损失计算
        # 真实图像的判别结果
        real_preds = self.discriminator(x)
        # 重建图像的判别结果
        fake_preds = self.discriminator(x_recon)
        
        # 判别器损失 (最大化这部分)
        disc_loss = -torch.mean(torch.log(torch.sigmoid(real_preds) + 1e-8) + 
                              torch.log(1 - torch.sigmoid(fake_preds) + 1e-8))
        
        # 生成器对抗损失 (最小化这部分)
        gen_loss = -torch.mean(torch.log(torch.sigmoid(fake_preds) + 1e-8))
        
        # 总损失
        # LStage1 = min_{E,D} max_D (L_rec - L_adv + log(D(x)) + L_reg)
        # 对于生成器(Encoder + Decoder)，目标是最小化：rec_loss - gen_loss + vq_loss
        # 对于判别器，目标是最小化：-disc_loss
        
        # 生成器总损失
        gen_total_loss = rec_weight * rec_loss + (commitment_loss + codebook_loss) - gen_loss # 示例性修改，实际看训练脚本
        
        return {
            'gen_total_loss': gen_total_loss,
            'disc_loss': disc_loss,
            'rec_loss': rec_loss,
            'vq_commitment_loss': commitment_loss,
            'vq_codebook_loss': codebook_loss,
            'gen_loss': gen_loss,
            'batch_usage': batch_usage,
            'cumulative_usage': cumulative_usage
        }
    
    def train_step(self, x, gen_optimizer, disc_optimizer, train_disc=True, train_gen=True):
        """
        执行一步训练，包括判别器和生成器的更新
        """
        losses = self.calculate_losses(x)
        
        # 训练判别器
        if train_disc:
            disc_optimizer.zero_grad()
            losses['disc_loss'].backward(retain_graph=True)
            disc_optimizer.step()
        
        # 训练生成器(Encoder + Decoder)
        if train_gen:
            gen_optimizer.zero_grad()
            losses['gen_total_loss'].backward()
            gen_optimizer.step()
        
        return losses

if __name__ == "__main__":
    # 测试代码
    model = AdvVQVAE(in_channels=3, latent_dim=256, num_embeddings=512, decay=0.99)
    x = torch.randn(2, 3, 256, 256)
    
    # 创建优化器
    gen_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.vq.parameters())
    gen_optimizer = optim.Adam(gen_params, lr=1e-4)
    disc_optimizer = optim.Adam(model.discriminator.parameters(), lr=1e-4)
    
    # 测试前向传播
    recon, commit_loss, codebook_loss, batch_usage, cumulative_usage = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {recon.shape}")
    print(f"VQ commitment loss: {commit_loss.item()}")
    print(f"VQ codebook loss: {codebook_loss.item()}")
    
    # 测试损失计算
    losses = model.calculate_losses(x)
    print(f"Generator total loss: {losses['gen_total_loss'].item()}")
    print(f"Discriminator loss: {losses['disc_loss'].item()}")
    
    # 测试训练步骤
    step_losses = model.train_step(x, gen_optimizer, disc_optimizer)
    print(f"After one training step - Rec loss: {step_losses['rec_loss'].item()}") 
