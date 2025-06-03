"""
LDM Diffusion Process - 基于CompVis/latent-diffusion
实现DDPM和DDIM扩散过程
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple
from tqdm import tqdm


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """线性beta调度"""
    return torch.linspace(start, end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """余弦beta调度，参考improved-diffusion"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class LDMDiffusion:
    """
    基于CompVis/latent-diffusion的扩散过程
    """
    def __init__(
        self,
        timesteps=1000,
        beta_schedule="cosine",
        beta_start=0.0001,
        beta_end=0.02,
        device="cuda"
    ):
        self.timesteps = timesteps
        self.device = device
        
        # 创建beta调度
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # 计算扩散参数
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 用于采样的参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 后验分布参数
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def extract(self, a, t, x_shape):
        """提取时间步对应的参数"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """
        前向过程：给x_0添加噪音
        q(x_t | x_0) = N(sqrt(α̅_t) * x_0, (1 - α̅_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        计算后验分布的均值和方差
        q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        """从噪音预测x_0"""
        sqrt_recip_alphas_cumprod = self.extract(
            1.0 / self.sqrt_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod = self.extract(
            self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod, t, x_t.shape
        )
        return sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * noise

    def p_mean_variance(self, model, x_t, t, class_labels=None, clip_denoised=True):
        """
        计算p(x_{t-1} | x_t)的均值和方差
        """
        # 模型预测噪音
        predicted_noise = model(x_t, t, class_labels)
        
        # 从噪音预测x_0
        x_start = self.predict_start_from_noise(x_t, t, predicted_noise)
        
        if clip_denoised:
            # 根据VAE潜变量的实际分布调整裁剪范围
            # Stable Diffusion VAE的潜变量通常在[-4, 4]范围内
            # 考虑到微调可能改变分布，使用稍微宽松的范围
            x_start = torch.clamp(x_start, -4.0, 4.0)
        
        # 计算后验均值和方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        
        return {
            "mean": model_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
            "pred_xstart": x_start,
            "pred_noise": predicted_noise,
        }

    def p_sample(self, model, x_t, t, class_labels=None, clip_denoised=True):
        """
        DDPM采样一步
        """
        out = self.p_mean_variance(model, x_t, t, class_labels, clip_denoised)
        
        noise = torch.randn_like(x_t)
        # 当t=0时不添加噪音
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def sample(
        self,
        model,
        shape,
        class_labels=None,
        num_inference_steps=None,
        eta=0.0,
        generator=None
    ):
        """
        DDPM/DDIM采样
        """
        if num_inference_steps is None:
            num_inference_steps = self.timesteps
            
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # 初始噪音
        if generator is not None:
            x_t = torch.randn(shape, generator=generator, device=device)
        else:
            x_t = torch.randn(shape, device=device)
        
        # 创建时间步序列
        if num_inference_steps < self.timesteps:
            # DDIM采样：跳过一些时间步
            step_size = self.timesteps // num_inference_steps
            # 修复：确保时间步序列正确
            timesteps = list(range(0, self.timesteps, step_size))[-num_inference_steps:]
            timesteps = torch.tensor(timesteps[::-1], device=device)  # 从大到小
        else:
            # DDPM采样：使用所有时间步
            timesteps = torch.arange(self.timesteps - 1, -1, -1, device=device)
        
        # 采样循环
        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            if eta == 0.0 and num_inference_steps < self.timesteps:
                # DDIM采样
                out = self.ddim_step(model, x_t, t_batch, class_labels, eta=eta)
                x_t = out["sample"]
            else:
                # DDPM采样
                out = self.p_sample(model, x_t, t_batch, class_labels)
                x_t = out["sample"]
        
        return x_t

    def ddim_step(self, model, x_t, t, class_labels=None, eta=0.0):
        """
        DDIM采样步骤
        """
        # 预测噪音
        predicted_noise = model(x_t, t, class_labels)
        
        # 获取当前时间步的alpha
        alpha_cumprod_t = self.extract(self.alphas_cumprod, t, x_t.shape)
        
        # 计算前一个时间步
        # 修复：正确计算前一个时间步
        prev_timestep = t - 1
        prev_timestep = torch.clamp(prev_timestep, 0)
        
        # 获取前一个时间步的alpha，处理t=0的情况
        alpha_cumprod_t_prev = torch.where(
            prev_timestep >= 0,
            self.alphas_cumprod.gather(0, prev_timestep.clamp(0)),
            torch.ones_like(alpha_cumprod_t.squeeze())
        ).view_as(alpha_cumprod_t)
        
        # 当t=0时，前一步应该是完全无噪音的
        alpha_cumprod_t_prev = torch.where(
            t == 0,
            torch.ones_like(alpha_cumprod_t),
            alpha_cumprod_t_prev
        )
        
        # 预测x_0（原始图像）
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        
        # 裁剪预测的x_0到合理范围
        pred_x0 = torch.clamp(pred_x0, -4.0, 4.0)
        
        # DDIM公式：计算方向向量
        if eta == 0.0:
            # 确定性DDIM
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * predicted_noise
        else:
            # 随机DDIM
            variance = eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * 
                (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - variance**2) * predicted_noise
        
        # 计算x_{t-1}
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt
        
        # 添加随机噪音（如果eta > 0）
        if eta > 0:
            noise = torch.randn_like(x_t)
            variance = eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * 
                (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )
            x_prev = x_prev + variance * noise
        
        return {"sample": x_prev, "pred_xstart": pred_x0}

    def training_losses(self, model, x_start, t, class_labels=None, noise=None):
        """
        计算训练损失
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 添加噪音
        x_t = self.q_sample(x_start, t, noise)
        
        # 模型预测
        predicted_noise = model(x_t, t, class_labels)
        
        # MSE损失
        loss = F.mse_loss(predicted_noise, noise, reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))  # 保持batch维度
        
        return {"loss": loss, "pred_noise": predicted_noise, "target_noise": noise}


class LDMScheduler:
    """
    LDM调度器，用于管理不同的扩散调度
    """
    def __init__(self, diffusion: LDMDiffusion):
        self.diffusion = diffusion
    
    def add_noise(self, original_samples, noise, timesteps):
        """添加噪音到原始样本"""
        return self.diffusion.q_sample(original_samples, timesteps, noise)
    
    def step(self, model_output, timestep, sample, class_labels=None, eta=0.0):
        """调度器步骤"""
        t = torch.full((sample.shape[0],), timestep, device=sample.device, dtype=torch.long)
        
        if eta == 0.0:
            # DDIM步骤
            return self.diffusion.ddim_step(
                lambda x, t, c: model_output, sample, t, class_labels, eta=eta
            )
        else:
            # DDPM步骤
            return self.diffusion.p_sample(
                lambda x, t, c: model_output, sample, t, class_labels
            )


def create_ldm_diffusion(
    timesteps=1000,
    beta_schedule="cosine",
    beta_start=0.0001,
    beta_end=0.02,
    device="cuda"
):
    """
    创建LDM扩散过程的工厂函数
    """
    return LDMDiffusion(
        timesteps=timesteps,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device
    )


if __name__ == "__main__":
    # 测试扩散过程
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建扩散过程
    diffusion = create_ldm_diffusion(device=device)
    
    # 创建假的模型输出函数
    def fake_model(x, t, class_labels=None):
        # 返回随机噪音作为预测
        return torch.randn_like(x)
    
    # 测试前向过程
    batch_size = 2
    x_start = torch.randn(batch_size, 4, 32, 32, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    
    # 添加噪音
    x_noisy = diffusion.q_sample(x_start, t)
    print(f"原始样本形状: {x_start.shape}")
    print(f"噪音样本形状: {x_noisy.shape}")
    
    # 测试损失计算
    losses = diffusion.training_losses(fake_model, x_start, t)
    print(f"训练损失: {losses['loss'].mean().item():.4f}")
    
    # 测试采样（少量步数）
    print("测试DDIM采样...")
    samples = diffusion.sample(fake_model, (1, 4, 32, 32), num_inference_steps=10, eta=0.0)
    print(f"采样结果形状: {samples.shape}")
    
    print("✅ LDM扩散过程测试成功!") 
