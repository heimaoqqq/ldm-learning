import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple, Union

class DDPMScheduler:
    """
    DDPM扩散调度器，实现前向加噪和反向去噪过程
    支持线性和余弦噪声调度
    """
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_denoised: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.clip_denoised = clip_denoised
        
        # 根据调度类型生成beta序列
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")
        
        # 计算alpha相关参数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # 用于采样的参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # 用于反向过程的参数
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
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        余弦噪声调度，来自 "Improved Denoising Diffusion Probabilistic Models"
        """
        def alpha_bar(time_step):
            return math.cos((time_step + s) / (1 + s) * math.pi / 2) ** 2
        
        betas = []
        for i in range(timesteps):
            t1 = i / timesteps
            t2 = (i + 1) / timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return torch.tensor(betas, dtype=torch.float32)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向过程：给原始样本添加噪声
        q(z_t | z_0) = N(z_t; sqrt(alpha_cumprod_t) * z_0, (1 - alpha_cumprod_t) * I)
        """
        # 确保参数在正确的设备上
        alphas_cumprod = self.sqrt_alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        
        sqrt_alpha_prod = alphas_cumprod[timesteps].flatten()
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        
        # 调整维度以匹配输入张量
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: Optional[float] = None,  # 🔧 添加eta参数以保持接口兼容
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[Tuple, "DDPMSchedulerOutput"]:
        """
        反向过程：从z_t预测z_{t-1}
        
        注意：DDPM调度器不使用eta参数，但接受它以保持接口兼容性
        """
        # 🔧 如果传入了eta参数，发出提示但不使用
        if eta is not None and eta != 0.0:
            print(f"💡 DDPM调度器不使用eta参数 (传入值: {eta})，将使用标准DDPM采样")
        
        t = timestep
        
        # 1. 计算系数
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t] if t > 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        
        # 2. 计算预测的原始样本 z_0
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        
        # 3. 裁剪预测样本
        if self.clip_denoised:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # 4. 计算后验均值和方差
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * model_output
        pred_prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        
        # 5. 添加噪声 (除了最后一步)
        if t > 0:
            device = model_output.device
            variance_noise = torch.randn(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )
            # 使用简化的方差计算
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
            pred_prev_sample = pred_prev_sample + variance ** (0.5) * variance_noise
        
        if not return_dict:
            return (pred_prev_sample,)
        
        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
    
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        设置推理时的时间步
        """
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) > num_train_timesteps ({self.num_train_timesteps})"
            )
        
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()
        timesteps = list(reversed(timesteps.astype(np.int64)))
        self.timesteps = torch.from_numpy(np.array(timesteps))
        
        if device is not None:
            self.timesteps = self.timesteps.to(device)
    
    def __len__(self):
        return self.num_train_timesteps


class DDPMSchedulerOutput:
    """
    DDPM调度器的输出类
    """
    def __init__(self, prev_sample: torch.Tensor, pred_original_sample: Optional[torch.Tensor] = None):
        self.prev_sample = prev_sample
        self.pred_original_sample = pred_original_sample


class DDIMScheduler(DDPMScheduler):
    """
    DDIM调度器，用于快速采样
    支持标准DDIM (eta=0) 和改进的自适应eta
    """
    def __init__(self, eta: float = 0.0, adaptive_eta: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta
        self.adaptive_eta = adaptive_eta  # 🆕 是否使用自适应eta
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: Optional[float] = None,
        use_clipped_model_output: bool = False,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ):
        """
        DDIM采样步骤
        支持标准eta和改进的自适应eta
        """
        if eta is None:
            eta = self.eta
        
        # 当前和前一时间步的索引
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        
        # 1. 获取alpha值
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        
        beta_prod_t = 1 - alpha_prod_t
        
        # 2. 计算预测的原始样本
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        
        # 3. 裁剪预测样本
        if self.clip_denoised:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # 4. 🆕 计算方差 - 支持自适应eta
        if self.adaptive_eta and eta > 0:
            # 使用论文中的改进公式: σ_τᵢ(η) = η√((1-α_τᵢ₋₁)/(1-α_τᵢ))√(1-α_τᵢ/α_τᵢ₋₁)
            variance = self._get_adaptive_variance(timestep, prev_timestep, eta)
        else:
            # 标准DDIM方差
            variance = self._get_variance(timestep, prev_timestep)
        
        # 5. 计算方向指向当前样本的"方向"
        # 确保开方参数非负
        term_for_sqrt = (1 - alpha_prod_t_prev - eta * variance).clamp(min=0.0)
        pred_sample_direction = term_for_sqrt.sqrt() * model_output
        
        # 6. 计算前一样本
        pred_prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        
        # 7. 添加噪声
        if eta > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=generator, device=device, dtype=model_output.dtype)
            # 确保开方参数非负
            variance_term_for_noise = (eta * variance).clamp(min=0.0)
            pred_prev_sample = pred_prev_sample + variance_term_for_noise.sqrt() * noise
        
        if not return_dict:
            return (pred_prev_sample,)
        
        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
    
    def _get_variance(self, timestep, prev_timestep):
        """标准DDIM方差计算"""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance
    
    def _get_adaptive_variance(self, timestep, prev_timestep, eta):
        """
        🆕 改进的自适应方差计算
        基于论文公式: σ_τᵢ(η) = η√((1-α_τᵢ₋₁)/(1-α_τᵢ))√(1-α_τᵢ/α_τᵢ₋₁)
        优化为: σ_τᵢ(η) = η√((1-α_τᵢ₋₁)(α_τᵢ₋₁-α_τᵢ)/(α_τᵢ₋₁(1-α_τᵢ)))
        """
        alpha_prod_t = self.alphas_cumprod[timestep]           # α_τᵢ
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)  # α_τᵢ₋₁
        
        # 使用优化的单个平方根公式
        # η√((1-α_τᵢ₋₁)(α_τᵢ₋₁-α_τᵢ)/(α_τᵢ₋₁(1-α_τᵢ)))
        numerator = (1 - alpha_prod_t_prev) * (alpha_prod_t_prev - alpha_prod_t)
        denominator = alpha_prod_t_prev * (1 - alpha_prod_t)
        
        adaptive_variance = eta * torch.sqrt(numerator / denominator)
        
        return adaptive_variance 
