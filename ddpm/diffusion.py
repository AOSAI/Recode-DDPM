import torch
import torch.nn as nn
import numpy as np
from .noise_schedule import get_noise_schedule
from .utils import ModelMeanType, ModelVarType, LossType, extract

class GaussianDiffusion:
    def __init__(self, model, model_mean_type, model_var_type, loss_type, 
                 timesteps=1000, device="cuda"):
        # 保存参数
        self.model = model
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.timesteps = timesteps
        self.device = device

        # 获取所有噪声调度参数
        noise_schedule = get_noise_schedule(timesteps)
        # 将所有参数注册为 self.xxx，并转移到 device 上
        for k, v in noise_schedule.items():
            setattr(self, k, torch.from_numpy(v).to(device))


    def q_sample(self, x_start, t, noise=None):
        """从原始图像添加噪声, 模拟前向扩散过程的采样"""
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def q_posterior(self, x_start, x_t, t):
        # 用于计算真后验分布（可能用于loss）
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_sample(self, model, x_t, t):
        """
        给定当前时刻 t 的图像 x_t，通过模型预测 x_0，计算 q(x_{t-1} | x_t, x_0)，
        并从该分布中采样得到 x_{t-1}。

        Args:
            model: 训练好的神经网络模型，用来预测噪声 ε。
            x_t: 当前时间步 t 的图像张量。
            t: 当前时间步的张量 (B,) 或 (1,)。

        Returns:
            x_{t-1} 的一个采样结果。
        """
        # 模型预测的是噪声 ε_theta(x_t, t)
        eps_theta = model(x_t, t)

        # 估计 x_0（用 DDPM 公式从 x_t 和 eps_theta 推出）
        x_0_pred = self.predict_start_from_noise(x_t, t, eps_theta)

        # 计算后验分布的均值和方差
        mean, _, log_variance = self.q_posterior(x_0_pred, x_t, t)

        # 从 N(mean, var) 中采样一个 x_{t-1}
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))  # t=0 时不加噪声
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

    def p_sample_loop(self, model, shape, device):
        """
        从纯噪声开始反复调用 p_sample 采样出最终图像。

        Args:
            model: 已训练好的模型
            shape: 要生成图像的 shape（例如 [batch, C, H, W]）
            device: 生成图像所在的设备

        Returns:
            x_0 的采样图像
        """
        x_t = torch.randn(shape, device=device)

        with torch.no_grad():
            for i in reversed(range(self.timesteps)):
                t = torch.full((shape[0],), i, device=device, dtype=torch.long)
                x_t = self.p_sample(model, x_t, t)

        return x_t

    def sample(self, model, image_size, batch_size=16):
        # 外部调用入口，封装 p_sample_loop
        shape = (batch_size, 3, image_size, image_size)
        device = next(model.parameters()).device  # 👈 自动获取模型所在设备
        return self.p_sample_loop(model, shape, device)
    
    def p_losses(self, model, x_start, t, noise=None):
        # 损失函数，预测 noise 或 x_0 的误差
        if self.loss_type == LossType.MSE:
            self.loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError("Only MSE loss is supported for now.")
        
        t = t.to(x_start.device)
        if noise is None:
            noise = torch.randn_like(x_start)
        # 加噪音得到 x_t
        x_t = self.q_sample(x_start, t, noise)
        # 模型预测噪音
        predicted_noise = model(x_t, t)

        # 计算损失
        return self.loss_fn(predicted_noise, noise)