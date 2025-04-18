import torch
import numpy as np

def linear_beta_schedule(timesteps):
    """
    原始 DDPM 中使用的 1000步的 beta 线性 schedule。增加了 IDM 中的步数缩放机制。
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, timesteps)

def get_noise_schedule(timesteps):
    """
    构造 beta, alpha, alpha_bar 等时间相关参数
    这些参数只依赖于 beta、是与训练和推理阶段无关的一锤子买卖
    """
    betas = linear_beta_schedule(timesteps)
    assert betas.ndim == 1 and (betas > 0).all() and (betas <= 1).all()

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.0)

    # 后验方差
    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    # 后验对数方差。由于扩散链开始时的后验方差为 0，因此对数计算被剪切。
    posterior_log_variance_clipped = np.log(
        np.append(posterior_variance[1], posterior_variance[1:])
    )
    # 均值的两个系数
    posterior_mean_coef1 = (
        betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coef2 = (
        (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
    )

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "alphas_cumprod_next": alphas_cumprod_next,
        "sqrt_alphas_cumprod": np.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": np.sqrt(1.0 - alphas_cumprod),
        "log_one_minus_alphas_cumprod": np.log(1.0 - alphas_cumprod),
        "sqrt_recip_alphas_cumprod": np.sqrt(1.0 / alphas_cumprod),
        "sqrt_recipm1_alphas_cumprod": np.sqrt(1.0 / alphas_cumprod - 1),
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": posterior_log_variance_clipped,
        "posterior_mean_coef1": posterior_mean_coef1,
        "posterior_mean_coef2": posterior_mean_coef2,
    }