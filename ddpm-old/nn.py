"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn


# def conv_nd(dims, *args, **kwargs):
#     return nn.Conv2d(*args, **kwargs)

# def linear(*args, **kwargs):
#     return nn.Linear(*args, **kwargs)

# def avg_pool_nd(dims, *args, **kwargs):
#     return nn.AvgPool2d(*args, **kwargs)


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


# 将模块参数清零并返回。
def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels):
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    创建正弦时间步嵌入, tensorflow同款。这不是噪声 schedule, 而是时间步的嵌入方式。
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    :param timesteps: 由 N 个指数组成的一维张量，每个批元素一个指数。这些指数可以是分数。
    :param dim: 输出的维度。
    :param max_period: 控制嵌入的最小频率。
    :return: 一个 [N x dim] 形状的位置嵌入张量。
    """
    # 一半sin，一半cos，dim是输出的通道数，比如128
    half = dim // 2
    # 构造频率 freqs
    # freqs[i] = exp(-log(10000) * i / half) = 1 / (10000^(i/half))
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    # 时间步 t 乘以频率的广播计算, 得到一个矩阵
    args = timesteps[:, None].float() * freqs[None]
    # 现在我们得到了两个 [batch_size, half] 的矩阵，一个是 cos(args)，一个是 sin(args)
    # 把它们拼起来, 这就是最终的时间步嵌入向量！
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    # 如果 dim 是奇数（比如 129），那我们需要补一维 0，让它维度凑够 dim
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
