import math
import torch

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    参数 timesteps: LongTensor, 形状为 [batch_size], 表示每张图像的时间步t
    参数 dim: int, 时间嵌入的维度 (通常等于 base_channels * 4)
    返回 Tensor: [batch_size, dim] 的时间嵌入向量
    """
    half_dim = dim // 2
    # log-scale频率
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_scale).to(timesteps.device)
    emb = timesteps[:, None].float() * emb[None, :]  # [B, half_dim]
    # 拼接 sin 和 cos
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, dim]
    return emb