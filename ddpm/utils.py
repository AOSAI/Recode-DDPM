import torch as th
import numpy as np
from enum import Enum

class ModelMeanType(Enum):
    PREVIOUS_X = "prev_x"   # 预测 x_{t-1}
    START_X = "x0"          # 预测 x_0
    EPSILON = "eps"         # 预测噪声 epsilon

class ModelVarType(Enum):
    FIXED_SMALL = "fixed_small"
    FIXED_LARGE = "fixed_large"

class LossType(Enum):
    MSE = "mse"     # 原始的 MSE 损失
    def is_vb(self): return False

def extract(arr, timesteps, broadcast_shape):
    """
    从 1D numpy 数组中按时间步采样，并 broadcast 到目标形状。

    :param arr: 1D numpy 数组 (如 beta schedule)
    :param timesteps: shape 为 [batch_size] 的时间步张量。
    :param broadcast_shape: 用于广播的目标形状（如 [B, 1, 1, 1]）。
    :return: 已广播的张量, shape 为 broadcast_shape。
    """
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)