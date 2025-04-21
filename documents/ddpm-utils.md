## 1. 三个枚举函数 Enum

### 1.1 ModelMeanType 和 LossType

原始的 DDPM 论文中，做了对比实验，预测的内容有三种，但只有预测噪声的选项是最好的。这里只是列举了一下三种方式，实际在 diffusion 模块种使用的只有预测噪声。同时，预测噪声的损失函数，可以直接使用 MSE，十分的便利。

| 预测目标 | 说明                         | 表现                 |
| -------- | ---------------------------- | -------------------- |
| x\_{t-1} | 预测前一个时刻的图像         | 表现最差，梯度不稳定 |
| x_0      | 预测去噪后的原始图像         | 稍好，易梯度爆炸     |
| epsilon  | 预测前向过程中加入的高斯噪声 | ==最佳，最稳定！==   |

### 1.2 ModelVarType

用途在函数的说明中已经讲明，在采样时决定反向过程中协方差的计算方式。在 diffusion 模块的 p_sample 函数中使用。回顾一下目前的采样方式：

```py
# mean: 来自网络预测的后验均值。
# log_variance: 控制采样时加入的随机性。
x_{t-1} = mean + exp(0.5 * log_variance) * noise
```

这里的 small 和 large 控制的就是 log_variance，我们可以说 variance 大小 = 采样随机性大小：

- 当 variance 大，就是高斯分布比较“散”，采样波动很大，生成图像更有“多样性”，但噪声可能也更大。
- 当 variance 小，就是分布很窄、靠近均值，采样就更“贴近模型预测”，结果更稳定、锐利，但多样性可能会下降。

在 DDPM 论文中，默认使用的是 large 的方差，也就是扩散过程中计算出来的后验方差；而 small 的方差，是将对应的时间 t 和噪声强度 beta 对应组合起来，取对数得到的，它数值结果更小。

**训练早期**时，模型预测的 mean 不够准，如果这时 variance 很小，采样就容易发散（错误积累）；所以训练时保持 variance 大是安全的。

但**采样阶段**，如果模型已经学得不错，mean 已经是好的估计。variance 小，表示更信任模型本身预测，于是采样结果就更锐利、更稳定。这也是为什么 DDIM、PLMS 等加速采样算法都不使用随机采样（直接去掉噪声项 = variance=0）。

| 采样方式           | 效果                                               |
| ------------------ | -------------------------------------------------- |
| fixed_large        | 样本多样性高，但可能略模糊                         |
| fixed_small        | 图像更锐利、边界更清晰                             |
| variance=0（DDIM） | 采样几乎不随机，极快、极稳定，但容易失去 diversity |

## 2. extract 函数

在 diffusion 模块中，extract 是一个被频繁使用的辅助函数，用于根据时间步 t 从预先计算好的向量（beta 等）中提取对应批次的参数值，并广播成与输入图像相同的形状，以便后续计算。

比如，diffusion 模块的 q_sample 函数中第一次调用 extract 函数的部分：

```py
extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
```

sqrt_alphas_cumprod 是一个根据 beta 参数数组算出来的数据，同样也是一个一维数组。

t 是一个 [B] 形状的张量，表示当前批次中每张图的时间步。B 就代表 batch_size，同一批次训练中有多少张图像的意思。

而 x_start 是图像数据，拿使用的 cifar10 数据集举例，它的形状为 [B, 3, 32, 32]：统一批次 B 张图像，彩色图像所以是 3 通道，高 32，宽 32。

我们本质上是要用 sqrt_alphas_cumprod 去乘一个后方的 x_start，但是 x_start 是一个图像，和 sqrt_alphas_cumprod 这样的一维数组它们的矩阵形状是不一样的，做矩阵乘法会直接报错。所以要先将 sqrt_alphas_cumprod 的 batch_size 和 t 对齐，然后再将其广播成 x_start 的形状，这样就能和 x_start 相乘了。

### 2.1 注意事项及方法解释

```py
# res 是一个形状为 [B] 的一维向量，包含每个样本的时间相关值
res = arr.to(device=timesteps.device)[timesteps].float()
```

在这里我们指明了需要使用 device，也就是 GPU，所以 arr 必须是 pytorch 中的 tensor 张量，而不是 numpy 的数组。

```py
# 逐个添加维度，把 res 从 [B] 变成 [B, 1, 1, 1]
while len(res.shape) < len(broadcast_shape):
    res = res[..., None]
```

```py
# expand 函数会将 res 不复制内存地将 [B, 1, 1, 1]
# 扩展为 [B, C, H, W]，匹配图像的形状。
res.expand(broadcast_shape)
```
