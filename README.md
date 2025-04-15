## 代码复现-DDPM

## 环境配置

## 目录结构

```py
CodeReproduction-DDPM/
├── diffusion/                # 扩散过程逻辑（正向q/反向p）
│   ├── beta_schedule.py      # β 线性时间表
│   ├── gaussian_diffusion.py # q_sample, p_sample 等核心函数
│   └── losses.py             # 训练用 loss（如 MSE）
├── models/
│   ├── unet.py               # 简化 U-Net 网络（残差结构 + 下采样/上采样）
│   └── layers.py             # 残差块、下采样、上采样模块等
├── scripts/
│   └── train.py              # 训练入口
├── utils/
│   └── logger.py             # 日志和进度打印（可选）
├── configs/
│   └── default.yaml          # 模型和训练超参配置
└── main.py                   # 运行入口
```
