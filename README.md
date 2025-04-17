## 代码复现-DDPM

## 环境配置

## 目录结构

```py
Recode-DDPM/
├── ddpm/                     # 扩散模型核心
│   ├── noise_schedule.py       # β 线性时间表 及 相关计算
│   ├── diffusion.py            # q_sample, p_sample 等核心函数
│   └── utils.py                # 辅助函数
├── models/                   # 网络模型选择
│   ├── unet1.py                # U-Net + ResBlock
│   ├── unet2.py                # U-Net
│   ├── unet3.py                # NaiveConvNet
│   └── utils.py                # 辅助函数
├── scripts/                  # 训练采样入口
│   └── ts_unet1.py             # ts 表示 training and sampling
│   └── ts_unet2.py
│   └── ts_unet3.py
│   └── train.py                # 训练模型
│   └── sample.py               # 采样生成
├── documents/                # 每个文件的讲解笔记
│   └── *.md
├── public/                   # 公共资源
│   └── dataImg/                # 自定义的训练图像集
│   └── docsImg/                # markdown所需图像
└── main.py                   # 运行入口
```
