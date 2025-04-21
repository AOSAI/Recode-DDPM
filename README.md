## 1. 代码复现-DDPM

该代码参考了 [improved-diffusion](https://github.com/openai/improved-diffusion) 的一些设定，重新解构出了 DDPM。欢迎各位道友用于学习，改造，和实验。

## 2. 环境配置

- system: win11
- conda 虚拟环境: python=3.12.0
- pytorch 版本: 2.4.1
- 其它相关依赖：查看 requirements.txt, 或代码中查看导入包的缺失

## 3. 目录结构

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
├── documents/                # 每个文件的讲解笔记《小白指南》
│   └── *.md
├── public/                   # 公共资源
│   └── dataImg/                # 自定义的训练图像集
│   └── docsImg/                # markdown所需图像
├── main.py                   # 运行入口

├── jupyter/                  # Jupyter Notebook
    └── ddpm-1.ipynb            # B站大佬的 DDPM 简单入门实验
```

## 4. 实验数据集参考

### 4.1 torchvision 数据集

CIFAR10 和 CIFAR100 是处理过的图像集，但是 CelebA 的原始尺寸应该在 178x218，最好做一个中心裁剪，然后 Resize 一下。

| 分辨率  | 数据集名称 | 图像数量 | 类别数 | 常用用途           |
| ------- | ---------- | -------- | ------ | ------------------ |
| 32×32   | CIFAR10    | 60,000   | 10     | 基础图像生成、分类 |
| 32×32   | CIFAR100   | 60,000   | 100    | 多类分类与生成     |
| 64、128 | CelebA     | 200,000  | -      | 人脸生成           |

### 4.2 scikit-learn 数据集

这些都是结构化点状图数据，非常适合小型 DDPM 训练，尤其是 MLPDiffusion！（MLP 是 Multilayer Perceptron，多层感知机的缩写。代表的就是一个输入层、若干个隐藏层、一个输出层的基础神经网络模型架构）。使用方式请参考 jupyter 文件夹。

```py
from sklearn.datasets import make_s_curve, make_moons, make_circles, make_blobs
```

| 名称         | 样式         | 加载方式                              |
| ------------ | ------------ | ------------------------------------- |
| make_s_curve | S 型三维结构 | make_s_curve(10000, noise=0.1)        |
| make_moons   | 两个弯月     | make_moons(n_samples=1000)            |
| make_circles | 同心圆       | make_circles(n_samples=1000)          |
| make_blobs   | 多中心点聚类 | make_blobs(n_samples=1000, centers=4) |

### 4.3 自定义数据集

有很多高质量图像的数据集都是需要手动去下载的，比如：

- FFHQ：高质量人脸，来源于 NVIDIA，常见于 StyleGAN 训练
- CelebA-HQ：高质量 CelebA
- ImageNet：超多分类版彩色图像

也可以使用自己收集的图像数据集，但是想要看到效果，至少有几百张图像做训练才有用。一张图像跑 DDPM 是没有效果的。
