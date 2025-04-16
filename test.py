import torch
from torchvision.utils import save_image

from models.Unet import SimpleUNet
from ddpm.diffusion import GaussianDiffusion
from ddpm.noise_schedule import get_noise_schedule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 构建模型
model = SimpleUNet(
    in_channels=3,
    out_channels=3,
    base_channels=64,
    num_res_blocks=2,
    num_downs=2
).to(device)

# 可选：加载预训练模型
# model.load_state_dict(torch.load("path/to/checkpoint.pth"))

# 2. 获取时间表
values = get_noise_schedule(1000)

# 3. 构建 GaussianDiffusion 实例
GD = GaussianDiffusion(
    betas=values.betas,
    alphas=values.alphas, # type: ignore
    alphas_cumprod=values.alphas_cumprod,
    alphas_cumprod_prev=values.alphas_cumprod_prev,
    sqrt_alphas_cumprod=values.sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod=values.sqrt_one_minus_alphas_cumprod,
)

# 4. 采样生成图像
with torch.no_grad():
    sample = GD.p_sample_loop(model, shape=(1, 3, 32, 32), device=device)

# 5. 保存图像
save_image(sample, "sample.png", normalize=True)