import torch
import os
from models.Unet1 import SimpleUNet as Unet1
from models.Unet2 import SimpleUNet as Unet2
from models.Unet3 import NaiveConvNet as Unet3
from ddpm.diffusion import GaussianDiffusion
from ddpm.utils import ModelMeanType, ModelVarType, LossType
from torchvision.utils import save_image
from tqdm import tqdm

def main(modelType, num_sample):
    # ----------- 1. 配置参数、参数文件 -----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "output\\training\\model_epoch_200.pt"
    image_size = 128
    save_dir = "./output/sampling"
    os.makedirs(save_dir, exist_ok=True)

    # ----------- 2. 实例化模型（要和训练时结构一模一样） -----------
    if modelType == "unet1":
        model = Unet1()
    elif modelType == "unet2":
        model = Unet2()
    else:
        model = Unet3()
    # model.load_state_dict(torch.load(model_path, map_location=device))
    torch.load(model_path, map_location=device, weights_only=True)  # 更保险
    model.to(device)
    model.eval()  # 切换到推理模式

    # ----------- 3. 初始化模型和 Diffusion -----------
    diffusion = GaussianDiffusion(
        model=model,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_LARGE,
        loss_type=LossType.MSE,
        timesteps=1000,
        device=device,
    )

    # ----------- 4. 一键采样、保存图片 -----------
    for i in range(num_sample):
        # 每次采样一个新图像时，使用独立的进度条
        with tqdm(total=1, desc=f"Sampling {i+1}", ncols=100) as pbar:
            sample = diffusion.sample(model, image_size, batch_size=1)
            # normalize 把 [-1, 1] 还原到 [0, 1]
            save_image(sample, f"{save_dir}/sample_{modelType}_{i}.png", normalize=True)
            pbar.update(1)  # 每个样本完成后更新进度条
