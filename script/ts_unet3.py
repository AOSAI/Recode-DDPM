import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.Unet1 import SimpleUNet
from ddpm.diffusion import GaussianDiffusion
from ddpm.utils import ModelMeanType, ModelVarType, LossType

def main():
    # ----------- 1. 配置参数 -----------
    epochs = 50
    batch_size = 64
    image_size = 32
    learning_rate = 1e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_dir = "./output/ts_3"
    os.makedirs(save_dir, exist_ok=True)

    # ----------- 2. 加载数据集 -----------
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    dataset = datasets.CIFAR10(root="./public/cifar10", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # ----------- 3. 初始化模型和 Diffusion -----------
    model = SimpleUNet().to(device)

    diffusion = GaussianDiffusion(
        model=model,
        # image_size=image_size,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_LARGE,
        loss_type=LossType.MSE,
        timesteps=1000,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ----------- 4. 训练主循环 -----------
    for epoch in range(epochs):
        pbar = tqdm(dataloader, dynamic_ncols=True)
        for i, (x, _) in enumerate(pbar):
            x = x.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device).long()
            loss = diffusion.p_losses(model, x, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch+1}")
            pbar.set_postfix(Loss=loss.item(), refresh=True)

        # ----------- 5. 每个 epoch 保存采样图像 -----------
        sampled_images = diffusion.sample(model=model, image_size=image_size, batch_size=16)
        utils.save_image(sampled_images, f"{save_dir}/sample_epoch_{epoch+1}.png", normalize=True)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{save_dir}/model_epoch_{epoch+1}.pt")

if __name__ == '__main__':
    main()