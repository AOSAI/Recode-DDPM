import torch
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.Unet1 import SimpleUNet as Unet1
from models.Unet2 import SimpleUNet as Unet2
from models.Unet3 import NaiveConvNet as Unet3
from ddpm.diffusion import GaussianDiffusion
from ddpm.utils import ModelMeanType, ModelVarType, LossType

def main(modelType):
    # ----------- 1. 配置参数 -----------
    epochs = 300
    batch_size = 16
    image_size = 128
    learning_rate = 1e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('mps' if (torch.backends.mps.is_available()) else 'cpu')

    save_dir = "./output/training"
    os.makedirs(save_dir, exist_ok=True)

    # ----------- 2. 加载数据集 -----------
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # ImageFolder 函数至少需要创建一个子文件夹，比如 dataImg/xxxx
    dataset = datasets.ImageFolder(root="./public/dataImg", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # ----------- 3. 初始化模型和 Diffusion -----------
    if modelType == "unet1":
        model = Unet1().to(device)
    elif modelType == "unet2":
        model = Unet2().to(device)
    else:
        model = Unet3().to(device)

    diffusion = GaussianDiffusion(
        model=model,
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

        # ----------- 5. 保存模型参数 -----------
        if epoch == 0 or (epoch + 1) % 15 == 0:
            torch.save(model.state_dict(), f"{save_dir}/model_epoch_{epoch+1}.pt")

if __name__ == '__main__':
    main()