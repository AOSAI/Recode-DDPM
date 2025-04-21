## 主循环训练

```py
for epoch in range(epochs):
    pbar = tqdm(dataloader, dynamic_ncols=True)
    for i, (x, _) in enumerate(pbar):
```

外层循环，控制训练多少个 epoch（轮次），每一轮，模型会把训练数据都过一遍。

pbar 是用 tqdm 包包装的 dataloader，用来显示一个带进度条的漂亮加载器 ✨，dataloader 是一批一批从数据集中抓样本，batch_size 是提前设定好的，dynamic_ncols=True 表示进度条宽度会自动根据终端大小调整（不重要，但好用）。

内层循环，遍历这个 epoch 里的所有 batch，(x, \_\) 表示从数据集中抓到一组样本 x，_ 是标签（并没有做分类任务，不需要它，所以用 _ 忽略），i 是当前 batch 的编号（可选）。

内层循环中，就是正常的调用模型训练的内容。

## 保存采样图像和模型参数

```py
sampled_images = diffusion.sample(model=model, image_size=image_size, batch_size=16)
utils.save_image(sampled_images, f"{save_dir}/sample_epoch_{epoch+1}.png", normalize=True)
```

保存采样图像，首先是调用 diffusion.sample 函数，同一个批次采样 16 张图像出来。

然后通过 torchvision 中的 utils.save_image 函数，保存这些图像。normalize=True 表示给图像做归一化操作，自动把图像的像素值缩放到 [0, 1] 区间 来保存。

因为采样出来的图像 sampled_images 一般在 [-1, 1]，因为扩散模型通常这样做归一化。但图像保存的时候，像素值要在 [0, 1]（浮点格式）或者 [0, 255]（整型）之间，不然就会全黑或者全白。

```py
torch.save(model.state_dict(), f"{save_dir}/model_epoch_{epoch+1}.pt")
```

这句话就是 pytorth 中标准的保存模型参数的方法。
