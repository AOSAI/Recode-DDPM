import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t):
        return self.mlp(t)[:, :, None, None]  # for broadcasting

class NaiveConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_dim=256):
        super().__init__()

        self.time_mlp = TimeEmbedding(time_dim, 128)

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        )

        self.time_proj = nn.Conv2d(128, 128, kernel_size=1)  # 将时间信息加到 feature 上

    def forward(self, x, t_emb):
        t = self.time_mlp(t_emb)
        out = self.conv_net[:3](x)  # 取前几层得到 feature map
        out = out + self.time_proj(t.expand_as(out))  # 融合时间信息
        out = self.conv_net[3:](out)  # 继续后续卷积
        return out