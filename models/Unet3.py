import torch.nn as nn
from .utils import timestep_embedding

class TimeAwareConv(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.act = nn.ReLU()

    def forward(self, x, t_emb=None):
        # t_emb: [B, time_dim] → [B, out_ch, 1, 1]
        if t_emb == None:
            result = self.act(self.conv(x))
        else:
            t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
            result = self.act(self.conv(x) + t)
        return result

class NaiveConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_dim=128):
        super().__init__()

        self.conv1 = TimeAwareConv(in_channels, 64, time_dim)
        self.conv2 = TimeAwareConv(64, 128, time_dim)
        self.conv3 = TimeAwareConv(128, 128, time_dim)
        self.final = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        # 1. 生成时间嵌入编码
        t_emb = timestep_embedding(t, 128).to(x.device)

        # 2. 每层融合时间信息
        x = self.conv1(x)
        x = self.conv2(x, t_emb)
        x = self.conv3(x, t_emb)

        # 3. 最后一层不再加时间了
        x = self.final(x)
        return x
