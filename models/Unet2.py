import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.down(x)
        return x, x_down  # skip + downsampled

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = DoubleConv(in_ch, out_ch)  # in_ch = skip + upsampled

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        return self.linear(t)[:, :, None, None]  # Expand for broadcasting

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_dim=256):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_dim, 128)

        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.middle = DoubleConv(256, 256)

        self.up3 = Up(256 + 128, 128)
        self.up2 = Up(128 + 64, 64)
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x, t_emb):
        t_emb = self.time_mlp(t_emb)

        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)

        x = self.middle(x)

        x = self.up3(x, skip2)
        x = self.up2(x, skip1)

        # Broadcast t_emb and add to final feature map
        x = x + t_emb
        return self.final(x)