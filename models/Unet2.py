import torch
import torch.nn as nn
from .utils import timestep_embedding
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.time_emb_proj = nn.Linear(time_dim, in_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, t_emb=None):
        if t_emb is not None:
            t_emb = self.time_emb_proj(t_emb)  # [B, out_ch]
            t_emb = t_emb[:, :, None, None]  # [B, out_ch, 1, 1]
            x = x + t_emb
        return self.block(x)
    
class TimeEmbedding(nn.Module):
    def __init__(self, time_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        return self.linear(t)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, time_dim)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t_emb=None):
        x = self.conv(x, t_emb)
        x_down = self.down(x)
        return x, x_down  # skip + downsampled

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, time_dim)  # in_ch = skip + upsampled

    def forward(self, x, skip, t_emb):
        x = self.up(x)

        # Resize x to match skip (in case of size mismatch)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        
        return self.conv(x, t_emb)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        time_dim = 4 * base_channels
        self.time_mlp = TimeEmbedding(base_channels, time_dim)

        self.down1 = Down(in_channels, 64, time_dim)
        self.down2 = Down(64, 128, time_dim)
        self.down3 = Down(128, 256, time_dim)

        self.middle = DoubleConv(256, 256, time_dim)

        self.up3 = Up(256, 128, 128, time_dim)
        self.up2 = Up(128, 64, 64, time_dim)
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x, t):
        t_emb = timestep_embedding(t, 64).to(x.device)
        t_emb = self.time_mlp(t_emb)

        skip1, x = self.down1(x)  # skip=64
        skip2, x = self.down2(x, t_emb)  # skip=128
        skip3, x = self.down3(x, t_emb)

        x = self.middle(x, t_emb)

        x = self.up3(x, skip2, t_emb)
        x = self.up2(x, skip1, t_emb)

        return self.final(x)