import torch
import torch.nn as nn
from .utils import timestep_embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.activation = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.activation(self.norm1(x)))
        h += self.time_emb_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.activation(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, 
                 channel_mults=(1, 2, 4), num_res_blocks=2):
        super().__init__()
        # print("✅ UNet initialized successfully!")

        self.base_channels = base_channels

        time_emb_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        skip_channels = []
        ch = base_channels
        for level, mult in enumerate(channel_mults):
            out_ch = mult * base_channels
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(ch, out_ch, time_emb_dim))
                skip_channels.append(out_ch)
                ch = out_ch
            if level != len(channel_mults) - 1:
                self.down_blocks.append(Downsample(ch))

        # Middle
        self.middle_block1 = ResidualBlock(ch, ch, time_emb_dim)
        self.middle_block2 = ResidualBlock(ch, ch, time_emb_dim)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mults))[::-1]:
            out_ch = mult * base_channels
            for i in range(num_res_blocks + 1):
                if i == num_res_blocks:
                    if level != 0:
                        self.up_blocks.append(Upsample(ch))
                    break

                self.up_blocks.append(
                    ResidualBlock(ch+skip_channels.pop(), out_ch, time_emb_dim)
                )
                ch = out_ch

        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        # print("🔥 Forward started...")
        t_emb = timestep_embedding(t, self.base_channels).to(x.device)
        t_emb = self.time_embedding(t_emb)

        h = self.input_conv(x)
        hs = [h]
        
        for module in self.down_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb) 
                hs.append(h)
            else: 
                h = module(h)

        h = self.middle_block1(h, t_emb)
        h = self.middle_block2(h, t_emb)

        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, t_emb)
            else:
                h = module(h)

        return self.output_conv(h)
