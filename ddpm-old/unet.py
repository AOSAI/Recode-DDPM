from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    zero_module,
    normalization,
    timestep_embedding,
)

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        这是一个 抽象基类 ( Abstract Base Class ), 或者说是模板. 虽然不实现具体的
        forward() 函数, 但强制所有子类都要自己实现这个带 x 和 emb 的 forward(x, emb) 方法
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    一个顺序模块，将"时间步嵌入"作为额外输入传递给支持它的子模块
    """
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    :param channels: 输入和输出的通道。
    :param use_conv: 决定是否应用卷积的布尔变量(bool)。
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        # 确保输入 x 的通道数等于 self.channels，避免后续出错
        assert x.shape[1] == self.channels
        # 相当于 F.interpolate(x, size=(H * 2, W * 2), mode='nearest')
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # 上采样之后再接一个卷积层，因为上采样后只是“扩大尺寸”，没有学习新特征
        # 用于 提取特征、降噪 / 消除插值痕迹、调整通道数
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    :param channels: 输入和输出的通道。
    :param use_conv: 决定是否应用卷积的布尔变量(bool)。
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # stride=2 表示卷积核每次滑动两个像素点，这会让输出尺寸变为原来的一半。
        # 通常用于下采样，代替 F.max_pool2d 或 F.avg_pool2d 这样的池化操作。
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    可选择更改通道数量的残差块。

    :param channels: 输入通道的数量。
    :param emb_channels: 时间步嵌入通道的数量。
    :param dropout: the rate of dropout.
    :param out_channels: 如果指定，则表示输出通道的数量。
    :param use_conv: 如果为 True 且指定了 out_channels, 则在跳转连接中使用空间卷积而不是较小的 1x1 卷积来改变通道。
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        # dims=2,
        # use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1)
        )
        emb_linear = 2 * self.out_channels if use_scale_shift_norm else self.out_channels
        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(emb_channels, emb_linear),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    # x 是图像输入张量，形状一般是 [B, C, H, W]
    # emb 是条件向量，比如时间步 t 的 embedding，或者类条件 embedding
    def forward(self, x, emb):
        # 主干网络的输入处理 x → norm → SiLU → Conv2d → h
        h = self.in_layers(x)
        # embedding 的线性映射处理 .type(h.dtype)是统一精度，防止精度不一致出错
        emb_out = self.emb_layers(emb).type(h.dtype)
        # 维度对齐（为了 broadcast）
        # 如果 h 是 [B, C, H, W]，而 emb_out 是 [B, C]，那么我们需要变成 [B, C, 1, 1]
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        # scale-shift norm 是一个更强的融合条件 embedding 的方法：
        # 把 embedding 映射成 scale 和 shift 两部分
        # 对 h 做归一化后，用 scale 放缩，用 shift 平移，比简单的加法更有表达能力
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    这块代码为原版 DDPM 中的 Self-Attention Block, 是可选的，原版没怎么用.
    是 OpenAI 做的 IDM 和 GDM, 以及现在很火的 SDM 才把注意力机制大量的引入.
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = normalization(channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_in = x
        x = self.norm(x)
        q = self.q(x).reshape(B, C, -1).permute(0, 2, 1)   # [B, HW, C]
        k = self.k(x).reshape(B, C, -1)                    # [B, C, HW]
        v = self.v(x).reshape(B, C, -1).permute(0, 2, 1)   # [B, HW, C]

        attn = th.bmm(q, k) * (C ** -0.5)               # [B, HW, HW]
        attn = F.softmax(attn, dim=2)
        out = th.bmm(attn, v)                           # [B, HW, C]
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return x_in + self.proj_out(out)

class UNetModel(nn.Module):
    """
    带有注意力和时间步嵌入的完整 UNet 模型。

    :param in_channels: 输入张量中的通道数。
    :param model_channels: 模型的基本通道数。
    :param out_channels: 输出张量中的通道数。
    :param num_res_blocks: 每个缩小样本的残差块数。
    :param attention_resolutions: 使用注意力机制的下采样率集合。可以是集合、列表或元组。例如，如果包含 4, 则在 4 倍频下采样时，将使用关注度。
    :param dropout: the dropout probability.
    :param channel_mult: UNet 各级的信道乘数。
    :param conv_resample: 如果为 True, 则使用学习到的卷积进行上采样和下采样。
    :param dims: 确定信号是一维、二维还是三维。
    :param num_classes: 如果指定(比如 int), 则此模型将以 `num_classes` 类作为类条件。
    :param use_checkpoint: 使用梯度检查点来减少内存使用量。
    :param num_heads: 每个注意力层的注意力头数。
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        # num_classes=None,
        # use_checkpoint=False,
        # num_heads=1,
        # num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        # if num_heads_upsample == -1:
        #     num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        # self.num_classes = num_classes
        # self.use_checkpoint = use_checkpoint
        # self.num_heads = num_heads
        # self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # if self.num_classes is not None:
        #     self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # 下采样过程
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        # level表示索引下标，mult表示下采样的通道倍率
        for level, mult in enumerate(channel_mult):
            # 2个残差块，也就是每一个残差块都走一次mult的倍率，128，256，512，1024
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            # 2个残差块同时进行下采样，最后一次不再执行
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch)))
                input_block_chans.append(ch)

        # 中间过程
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # 上采样过程
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                # 这里的 if level 表示的是 if level != 0
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1)),
        )

    # def convert_to_fp16(self):
    #     """
    #     Convert the torso of the model to float16.
    #     """
    #     self.input_blocks.apply(convert_module_to_f16)
    #     self.middle_block.apply(convert_module_to_f16)
    #     self.output_blocks.apply(convert_module_to_f16)

    # def convert_to_fp32(self):
    #     """
    #     Convert the torso of the model to float32.
    #     """
    #     self.input_blocks.apply(convert_module_to_f32)
    #     self.middle_block.apply(convert_module_to_f32)
    #     self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        @property 为 Python 属性装饰器, 可以像属性一样访问，而不是函数调用
        查询一下 self.input_blocks 里第一个参数的 dtype, 然后返回
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        """
        将模型应用到输入批次中。

        :param x: 一个 [N x C x ...] 形状的输入张量。
        :param timesteps: 一个 1-D 批次的时间步。
        :param y: 一个 [N] 标签张量，如果以类别为条件。
        :return: 一个 [N x C x ...] 形状的输出张量。
        """
        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # if self.num_classes is not None:
        #     assert y.shape == (x.shape[0],)
        #     emb = emb + self.label_emb(y)

        # 把输入的 x 强制转换成模型内部所使用的精度（float16 或 float32）
        # 只做 DDPM 可以全 32 精度，可以删除
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        应用模型并返回所有中间张量。

        :param x: 一个 [N x C x ...] 形状的输入张量。
        :param timesteps: 一个 1-D 批次的时间步。
        :param y: 一个 [N] 标签张量，如果以类别为条件。
        :return: 一个包含以下键(key)的字典(dict):
                 - 'down': 下采样得到的隐藏状态张量列表。
                 - 'middle': 模型中分辨率最低区块的输出张量。
                 - 'up': 上采样产生的隐藏状态张量列表。
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # if self.num_classes is not None:
        #     assert y.shape == (x.shape[0],)
        #     emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result
