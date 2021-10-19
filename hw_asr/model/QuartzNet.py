import torch
from torch import nn
from hw_asr.base import BaseModel


class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [batch_size x in_channels x length]
        return self.net(x)


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.channelwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.union_conv = PointwiseConv(in_channels, out_channels)

    def forward(self, x):
        # x:[batch_size x in_channels x seq_len]
        out = self.channelwise_conv(x)
        return self.union_conv(out)


class BaseModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout):
        super().__init__()
        self.net = nn.Sequential(
            SeparableConv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x:[batch_size x in_channels x seq_len]
        return self.net(x)


class BaseBlock(nn.Module):
    def __init__(self, R, in_channels, out_channels, kernel_size, stride, padding, dropout):
        super().__init__()
        my_layers = [BaseModule(in_channels, out_channels, kernel_size, stride, padding,dropout)]
        for i in range(R-2):
            my_layers.append(BaseModule(out_channels, out_channels, kernel_size, stride, padding, dropout))
        self.head = nn.Sequential(
            *my_layers,
            SeparableConv1d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels)
        )
        self.second_head = nn.Sequential(
            PointwiseConv(in_channels, out_channels),
            nn.BatchNorm1d(out_channels)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        fpart = self.head(x)
        spart = self.second_head(x)
        return self.drop(nn.ReLU()(fpart+spart))


class QuartzNet(BaseModel):
    def __init__(self, S, R, B_num, n_feats, n_class, kernel_sizes, channel_sizes,
                 stride=1, padding='same', dropout=0.1, *args, **kwargs):
        """
        Defines model QuartzNet ((S*B_num)x B_num)
        B_num: number of BaseBlocks
        S: number of repeats of each base block
        R: repeats of BaseModule in each BaseBlock
        n_feats: len of input spectrogram
        n_class: length of dictionary
        kernel_sizes: array of kernel sizes as in paper: [C_1, B_1, ..., B_{B_num}, C_2, C_3].
                    C_4 is missed, because it is the last one
        channel_sizes: same as kernel_sizes, but for number of channels
        """
        super().__init__(n_feats, n_class, *args, **kwargs)
        if len(kernel_sizes) != 3 + B_num or len(channel_sizes) != 3 + B_num:
            raise RuntimeError('Check input dimensions')

        self.c1 = ConvBnReLU(n_feats, channel_sizes[0], kernel_sizes[0], stride, padding)
        b_blocks = []
        for i in range(1, B_num+1):
            for _ in range(S):
                b_blocks.append(BaseBlock(R, channel_sizes[i-1], channel_sizes[i], kernel_sizes[i],
                                          stride, padding, dropout))

        self.block_part = nn.Sequential(*b_blocks)
        self.c2 = ConvBnReLU(channel_sizes[-3], channel_sizes[-2], kernel_sizes[-2], stride, padding)
        self.c3 = ConvBnReLU(channel_sizes[-2], channel_sizes[-1], kernel_sizes[-1], stride, padding)
        self.pointwise = PointwiseConv(channel_sizes[-1], n_class)

    def forward(self, spectrogram, *args, **kwargs):
        # spectrogram: [batch_size x input_len x n_feats]
        spectrogram = spectrogram.transpose(-2, -1)
        x = self.c1(spectrogram)
        x = self.block_part(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.pointwise(x)
        return {"logits": x.transpose(-2, -1)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
