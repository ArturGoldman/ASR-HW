import torch
from torch import nn
from hw_asr.base import BaseModel
from collections import OrderedDict


class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, is_tcs=False, add_ReLU=True):
        super().__init__()
        mod_list = [
            nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, dilation=dilation)
        ]

        if is_tcs:
            mod_list.append(PointwiseConv(out_channels, out_channels))

        mod_list.append(nn.BatchNorm1d(out_channels))

        if add_ReLU:
            mod_list.append(nn.ReLU())

        self.net = nn.Sequential(*mod_list)

    def forward(self, x):
        # x: [batch_size x in_channels x length]
        return self.net(x)


class BaseBlock(nn.Module):
    def __init__(self, R, in_channels, out_channels, kernel_size, dropout, padding='same', is_tcs=True):
        super().__init__()
        my_layers = [ConvBn(in_channels, out_channels, kernel_size, padding=padding, is_tcs=is_tcs)]
        for i in range(R-2):
            my_layers.append(ConvBn(out_channels, out_channels, kernel_size, padding=padding, is_tcs=is_tcs))

        self.head = nn.Sequential(
            *my_layers,
            ConvBn(out_channels, out_channels, kernel_size, padding=padding, is_tcs=is_tcs, add_ReLU=False),
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
                 dropout=0.1, *args, **kwargs):
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

        self.kernel_sizes = kernel_sizes

        self.C1 = ConvBn(n_feats, channel_sizes[0], kernel_sizes[0], stride=2)
        b_blocks = []
        for i in range(1, B_num+1):
            for j in range(S):
                b_blocks.append(('B{}-{}'.format(i, j+1),
                                 BaseBlock(R, channel_sizes[i-1], channel_sizes[i], kernel_sizes[i], dropout,
                                           padding='same', is_tcs=True)))

        self.block_part = nn.Sequential(OrderedDict(b_blocks))
        self.conv_ending = nn.Sequential(OrderedDict([
            ('C2', ConvBn(channel_sizes[-3], channel_sizes[-2], kernel_sizes[-2], dilation=2)),
            ('C3', ConvBn(channel_sizes[-2], channel_sizes[-1], kernel_sizes[-1])),
            ('C4', PointwiseConv(channel_sizes[-1], n_class)),
            ])
        )

    def forward(self, spectrogram, *args, **kwargs):
        # spectrogram: [batch_size x input_len x n_feats]
        spectrogram = spectrogram.transpose(-2, -1)
        x = self.C1(spectrogram)
        x = self.block_part(x)
        x = self.conv_ending(x)
        return {"logits": x.transpose(-2, -1)}

    def transform_input_lengths(self, input_lengths):
        # C1 changed len
        new_ln = torch.floor((input_lengths-self.kernel_sizes[0])/2 + 1).int()

        # C2 changed len
        new_ln = new_ln-2*(self.kernel_sizes[-2] - 1)

        # C3 changed len (based on paper it also doesn't change len: K should be 1
        new_ln = new_ln-self.kernel_sizes[-1] + 1

        # C4 does not change len
        return new_ln
