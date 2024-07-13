import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1):
        super(DepthwiseSeparableConvolution, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv1d(nin, nin, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      groups=nin,
                      bias=False),
            nn.BatchNorm1d(nin),
            nn.GELU(),
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(nin, nout, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(nout),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class ECALayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)

        # Two different branches of ECA module
        y = self.conv(y).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, use_act=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv1d(in_channels, init_channels, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm1d(init_channels),
            nn.GELU() if use_act else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv1d(init_channels, new_channels, kernel_size=dw_size, stride=1,
                      padding=(dw_size - 1) // 2, groups=init_channels, bias=False),
            nn.BatchNorm1d(new_channels),
            nn.GELU() if use_act else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)

        return out[:, :self.out_channels, :]


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, L = x.size()
        g = self.groups

        return x.view(N, g, C // g, L).permute(0, 2, 1, 3).reshape(N, C, L)


class ShuffleGhostBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, stride=1, use_se=False, shuffle=False):
        super(ShuffleGhostBottleneck, self).__init__()
        assert stride in [1, 2]
        # hidden_channels = hidden_ratio * in_channels

        self.shuffle = ShuffleBlock(groups=2) if shuffle == 2 else nn.Sequential()

        self.conv = nn.Sequential(
            # pw
            GhostModule(in_channels, hidden_channels, kernel_size=1, use_act=True),

            # dw
            nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2, groups=hidden_channels, bias=False),
                nn.BatchNorm1d(hidden_channels),
            ) if stride == 2 else nn.Sequential(),

            # Squeeze-and-Excite
            ECALayer(hidden_channels) if use_se else nn.Sequential(),

            # pw-linear
            GhostModule(hidden_channels, out_channels, kernel_size=1, use_act=False),
        )

        if in_channels == out_channels and stride == 1:
            self.shortcut = lambda x: x
        else:
            self.shortcut = DepthwiseSeparableConvolution(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv(self.shuffle(x))

        return x + residual


class NetworkCore(nn.Module):
    def __init__(self, cfgs, num_classes=26):
        super(NetworkCore, self).__init__()
        self.cfgs = cfgs
        num_stages = len(self.cfgs)
        in_proj_channel = self.cfgs[0][0][0]
        out_proj_channel = self.cfgs[-1][-1][2]

        self.in_proj = nn.Sequential(
            nn.Conv1d(12, in_proj_channel, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(in_proj_channel),
            nn.LeakyReLU(inplace=True),
        )

        layers = []
        for i in range(num_stages):
            for in_c, hid_c, out_c, k, s, use_se, shuffle in self.cfgs[i]:
                layers.append(ShuffleGhostBottleneck(in_c, hid_c, out_c, k, s, use_se, shuffle))
        self.layers = nn.Sequential(*layers)

        self.mha = nn.MultiheadAttention(out_proj_channel, 8)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.fc_1 = nn.Linear(out_proj_channel + 12, num_classes)
        self.ch_fc1 = nn.Linear(num_classes, out_proj_channel)
        self.ch_bn = nn.BatchNorm1d(out_proj_channel)
        self.ch_fc2 = nn.Linear(out_proj_channel, num_classes)

    def forward(self, x, l):
        x = self.in_proj(x)
        x = self.layers(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = x.squeeze(2).permute(2, 0, 1)
        x, s = self.mha(x, x, x)
        x = x.permute(1, 2, 0)
        x = self.pool(x).squeeze(2)
        x = torch.cat((x, l), dim=1)

        x = self.fc_1(x)
        p = x.detach()
        p = F.leaky_relu(self.ch_bn(self.ch_fc1(p)))
        p = torch.sigmoid(self.ch_fc2(p))

        return x, p


if __name__ == '__main__':
    # in_channels, hidden_channels, out_channels, kernel_size, stride, use_se, shuffle
    cfgs = [
        [
            [32, 48, 32, 3, 2, 0, 0],
            [32, 64, 32, 3, 1, 1, 1],
        ],
        [
            [32, 96, 64, 3, 2, 1, 0],
            [64, 128, 64, 3, 1, 1, 1],
            [64, 160, 64, 3, 1, 0, 0],
            [64, 192, 64, 3, 1, 1, 1],
        ],
        [
            [64, 144, 96, 5, 2, 1, 0],
            [96, 192, 96, 5, 1, 1, 1],
            [96, 240, 96, 5, 1, 0, 0],
            [96, 248, 96, 5, 1, 1, 1],
        ],
        [
            [96, 192, 128, 3, 2, 1, 0],
            [128, 256, 128, 3, 1, 1, 1],
            [128, 320, 128, 3, 1, 0, 0],
            [128, 384, 128, 3, 1, 1, 1],
        ],
        [
            [128, 512, 256, 5, 2, 1, 0],
            [256, 512, 256, 5, 1, 1, 1],
        ],
    ]

    x = torch.rand(2, 12, 8192)
    l = torch.Tensor([
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    ]).type(torch.LongTensor)
    model = NetworkCore(cfgs)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    out, p = model(x, l)
    print(out.shape)
    print(p.shape)
