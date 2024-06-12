# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class ME(nn.Module):
    """Motion Exciation (ME) module for TEA. This module is proposed in
    `TEA: Temporal Excitation and Aggregation for Action Recognition`
    refer: https://github.com/Phoenix1327/tea-action-recognition
    """
    def __init__(self, in_channels, num_segments=8, reduction=16):
        super(ME, self).__init__()
        self.in_channels = in_channels
        self.num_segments = num_segments
        self.reduction = reduction
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels//self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.in_channels//self.reduction)

        self.conv2 = nn.Conv2d(
            in_channels=self.in_channels//self.reduction,
            out_channels=self.in_channels//self.reduction,
            kernel_size=3,
            padding=1,
            groups=self.in_channels//self.reduction,
            bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)

        self.conv3 = nn.Conv2d(
            in_channels=self.in_channels//self.reduction,
            out_channels=self.in_channels,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.in_channels)

        self.identity = nn.Identity()

    def forward(self, x):
        
        bottleneck = self.conv1(x) # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck) # nt, c//r, h, w

        # t feature
        reshape_bottleneck = bottleneck.view((-1, self.num_segments) + bottleneck.size()[1:])  # n, t, c//r, h, w
        t_fea, __ = reshape_bottleneck.split([self.num_segments-1, 1], dim=1) # n, t-1, c//r, h, w

        # apply transformation conv to t+1 feature
        conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
        # reshape fea: n, t, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view((-1, self.num_segments) + conv_bottleneck.size()[1:])
        __, tPlusone_fea = reshape_conv_bottleneck.split([1, self.num_segments-1], dim=1)  # n, t-1, c//r, h, w
        
        # motion fea = t+1_fea - t_fea
        # pad the last timestamp
        diff_fea = tPlusone_fea - t_fea # n, t-1, c//r, h, w
        # pad = (0,0,0,0,0,0,0,1)
        diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, t, c//r, h, w
        diff_fea_pluszero = diff_fea_pluszero.view((-1,) + diff_fea_pluszero.size()[2:])  #nt, c//r, h, w
        y = self.avg_pool(diff_fea_pluszero)  # nt, c//r, 1, 1
        y = self.conv3(y)  # nt, c, 1, 1
        y = self.bn3(y)  # nt, c, 1, 1
        y = self.sigmoid(y)  # nt, c, 1, 1
        y = y - 0.5
        output = x + x * y.expand_as(x)
        return output
