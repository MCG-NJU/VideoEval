import torch.nn as nn
import torch.nn.functional as F


class ShortTermTDM(nn.Module):
    """ This module are proposed in 
    `Temporal Difference Networks for Efficient Action Recognition`.
    """

    def __init__(self):
        super(ShortTermTDM, self).__init__()
        raise NotImplementedError(
            "It's implemented by .backbone/tdn.py, not here!!!")


class LongTermTDM(nn.Module):
    """ This module are proposed in 
    `Temporal Difference Networks for Efficient Action Recognition`.
    """

    def __init__(self, channel, n_segment=8, index=1):
        super(LongTermTDM, self).__init__()
        self.channel = channel
        self.reduction = 16
        self.n_segment = n_segment
        self.stride = 2**(index - 1)
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel // self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel // self.reduction)
        self.conv2 = nn.Conv2d(
            in_channels=self.channel // self.reduction,
            out_channels=self.channel // self.reduction,
            kernel_size=3,
            padding=1,
            groups=self.channel // self.reduction,
            bias=False)

        self.avg_pool_forward2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.avg_pool_forward4 = nn.AvgPool2d(kernel_size=4, stride=4)

        self.sigmoid_forward = nn.Sigmoid()

        self.avg_pool_backward2 = nn.AvgPool2d(
            kernel_size=2, stride=2)  # nn.AdaptiveMaxPool2d(1)
        # self.avg_pool_backward4 = nn.AvgPool2d(kernel_size=4, stride=4)

        self.sigmoid_backward = nn.Sigmoid()

        self.pad1_forward = (0, 0, 0, 0, 0, 0, 0, 1)  # 给时间维后面pad
        self.pad1_backward = (0, 0, 0, 0, 0, 0, 1, 0)  # 给时间维前面pad

        self.conv3 = nn.Conv2d(
            in_channels=self.channel // self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        self.conv3_smallscale2 = nn.Conv2d(
            in_channels=self.channel // self.reduction,
            out_channels=self.channel // self.reduction,
            padding=1,
            kernel_size=3,
            bias=False)
        self.bn3_smallscale2 = nn.BatchNorm2d(num_features=self.channel //
                                              self.reduction)

        self.conv3_smallscale4 = nn.Conv2d(
            in_channels=self.channel // self.reduction,
            out_channels=self.channel // self.reduction,
            padding=1,
            kernel_size=3,
            bias=False)
        self.bn3_smallscale4 = nn.BatchNorm2d(num_features=self.channel //
                                              self.reduction)

    def forward(self, x):
        bottleneck = self.conv1(x)  # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck)  # nt, c//r, h, w
        reshape_bottleneck = bottleneck.view(
            (-1, self.n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w

        t_fea_forward, _ = reshape_bottleneck.split(
            [self.n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w
        _, t_fea_backward = reshape_bottleneck.split(
            [1, self.n_segment - 1], dim=1)  # n, t-1, c//r, h, w

        conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view(
            (-1, self.n_segment) +
            conv_bottleneck.size()[1:])  # n, t, c//r, h, w
        _, tPlusone_fea_forward = reshape_conv_bottleneck.split(
            [1, self.n_segment - 1], dim=1)  # n, t-1, c//r, h, w
        tPlusone_fea_backward, _ = reshape_conv_bottleneck.split(
            [self.n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w
        diff_fea_forward = tPlusone_fea_forward - t_fea_forward  # n, t-1, c//r, h, w
        diff_fea_backward = tPlusone_fea_backward - t_fea_backward  # n, t-1, c//r, h, w
        diff_fea_pluszero_forward = F.pad(
            diff_fea_forward, self.pad1_forward, mode="constant",
            value=0)  # n, t, c//r, h, w
        diff_fea_pluszero_forward = diff_fea_pluszero_forward.view(
            (-1, ) + diff_fea_pluszero_forward.size()[2:])  # nt, c//r, h, w
        diff_fea_pluszero_backward = F.pad(
            diff_fea_backward, self.pad1_backward, mode="constant",
            value=0)  # n, t, c//r, h, w
        diff_fea_pluszero_backward = diff_fea_pluszero_backward.view(
            (-1, ) + diff_fea_pluszero_backward.size()[2:])  # nt, c//r, h, w
        y_forward_smallscale2 = self.avg_pool_forward2(
            diff_fea_pluszero_forward
        )  # nt, c//r, 1, 1 这个shape应该是 nt, c//r, h//2, w//2吧
        y_backward_smallscale2 = self.avg_pool_backward2(
            diff_fea_pluszero_backward)  # nt, c//r, 1, 1 同上

        y_forward_smallscale4 = diff_fea_pluszero_forward
        y_backward_smallscale4 = diff_fea_pluszero_backward
        y_forward_smallscale2 = self.bn3_smallscale2(
            self.conv3_smallscale2(y_forward_smallscale2))
        y_backward_smallscale2 = self.bn3_smallscale2(
            self.conv3_smallscale2(y_backward_smallscale2))

        y_forward_smallscale4 = self.bn3_smallscale4(
            self.conv3_smallscale4(y_forward_smallscale4))
        y_backward_smallscale4 = self.bn3_smallscale4(
            self.conv3_smallscale4(y_backward_smallscale4))

        y_forward_smallscale2 = F.interpolate(
            y_forward_smallscale2,
            diff_fea_pluszero_forward.size()[2:])
        y_backward_smallscale2 = F.interpolate(
            y_backward_smallscale2,
            diff_fea_pluszero_backward.size()[2:])

        y_forward = self.bn3(
            self.conv3(1.0 / 3.0 * diff_fea_pluszero_forward +
                       1.0 / 3.0 * y_forward_smallscale2 +
                       1.0 / 3.0 * y_forward_smallscale4))  # nt, c, 1, 1
        y_backward = self.bn3(
            self.conv3(1.0 / 3.0 * diff_fea_pluszero_backward +
                       1.0 / 3.0 * y_backward_smallscale2 +
                       1.0 / 3.0 * y_backward_smallscale4))  # nt, c, 1, 1

        y_forward = self.sigmoid_forward(y_forward) - 0.5  # 不知道直接用tanh会不会有影响
        y_backward = self.sigmoid_backward(y_backward) - 0.5

        y = 0.5 * y_forward + 0.5 * y_backward
        output = x + x * y
        return output
