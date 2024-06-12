import torch.nn as nn
import torch
from ..utils.embed import AdaptivePadding1d

class TemporalConv3d(nn.Module):
    """A 3d version of temporal conv to improve the temporal modeling ability of 3d CNN.
    1D Temporal convolutions, the convs are initialized to act as the "Part shift" layer
    """
    def __init__(self, in_channels, kernel_size=8, n_div=8, mode='shift'):
        super(TemporalConv3d, self).__init__()
        self.in_channels = in_channels
        self.fold_div = n_div
        assert self.in_channels % self.fold_div == 0
        self.fold = self.in_channels // self.fold_div
        self.ada = AdaptivePadding1d(kernel_size=kernel_size)
        self.conv = nn.Conv1d(self.in_channels, self.in_channels,
                kernel_size=kernel_size, padding=0, groups=self.in_channels,
                bias=False)

        mid = kernel_size // 2
        if mode == 'shift':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, (mid+1):] = 1 / (kernel_size-mid-1) # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0:mid] = 1 / mid # shift right
            if 2*self.fold < self.in_channels:
                self.conv.weight.data[2 * self.fold:, 0, mid] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, mid] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        n, c, t, h, w = x.size()
        x = x.permute(0, 3, 4, 1, 2) # (n, h, w, c, t)
        x = x.contiguous().view(n*h*w, c, t)
        x = self.ada(x)
        x = self.conv(x)
        x = x.view(n, h, w, c, t)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return x


class TemporalConv(nn.Module):
    """Also known as ShiftModule, used by TDN, EAN...
    1D Temporal convolutions, the convs are initialized to act as the "Part shift" layer
    """
    def __init__(self, in_channels, n_segment=8, n_div=8, mode='shift', dwconv_cfg=dict(kernel_size=3, padding=1)):
        super(TemporalConv, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.in_channels // self.fold_div
        self.conv = nn.Conv1d(self.fold_div*self.fold, self.fold_div*self.fold,
                kernel_size=dwconv_cfg.get('kernel_size'), padding=dwconv_cfg.get('padding'), groups=self.fold_div*self.fold,
                bias=False)

        if mode == 'shift': # 初始化要不要扩大感受野就再说
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1 # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
            if 2*self.fold < self.in_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute(0, 3, 4, 2, 1) # (n_batch, h, w, c, n_segment)
        x = x.contiguous().view(n_batch*h*w, c, self.n_segment)
        x = self.conv(x) # (n_batch*h*w, c, n_segment)
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute(0, 4, 3, 1, 2) # (n_batch, n_segment, c, h, w)
        x = x.contiguous().view(nt, c, h, w)
        return x

class TemporalConv2d(nn.Module):
    """Temporal Convolution Block
    Args:
        net (nn.module): Module to make temporal convolution.
        num_segments (int): Number of frame segments. Default: 3.
        kernel_size (tuple[int, int] or int): Size of the temporal convolving kernel (Conv2D) Default: (3, 1).
        stride (tuple[int, int] or int): Default: (1, 1).
        padding (tuple[int, int] or int): Default: (1, 0).
        init_type (str): The way of initializing temporal convolution weight,
            which is chosen from ['tsm', 'r_tsm', 'tsn'] or anything else. Default: 'tsm'. 
        shift_div (int): Number of divisions for shift when the weights
            are initialized in 'tsm' way. Default: 8. 
        requires_grad (bool): Default: True.
    """
    def __init__(self,
                 in_channels,
                 num_segments,
                 kernel_size=(3, 1),
                 stride=(1, 1),
                 padding=(1, 0),
                 init_type='tsm',
                 shift_div=8,
                 requires_grad=True
                 ):
        super().__init__()

        self.num_segments = num_segments
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.shift_div = shift_div
        self.requires_grad = requires_grad
        self.num_channels = in_channels

        # TC is a channel-wise/depth-wise 2D/1D convolution
        self.dwconv = nn.Conv2d(self.num_channels, self.num_channels,
                                self.qq, self.stride, self.padding,
                                groups=self.num_channels, bias=True) 
        self.dwconv.weight.data.requires_grad = self.requires_grad

        conv_params = self.dwconv.weight.data 
        if isinstance(self.kernel_size, int):
            conv_params = torch.zeros((self.num_channels, 1, self.kernel_size, 1)) 
        else:
            conv_params = torch.zeros((self.num_channels, 1, *self.kernel_size)) 

        # TSM initialization
        if init_type == 'r_tsm':
            for i in range(self.num_channels):
                import random
                j = random.randint(0, kernel_size - 1)
                conv_params[i, :, j] = 1
        elif init_type == 'tsm': # Temporal shift
            fold = self.num_channels // shift_div
            conv_params[:fold, :, kernel_size[0] // 2 + 1] = 1
            conv_params[fold:2 * fold, :, kernel_size[0] // 2 - 1] = 1
            conv_params[2 * fold:, :, kernel_size[0] // 2] = 1
        elif init_type == 'tsn': # Identical mapping
            conv_params[:, :, kernel_size // 2] = 1  
        else:
            nn.init.kaiming_uniform_(self.weight, a=2)

        self.dwconv.weight.data = conv_params



    def forward(self, x):

        return self.dwconv(x)