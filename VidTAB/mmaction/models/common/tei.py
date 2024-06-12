import torch
from torch import nn

import torch.nn as nn
class TEI(nn.Module):
    """Temporal Enhanced-Interaction Module in TEINet

    This module is proposed in `TEINet: Towards an Efficient Architecture
    for Video Recognition <https://arxiv.org/pdf/1911.09435>`

    """
    def __init__(self,
                 in_channels,
                 num_segments,
                 mem_cfg=dict(),
                 tim_cfg=dict()
                 ):
        super().__init__()
        self.num_channels = in_channels
        self.num_segments = num_segments
        self.mem = MEM(self.num_channels, **mem_cfg)
        self.tim = TIM(self.num_channels, **tim_cfg)

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.num_segments
        new_x = x.view(n_batch, self.num_segments, c, h * w)
        out = self.mem(new_x)
        out = self.tim(out)
        out = out.view(nt, c, h, w)
        
        return out

class MEM(nn.Module):
    """Motion Enhanced Module in TEINet

    This module is proposed in `TEINet: Towards an Efficient Architecture
    for Video Recognition <https://arxiv.org/pdf/1911.09435>`

    """
    def __init__(self,
                 in_channels,
                 reduction_ratio=8,
                 requires_grad=True
                 ):
        super().__init__()
        self.num_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(self.num_channels, self.num_channels // self.reduction_ratio) # conv1
        self.fc2 = nn.Linear(self.num_channels, self.num_channels // self.reduction_ratio) # conv2
        self.fc3 = nn.Linear(self.num_channels // self.reduction_ratio, self.num_channels) # conv3
        self.sigmoid = nn.Sigmoid()

        self.requires_grad = requires_grad
        self.fc1.weight.data.requires_grad = self.requires_grad
        self.fc2.weight.data.requires_grad = self.requires_grad
        self.fc3.weight.data.requires_grad = self.requires_grad

    def forward(self, x):
        """
        Shape of input x: (n, t, c, h*w)
        Shape of output: (n, t, c, h*w)
        """
        x_squeeze = x.mean(dim=-1) # global_avgpool
        x_prev = x_squeeze[:, :-1, :]
        x_next = x_squeeze[:, 1:, :]
        x_diff = self.fc3(self.sigmoid(self.fc2(x_next) - self.fc1(x_prev))).unsqueeze(dim=-1)
        out = torch.cat([x_diff * x[:, :-1, :, :], x[:, -1:, :, :]], dim=1)
        # out = x.clone() #  没有用cat快
        # out[:, :-1, :, :] = x[:, :-1, :, :] * x_diff # 这样比 *= 操作少一次clone更快
        return out


class TIM(nn.Module):
    """Temporal Interaction Module in TEINet

    This module is proposed in `TEINet: Towards an Efficient Architecture
    for Video Recognition <https://arxiv.org/pdf/1911.09435>`

    Args:
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
                 kernel_size=(3, 1),
                 stride=(1, 1),
                 padding=(1, 0),
                 init_type='tsm',
                 shift_div=8,
                 requires_grad=True
                 ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.shift_div = shift_div
        self.requires_grad = requires_grad
        self.num_channels = in_channels


        # TIM is a channel-wise/depth-wise 2D/1D convolution
        self.dwconv = nn.Conv2d(self.num_channels, self.num_channels,
                                self.kernel_size, self.stride, self.padding,
                                groups=self.num_channels, bias=False) 
        self.dwconv.weight.data.requires_grad = self.requires_grad

        conv_params = self.dwconv.weight.data 
        if isinstance(self.kernel_size, int): # 默认是正态分布初始化的，所以需要先填0
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
        """
        Shape of input x: (n, t, c, h*w)
        Shape of output: (n, t, c, h*w)
        """
        x = x.permute(0, 2, 1, 3).contiguous()
        x = self.dwconv(x)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x
