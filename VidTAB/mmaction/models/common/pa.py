# Code for paper:
# [Title]  - "PAN: Towards Fast Action Recognition via Learning Persistence of Appearance"
# [Author] - Can Zhang, Yuexian Zou, Guang Chen, Lei Gan
# [Github] - https://github.com/zhang-can/PAN-PyTorch

import torch
from torch import nn

class PA(nn.Module):
    """ Persistence Appearance (PA) Module from
    `PAN: Towards Fast Action Recognition via Learning Persistence of Appearance`,
    refer: https://github.com/zhang-can/PAN-PyTorch
    """
    def __init__(self, n_length):
        super(PA, self).__init__()
        self.shallow_conv = nn.Conv2d(3,8,7,1,3)
        self.n_length = n_length
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.001)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        # shape of input x: (N*n_length, C, H, W)
        # shape of output: (N, n_length-1, H, W)
        h, w = x.size(-2), x.size(-1)
        x = self.shallow_conv(x)
        x = x.view(-1, self.n_length, x.size(-3), x.size(-2)*x.size(-1)) # N T C H*W
        for i in range(self.n_length-1):
            d_i = nn.PairwiseDistance(p=2)(x[:,i,:,:], x[:,i+1,:,:]).unsqueeze(1)
            # 高版本pytorch需要改成  nn.PairwiseDistance(p=2)(x[:,i,:,:].transpose(-1, -2), x[:,i+1,:,:].transpose(-1, -2)).unsqueeze(1)
            d = d_i if i == 0 else torch.cat((d, d_i), 1)
        return d.view(-1, self.n_length-1, h, w)
