import math
import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, skip_connect=True, dropout_rate=0.1):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features) # TODO 分组先不实现
        self.dropout = nn.Dropout(dropout_rate)  #, inplace=True)
        self.act = nn.GELU()
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.scale_act = None

        

    def forward(self, x):
        # x is (HW+1, BT, D)
        xs = self.D_fc1(x)
        xs = self.dropout(self.act(xs))
        xs = self.D_fc2(xs)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x