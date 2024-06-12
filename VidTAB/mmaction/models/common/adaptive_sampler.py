'''
Rethinking Video samplers: A good sampler is all you need for efficient action recognition

'''

import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mmengine.model.weight_init import trunc_normal_
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .attention import cos_pairwise


# TODO 可能需要考虑位置编码


class FrameSoftAttention(nn.Module):
    """
    只对输入帧计算attention权重后融合，输入帧
    """
    def __init__(self, input_dim, inner_dim, output_dim, heads = 1, dropout = 0., embed_frame=False, use_cos=False):
        super().__init__()
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.embed_frame = embed_frame
        self.heads = heads
        assert inner_dim % heads == 0
        self.use_cos = use_cos
        if not use_cos:
            self.scale = (inner_dim // heads) ** -0.5

        self.to_q = nn.Linear(input_dim, inner_dim, bias=False)
        trunc_normal_(self.to_q.weight, std=.02)
        self.to_k = nn.Linear(input_dim, inner_dim, bias=False)
        trunc_normal_(self.to_k.weight, std=.02)
        if embed_frame:
            self.to_v = nn.Linear(input_dim, inner_dim, bias=False)
            trunc_normal_(self.to_v.weight, std=.02)

        assert heads == 1, "多头注意力对于头的融合不是很好处理，先不处理了"
        # self.to_out = nn.Sequential(
        #     nn.Linear(inner_dim, output_dim),
        #     nn.Dropout(dropout)
        # ) if (heads != 1 and embed_frame) else nn.Identity()
        
        
        
    def forward(self, q, k, v):
        # input format: b n d
        q, k = self.to_q(q), self.to_k(k)
        if self.embed_frame:
            v = self.to_v(v)

        if self.use_cos:
            dots = cos_pairwise(q.unsqueeze(1), k.unsqueeze(1)).squeeze(1)
        else:
            dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        # k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        # v = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        # dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)
        # plt.imshow(attn.squeeze(1).detach().numpy())
        # print(np.argmax(attn.squeeze(1).detach().numpy(), axis=-1))
        # out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        # out =  self.to_out(out)
        out = einsum('b i j, b j d -> b i d', attn, v)
        
        return out

class AttentionFrameSampler(nn.Module):
    def __init__(self, output_num_frames=8, input_dim=3*224*224, inner_dim=3*56*56,  spatial_pool_size=(4, 4), output_dim=3*224*224, num_heads=1, attn_drop=0., embed_frame=False) -> None:
        super().__init__()
        self.output_num_frames = output_num_frames
        self.spatial_downsample = nn.MaxPool2d(kernel_size=spatial_pool_size, stride=spatial_pool_size) # 这里可以改成dwconv
        self.downsample_ratio = spatial_pool_size[0] * spatial_pool_size[1]
        self.attn = FrameSoftAttention(input_dim=input_dim//self.downsample_ratio, inner_dim=inner_dim, output_dim=output_dim, heads=num_heads, dropout=attn_drop, embed_frame=embed_frame)

    def forward(self, x):
        # 原来是8x32x1 我们通过samper将 （8*16)x2x1 转换为 8x32x1，即每个中间帧attention周围16帧
        # x (N, 8*16, 3, 224, 224)
        N, T, C, H, W = x.shape
        assert T % self.output_num_frames == 0
        v = x.view(N * self.output_num_frames, T//self.output_num_frames, C * H * W) # value直接是原视频帧，没有下采样
        x = self.spatial_downsample(x.view(N*T, C, H, W))
        x = x.view(N * self.output_num_frames, T//self.output_num_frames, C * H * W // self.downsample_ratio) # (N * 8, 16, 3 * 56 * 56)
        q_ind = T//(2*self.output_num_frames)
        x = self.attn(q=x[:, q_ind:q_ind+1, :], k=x, v=v) # 用中间的第9帧attend周围16帧，TODO 其实这里我感觉取第1帧效果也应该差不多
        return x.view(N, self.output_num_frames, C, H, W) 


# class ConvFrameSampler(nn.Module):
#     def __init__(self, input_num_frames=32, output_num_frames=8, input_dim=3, embed_dims=96, num_heads=1, attn_drop=0.) -> None:
#         self.conv = nn.Conv1d(input_dims, embed_dims )





class PyramidAttentionFrameSampler(nn.Module):
    # 层级式的聚合，可以用卷积实现也可以用attention实现
    pass

class ConvPatchSampler(nn.Module):
    pass

class AttentionPatchSampler(nn.Module):
    pass

class DeformableConvPatchSampler(nn.Module):
    pass

class DeformableAttentionPatchSampler(nn.Module):
    pass

class MutiScalePyramidConvPatchSampler(nn.Module):
    pass

class MutiScalePyramidAttentionPatchSampler(nn.Module):
    pass

class MutiScalePoolingConvPatchSampler(nn.Module):
    pass

class MutiScalePoolingAttentionPatchSampler(nn.Module):
    pass

# 实验计划：先在freeze的TSN, TSM, SlowOnly和divided-Timesformer和joint-timesformer上看有没有效果