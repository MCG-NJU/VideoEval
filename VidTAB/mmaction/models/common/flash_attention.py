from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
try:
    import xformers.ops as xops
except:
    print("xformers isn't installed")

try:
    from flash_attn import flash_attn_qkvpacked_func
except:
    print("flash_attn isn't installed")
    


class FlashAttention_pytorch(nn.Module):
    r"""
    对齐pytorch nn.MultiHeadAttention表现
    """
    def __init__(self,
                 embed_dim,
                 num_heads=8,
                 qkvo_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_type='flash_v2'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))

        if qkvo_bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.in_proj_bias = None

        if attn_type == 'flash_v2':
            self.attn_drop = attn_drop
        else:
            self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj  = nn.Linear(all_head_dim, embed_dim, bias=qkvo_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_type = attn_type

    def forward(self, x, is_causal=False, need_weights=False, key_padding_mask=None, attn_mask=None):

        if need_weights or attn_mask is not None:
            raise NotImplementedError("Please use nn.MultiheadAttention.")
        if key_padding_mask:
            raise NotImplementedError("Wait for me to implement it in the future.")

        N, B, C = x.shape

        qkv = F.linear(input=x, weight=self.in_proj_weight, bias=self.in_proj_bias)

        if self.attn_type == 'flash_v2':
            qkv = qkv.permute(1, 0, 2).reshape(B, N, 3, self.num_heads, -1)
            x = flash_attn_qkvpacked_func(qkv, dropout_p=self.attn_drop, softmax_scale=self.scale, causal=is_causal).reshape(B, N, -1).permute(1, 0, 2)
        else:
            raise NotImplementedError(self.attn_type)
            

        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x

class FlashAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None,
                 attn_type='flash_v2'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if attn_type == 'flash_v2':
            self.attn_drop = attn_drop
        else:
            self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_type = attn_type

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)

        if self.attn_type == 'flash_v2':
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1)
            x = flash_attn_qkvpacked_func(qkv, dropout_p=self.attn_drop, softmax_scale=self.scale, causal=False).reshape(B, N, -1)
        else:
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[
                2]  # make torchscript happy (cannot use tensor as tuple)
            # B num_heads N head_dim
            if self.attn_type == 'xformer': # flash v1
                x = xops.memory_efficient_attention(q, k, v).transpose(1, 2).reshape(B, N, -1)
            else: # origin
                q = q * self.scale
                attn = (q @ k.transpose(-2, -1))

                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x