# Copyright (c) OpenMMLab. All rights reserved.
from functools import lru_cache, reduce
from operator import mul
from typing import Dict, List, Optional, Sequence, Tuple, Union

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner.checkpoint import _load_checkpoint

from mmaction.registry import MODELS
from ..common import TemporalConv, TopkRouting, QKVLinear, KVGather
from .swin import window_partition, window_reverse, get_window_size, compute_mask, PatchEmbed3D, PatchMerging, Mlp
from .swin_refiner import WindowAttention2D
from .swin_ifa import InterFrameWindowAttention2D
from .swin_tps import TemporalShift
# from ..utils import print_cuda_memory

'''
NOTE 这里实现和隔壁ifa不一样，隔壁是尝试替换attention，这里是直接加模块
'''


class BiLevelRoutingAttention3D(nn.Module):
    pass

class ShiftAdapter(nn.Module):
    """
    为了简便阉割了原版好多选项
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights 
    """

    def __init__(self, dim, num_frames, num_heads, shift=False, shift_type='tsm'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.shift = shift
        self.shift_type = shift_type


        ################ shift ########################
        if self.shift:
            if self.shift_type == 'tsm':
                self.shift_op = TemporalShift(8)
            elif self.shift_type == 'tc':
                self.shift_op = TemporalConv(dim, n_segment=num_frames)

        self.fc = nn.Linear(dim, dim)
        self.act = nn.GELU()
        

    def forward(self, x, ret_attn_mask=False):
        """
        x: NDHWC tensor

        Return:
            NDHWC tensor
        """

        N, D, H, W, C = x.size()

        if self.shift:
            if self.shift_type == 'tsm':
                x = x.view(N*D, H*W, self.num_heads, C //
                           self.num_heads).permute(0, 2, 1, 3)
                x = self.shift_op(x, N, D)
                x = x.permute(0, 2, 1, 3).reshape(N, D, H, W, C)
            else:
                raise NotImplementedError(
                    f"Not support shift_type: {self.shift_type}")

        out = self.act(self.fc(x))
        
        return out.view(N, D, H, W, C)
        

class InterFrameBiLevelRoutingAttention(nn.Module):
    """
    为了简便阉割了原版好多选项
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights 
    """

    def __init__(self, dim, num_frames, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, side_dwconv_cfg=dict(type='identity'), shift=False, shift_type='tsm'):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        self.shift = shift
        self.shift_type = shift_type
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5

        ################ shift ########################
        if self.shift:
            if self.shift_type == 'tsm':
                self.shift_op = TemporalShift(8)
            elif self.shift_type == 'tc':
                self.shift_op = TemporalConv(dim, n_segment=num_frames)

        ################ side_dwconv (i.e. LCE in ShuntedTransformer)###########
        if side_dwconv_cfg.get('type', '') == 'spatial_only':
            self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv_cfg.get(
                'k'), stride=1, padding=side_dwconv_cfg.get('k')//2, groups=dim)
        elif side_dwconv_cfg.get('type', '') == 'temporal_only':
            # self.lepe =
            pass
        elif side_dwconv_cfg.get('type', '') == 'joint_ST':
            pass
        elif side_dwconv_cfg.get('type', '') == 'divided_ST':
            pass
        elif side_dwconv_cfg.get('type', '') == 'identity':
            pass  # lambda x: torch.zeros_like(x)
        elif side_dwconv_cfg.get('type'):
            raise NotImplementedError(
                f"Not support side_dwconv type:{side_dwconv_cfg.get('type')}")
        ################ global routing setting #################
        self.topk = topk
        # router
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=False,
                                  param_routing=False)

        # hard non-differentiable routing
        self.kv_gather = KVGather(mul_weight='none')

        # qkv mapping (shared by both global routing and local attention)
        self.qkv = QKVLinear(self.dim, self.qk_dim)
        self.wo = nn.Linear(dim, dim)

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(
                self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(
                self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity':  # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v
            raise NotImplementedError(
                'fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(
                f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

    def forward(self, x, ret_attn_mask=False):
        """
        x: NDHWC tensor

        Return:
            NDHWC tensor
        """

        N, D, H, W, C = x.size()
        assert H % self.n_win == 0 and W % self.n_win == 0

        if self.shift:
            if self.shift_type == 'tsm':
                x = x.view(N*D, H*W, self.num_heads, C //
                           self.num_heads).permute(0, 2, 1, 3)
                x = self.shift_op(x, N, D)
                x = x.permute(0, 2, 1, 3).reshape(N, D, H, W, C)
            else:
                raise NotImplementedError(
                    f"Not support shift_type: {self.shift_type}")

        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n t (j h) (i w) c -> n t (j i) h w c",
                      j=self.n_win, i=self.n_win)

        ################# qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather

        q, kv = self.qkv(x)

        ################## side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        # lepe = self.lepe(rearrange( # TODO 放弃使用lepe
        #     kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
        # lepe = rearrange(
        #     lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        # 帧间attention，向左shift一个时间位
        shift_kv = kv.clone()
        shift_kv[:, :-1, :, :, :, :] = kv[:, 1:, :, :, :, :]
        kv = shift_kv

        q, kv = q.view(N*D, self.n_win * self.n_win, H // self.n_win, W // self.n_win,
                       C), kv.view(N*D, self.n_win * self.n_win, H // self.n_win, W // self.n_win, 2*C)

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(
            kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean(
            [2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ############ gather q dependent k/v #################

        # both are (n, p^2, topk) tensors
        r_weight, r_idx = self.router(q_win, k_win)

        # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)
        k_pix_sel, v_pix_sel = kv_pix_sel.split(
            [self.qk_dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)

        ######### do attention as normal ####################
        # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        k_pix_sel = rearrange(
            k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads)
        # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        v_pix_sel = rearrange(
            v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads)
        # to BMLC tensor (n*p^2, m, w^2, c_qk//m)
        q_pix = rearrange(
            q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads)

        # param-free multihead attention
        # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = (q_pix * self.scale) @ k_pix_sel
        attn_weight = self.attn_act(attn_weight)
        # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = attn_weight @ v_pix_sel
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H//self.n_win, w=W//self.n_win)

        # out = out + lepe TODO 编码交给shift

        # output linear
        out = self.wo(out)

        if ret_attn_mask:
            return out.view(N, D, H, W, C), r_weight, r_idx, attn_weight
        else:
            return out.view(N, D, H, W, C)


class SwinTransformerBlock3D(BaseModule):
    """Swin Transformer Block.

    Args:
        embed_dims (int): Number of feature channels.
        num_heads (int): Number of attention heads.
        window_size (Sequence[int]): Window size. Defaults to ``(8, 7, 7)``.
        shift_size (Sequence[int]): Shift size for SW-MSA or W-MSA.
            Defaults to ``(0, 0, 0)``.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            Defaults to 4.0.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop (float): Dropout rate. Defaults to 0.0.
        attn_drop (float): Attention dropout rate. Defaults to 0.0.
        drop_path (float): Stochastic depth rate. Defaults to 0.1.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for norm layer.
            Defaults to ``dict(type='LN')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Defaults to False.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 num_frames: int,
                 height: int,
                 width: int,
                 window_size: Sequence[int] = (8, 7, 7),
                 shift_size: Sequence[int] = (0, 0, 0),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.1,
                 act_cfg: Dict = dict(type='GELU'),
                 norm_cfg: Dict = dict(type='LN'),
                 with_cp: bool = False,
                 shift=False,
                 shift_type='tsm',
                 use_ifa=False,
                 ifa_cfg=dict(use_motion=False),
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp
        self.shift = shift
        self.shift_type = shift_type
        self.use_ifa = use_ifa
        self.ifa_cfg = copy.deepcopy(ifa_cfg)

        assert 0 <= self.shift_size[0] < self.window_size[
            0], 'shift_size[0] must in [0, window_size[0])'
        assert 0 <= self.shift_size[1] < self.window_size[
            1], 'shift_size[1] must in [0, window_size[0])'
        assert 0 <= self.shift_size[2] < self.window_size[
            2], 'shift_size[2] must in [0, window_size[0])'

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        _attn_cfg = {
            'embed_dims': embed_dims,
            # NOTE 可能要考虑用get_window_size((D, H, W), self.window_size, self.shift_size)
            'num_spatial_windows': (height // window_size[1]) * (width // window_size[2]),
            'num_heads': num_heads,
            'num_frames': num_frames,
            'window_size': window_size,
            'qkv_bias': qkv_bias,
            'qk_scale': qk_scale,
            'attn_drop': attn_drop,
            'proj_drop': drop,
            'shift': self.shift,
            'shift_type': self.shift_type
        }

        self.attn = WindowAttention2D(**_attn_cfg)

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        _mlp_cfg = {
            'in_features': embed_dims,
            'hidden_features': int(embed_dims * mlp_ratio),
            'act_cfg': act_cfg,
            'drop': drop
        }
        self.mlp = Mlp(**_mlp_cfg)

        if use_ifa:
            self.norm_ifa = build_norm_layer(norm_cfg, embed_dims)[1]
            if self.ifa_cfg.get('type', '') == 'bra':
                assert self.ifa_cfg.get('topk', 1) > 0
                self.attn_bra = InterFrameBiLevelRoutingAttention(dim=embed_dims, num_frames=num_frames,
                                                                  num_heads=num_heads, n_win=self.ifa_cfg.get('n_win', 7), topk=self.ifa_cfg.get('topk', 1),
                                                                  shift=self.shift, shift_type=self.shift_type)
            elif self.ifa_cfg.get('type', '') == 'wa':
                _ifa_attn_cfg = copy.deepcopy(_attn_cfg)
                # _ifa_attn_cfg['shift'] = False shift他妈的
                self.attn_wa = InterFrameWindowAttention2D(**_ifa_attn_cfg)
            elif self.ifa_cfg.get('type', '') == 'ori': # 消融实验，单纯加层数
                self.attn_ori = WindowAttention2D(**_attn_cfg) 
            elif self.ifa_cfg.get('type', '') == 'shift':
                self.attn_shift = ShiftAdapter(dim=embed_dims, num_frames=num_frames, num_heads=num_heads, shift=shift, shift_type=shift_type)
            elif self.ifa_cfg.get('type'):
                raise NotImplementedError(
                    f"Not support ifa type: {self.ifa_cfg.get('type')}")

            if self.ifa_cfg.get('use_motion', False) and self.ifa_cfg.get('fuse_type', '') == 'concat':
                self.fuse_fc = nn.Linear(embed_dims*2, embed_dims)

    def get_loc_map(self, shape):  # TODO 对于不同大小输入可能需要重新生成
        tenHorizontal = torch.linspace(-1.0, 1.0, shape[2]).view(
            1, 1, shape[2], 1).expand(shape[0], shape[1], -1, -1)
        tenVertical = torch.linspace(-1.0, 1.0, shape[1]).view(
            1, shape[1], 1, 1).expand(shape[0], -1, shape[2], -1)
        return torch.cat([tenHorizontal, tenVertical], -1).cuda()

    def forward_part1(self, x: torch.Tensor,
                      mask_matrix: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size,
                                                  self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x,
                                     window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=attn_mask, batch_size=B, frame_len=D)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C, )))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp,
                                   Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_bra(self, x: torch.Tensor) -> torch.Tensor:
        # print('use_bra')
        B, D, H, W, C = x.shape
        x = self.norm_ifa(x)

        if self.ifa_cfg.get('use_motion', False):
            # loc_map = self.get_loc_map([B*D, H, W]).view(B, D, H, W, 2)
            raise NotImplementedError
        else:
            x = self.attn_bra(x)

        return x

    def forward_shift(self, x: torch.Tensor) -> torch.Tensor:
        # print('use_shift')
        B, D, H, W, C = x.shape
        x = self.norm_ifa(x)

        if self.ifa_cfg.get('use_motion', False):
            # loc_map = self.get_loc_map([B*D, H, W]).view(B, D, H, W, 2)
            raise NotImplementedError
        else:
            x = self.attn_shift(x)

        return x
    
    def forward_wa(self, x: torch.Tensor, mask_matrix: torch.Tensor) -> torch.Tensor:
        # print('use_wa')
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size,
                                                  self.shift_size)

        x = self.norm_ifa(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x,
                                     window_size)  # B*nW, Wd*Wh*Ww, C

        if self.ifa_cfg.get('use_motion', False):
            loc_map = self.get_loc_map([B*D, H, W]).view(B, D, H, W, 2)
            loc_map = F.pad(loc_map, (0, 0, pad_l, pad_r,
                            pad_t, pad_b, pad_d0, pad_d1))
            if any(i > 0 for i in shift_size):
                loc_map = torch.roll(
                    loc_map,
                    shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                    dims=(1, 2, 3))
            # partition windows
            loc_map = window_partition(
                loc_map, window_size)  # B*nW, Wd*Wh*Ww, C

            # W-MSA/SW-MSA
            attn_windows, x_motion = self.attn_wa(
                x_windows, mask=attn_mask, batch_size=B, frame_len=D, loc_map=loc_map)  # B*nW, Wd*Wh*Ww, C
            # print('refine motion')
            # merge windows
            x_motion = x_motion.view(-1, *(window_size + (C, )))
            shifted_x_motion = window_reverse(x_motion, window_size, B, Dp, Hp,
                                              Wp)  # B D' H' W' C
            # reverse cyclic shift
            if any(i > 0 for i in shift_size):
                x_motion = torch.roll(
                    shifted_x_motion,
                    shifts=(shift_size[0], shift_size[1], shift_size[2]),
                    dims=(1, 2, 3))
            else:
                x_motion = shifted_x_motion

            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x_motion = x_motion[:, :D, :H, :W, :].contiguous()
        else:
            # W-MSA/SW-MSA
            attn_windows = self.attn_wa(
                x_windows, mask=attn_mask, batch_size=B, frame_len=D)  # B*nW, Wd*Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C, )))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp,
                                   Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        if self.use_ifa and self.ifa_cfg.get('use_motion', False):
            # print('fuse motion and apperance')
            # fuse appearance information (x) and motion information (x_motion)
            fuse_type = self.ifa_cfg.get('fuse_type')
            if fuse_type == 'add':
                x = x + x_motion
            elif fuse_type == 'mul':
                x = x + x_motion * x
            elif fuse_type == 'concat':
                x = self.fuse_fc(torch.cat([x, x_motion], dim=-1))
        return x

    def forward_ori(self, x: torch.Tensor,
                      mask_matrix: torch.Tensor) -> torch.Tensor:
        # 不使用帧间注意力，单纯加层数，和forward_part1完全一样
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size,
                                                  self.shift_size)

        x = self.norm_ifa(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x,
                                     window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn_ori(
            x_windows, mask=attn_mask, batch_size=B, frame_len=D)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C, )))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp,
                                   Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x
    
    def forward_part2(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function part2."""
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x: torch.Tensor,
                mask_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features of shape :math:`(B, D, H, W, C)`.
            mask_matrix (torch.Tensor): Attention mask for cyclic shift.
        """

        shortcut = x
        if self.with_cp:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_ifa:
            if self.ifa_cfg.get('type') == 'bra':
                x = x + self.drop_path(self.forward_bra(x))
            elif self.ifa_cfg.get('type') == 'wa':
                x = x + self.drop_path(self.forward_wa(x, mask_matrix))
            elif self.ifa_cfg.get('type') == 'ori':
                x = x + self.drop_path(self.forward_ori(x, mask_matrix))
            elif self.ifa_cfg.get('type') == 'shift':
                x = x + self.drop_path(self.forward_shift(x))
            elif self.ifa_cfg.get('type'):
                raise NotImplementedError(
                    f"Not support ifa type: {self.ifa_cfg.get('type')}")

        if self.with_cp:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class BasicLayer(BaseModule):
    """A basic Swin Transformer layer for one stage.

    Args:
        embed_dims (int): Number of feature channels.
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (Sequence[int]): Local window size.
            Defaults to ``(8, 7, 7)``.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop (float): Dropout rate. Defaults to 0.0.
        attn_drop (float): Attention dropout rate. Defaults to 0.0.
        drop_paths (float or Sequence[float]): Stochastic depth rates.
            Defaults to 0.0.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict, optional): Config dict for norm layer.
            Defaults to ``dict(type='LN')``.
        downsample (:class:`PatchMerging`, optional): Downsample layer
            at the end of the layer. Defaults to None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will
            save some memory while slowing down the training speed.
            Defaults to False.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 depth: int,
                 num_heads: int,
                 num_frames: int,
                 height: int,
                 width: int,
                 window_size: Sequence[int] = (1, 7, 7),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_paths: Union[float, Sequence[float]] = 0.,
                 act_cfg: Dict = dict(type='GELU'),
                 norm_cfg: Dict = dict(type='LN'),
                 downsample: Optional[PatchMerging] = None,
                 with_cp: bool = False,
                 shift_type='tsm',
                 shift=True,
                 ifa_cfg=dict(use_motion=False),
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.with_cp = with_cp
        self.shift_type = shift_type

        if not isinstance(drop_paths, Sequence):
            drop_paths = [drop_paths] * depth

        # build blocks
        self.blocks = ModuleList()
        for i in range(depth):
            _block_cfg = {
                'embed_dims': embed_dims,
                'num_heads': num_heads,
                'num_frames': num_frames,
                'height': height,
                'width': width,
                'window_size': window_size,
                'shift_size': (0, 0, 0) if (i % 2 == 0) else self.shift_size,
                'mlp_ratio': mlp_ratio,
                'qkv_bias': qkv_bias,
                'qk_scale': qk_scale,
                'drop': drop,
                'attn_drop': attn_drop,
                'drop_path': drop_paths[i],
                'act_cfg': act_cfg,
                'norm_cfg': norm_cfg,
                'with_cp': with_cp,
                'shift': shift,
                'shift_type': self.shift_type
            }

            if i in ifa_cfg.get('loc'):  # NOTE 在初始化参数里设置要添加ifa的位置
                _block_cfg['use_ifa'] = True
                _block_cfg['ifa_cfg'] = copy.deepcopy(ifa_cfg)
            else:
                _block_cfg['use_ifa'] = False

            block = SwinTransformerBlock3D(**_block_cfg)
            self.blocks.append(block)

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(
                embed_dims=embed_dims, norm_cfg=norm_cfg)

    def forward(self,
                x: torch.Tensor,
                do_downsample: bool = True) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): Input feature maps of shape
                :math:`(B, C, D, H, W)`.
            do_downsample (bool): Whether to downsample the output of
                the current layer. Defaults to True.
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size,
                                                  self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
            # print_cuda_memory('经过一个block')
        if self.downsample is not None and do_downsample:
            x = self.downsample(x)
        # print_cuda_memory('经过一个basic layer')
        return x

    @property
    def out_embed_dims(self):
        if self.downsample is not None:
            return self.downsample.out_embed_dims
        else:
            return self.embed_dims


@MODELS.register_module()
class SwinTransformer2D_BRA(BaseModule):
    """Video Swin Transformer backbone.

    A pytorch implement of: `Video Swin Transformer
    <https://arxiv.org/abs/2106.13230>`_

    Args:
        arch (str or dict): Video Swin Transformer architecture. If use string,
            choose from 'tiny', 'small', 'base' and 'large'. If use dict, it
            should have below keys:
            - **embed_dims** (int): The dimensions of embedding.
            - **depths** (Sequence[int]): The number of blocks in each stage.
            - **num_heads** (Sequence[int]): The number of heads in attention
            modules of each stage.
        pretrained (str, optional): Name of pretrained model.
            Defaults to None.
        pretrained2d (bool): Whether to load pretrained 2D model.
            Defaults to True.
        patch_size (int or Sequence(int)): Patch size.
            Defaults to ``(2, 4, 4)``.
        in_channels (int): Number of input image channels. Defaults to 3.
        window_size (Sequence[int]): Window size. Defaults to ``(8, 7, 7)``.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop_rate (float): Dropout rate. Defaults to 0.0.
        attn_drop_rate (float): Attention dropout rate. Defaults to 0.0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for norm layer.
            Defaults to ``dict(type='LN')``.
        patch_norm (bool): If True, add normalization after patch embedding.
            Defaults to True.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        with_cp (bool): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Defaults to False.
        out_indices (Sequence[int]): Indices of output feature.
            Defaults to ``(3, )``.
        out_after_downsample (bool): Whether to output the feature map of a
            stage after the following downsample layer. Defaults to False.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ]``.
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 96,
                         'depths': [2, 2, 6, 2],
                         'num_heads': [3, 6, 12, 24]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 96,
                         'depths': [2, 2, 18, 2],
                         'num_heads': [3, 6, 12, 24]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 128,
                         'depths': [2, 2, 18, 2],
                         'num_heads': [4, 8, 16, 32]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 192,
                         'depths': [2, 2, 18, 2],
                         'num_heads': [6, 12, 24, 48]}),
    }  # yapf: disable

    def __init__(
        self,
        arch: Union[str, Dict],
        pretrained: Optional[str] = None,
        # transfer2ifa: bool = False, # 借助原来swin的权重给ifa (目前只支持给wa)
        pretrained2d: bool = True,
        patch_size: Union[int, Sequence[int]] = (2, 4, 4),
        input_size=(32, 224, 224),  # T H W
        in_channels: int = 3,
        window_size: Sequence[int] = (1, 7, 7),
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        act_cfg: Dict = dict(type='GELU'),
        norm_cfg: Dict = dict(type='LN'),
        patch_norm: bool = True,
        frozen_stages: int = -1,
        with_cp: bool = False,
        shift_type='tsm',
        shift=False,
        ifa_types=['bra', 'bra', 'bra', 'wa'],
        ifa_locs=[[0], [0], [0, 3], [0]],
        bra_topks=[1, 4, 16, -1],  # 窗口长宽各stage依次为 8 4 2 1, -1表示不使用bra
        ifa_cfg=dict(use_motion=False, n_win=7),
        out_indices: Sequence[int] = (3, ),
        out_after_downsample: bool = False,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        # self.transfer2ifa = transfer2ifa

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'num_heads'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        assert len(self.depths) == len(self.num_heads)
        self.num_layers = len(self.depths)
        assert 1 <= self.num_layers <= 4
        self.out_indices = out_indices
        assert max(out_indices) < self.num_layers
        self.out_after_downsample = out_after_downsample
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        _patch_cfg = {
            'patch_size': patch_size,
            'in_channels': in_channels,
            'embed_dims': self.embed_dims,
            'norm_cfg': norm_cfg if patch_norm else None,
            'conv_cfg': dict(type='Conv3d')
        }
        self.patch_embed = PatchEmbed3D(**_patch_cfg)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.num_frames = input_size[0] // patch_size[0]
        height = input_size[1] // patch_size[1]
        width = input_size[2] // patch_size[2]

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        # build layers
        self.layers = ModuleList()
        embed_dims = [self.embed_dims]
        for i, (depth, num_heads) in \
                enumerate(zip(self.depths, self.num_heads)):
            downsample = PatchMerging if i < self.num_layers - 1 else None
            stage_ifa_cfg = copy.deepcopy(ifa_cfg)
            stage_ifa_cfg['topk'] = bra_topks[i]
            stage_ifa_cfg['type'] = ifa_types[i]
            stage_ifa_cfg['loc'] = ifa_locs[i]
            _layer_cfg = {
                'embed_dims': embed_dims[-1],
                'depth': depth,
                'num_heads': num_heads,
                'num_frames': self.num_frames,
                'height': height,
                'width': width,
                'window_size': window_size,
                'mlp_ratio': mlp_ratio,
                'qkv_bias': qkv_bias,
                'qk_scale': qk_scale,
                'drop': drop_rate,
                'attn_drop': attn_drop_rate,
                'drop_paths': dpr[:depth],
                'act_cfg': act_cfg,
                'norm_cfg': norm_cfg,
                'downsample': downsample,
                'with_cp': with_cp,
                'shift_type': shift_type,
                'shift': shift,
                'ifa_cfg': stage_ifa_cfg
            }

            layer = BasicLayer(**_layer_cfg)
            self.layers.append(layer)

            dpr = dpr[depth:]
            embed_dims.append(layer.out_embed_dims)

            height = height // 2
            width = width // 2

        if self.out_after_downsample:
            self.num_features = embed_dims[1:]
        else:
            self.num_features = embed_dims[:-1]

        for i in out_indices:
            if norm_cfg is not None:
                norm_layer = build_norm_layer(norm_cfg,
                                              self.num_features[i])[1]
            else:
                norm_layer = nn.Identity()

            self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger: MMLogger) -> None:
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model, the weight
        of swin2d models should be inflated to fit in the shapes of the
        3d counterpart.

        Args:
            logger (MMLogger): The logger used to print debugging information.
        """
        checkpoint = _load_checkpoint(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [
            k for k in state_dict.keys() if 'relative_position_index' in k
        ]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if 'attn_mask' in k]
        for k in attn_mask_keys:
            del state_dict[k]
        state_dict['patch_embed.proj.weight'] = \
            state_dict['patch_embed.proj.weight'].unsqueeze(2).\
            repeat(1, 1, self.patch_size[0], 1, 1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [
            k for k in state_dict.keys() if 'relative_position_bias_table' in k
        ]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            # wd = self.window_size[0]
            wd = self.num_frames  # 16  NOTE
            if nH1 != nH2:
                logger.warning(f'Error in loading {k}, passing')
            else:
                if L1 != L2:
                    S1 = int(L1**0.5)
                    relative_position_bias_table_pretrained_resized = \
                        torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.permute(
                                1, 0).view(1, nH1, S1, S1),
                            size=(2 * self.window_size[1] - 1,
                                  2 * self.window_size[2] - 1),
                            mode='bicubic')
                    relative_position_bias_table_pretrained = \
                        relative_position_bias_table_pretrained_resized. \
                        view(nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(
                2 * wd - 1, 1)

        # In the original swin2d checkpoint, the last layer of the
        # backbone is the norm layer, and the original attribute
        # name is `norm`. We changed it to `norm3` which means it
        # is the last norm layer of stage 4.
        if hasattr(self, 'norm3'):
            state_dict['norm3.weight'] = state_dict['norm.weight']
            state_dict['norm3.bias'] = state_dict['norm.bias']
            del state_dict['norm.weight']
            del state_dict['norm.bias']

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)

    def init_weights(self) -> None:
        """Initialize the weights in backbone."""
        for m in self.modules():  # noqa lxh: 提前初始化保险一哈
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if self.pretrained2d:
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            # Inflate 2D model into 3D model.
            self.inflate_weights(logger)
        else:
            if self.pretrained:
                self.init_cfg = dict(
                    type='Pretrained', checkpoint=self.pretrained)
            super().init_weights()

    def forward(self, x: torch.Tensor) -> \
            Union[Tuple[torch.Tensor], torch.Tensor]:
        """Forward function for Swin3d Transformer."""
        # print_cuda_memory('开始')
        x = self.patch_embed(x)
        # print_cuda_memory('经过patch embed')
        x = self.pos_drop(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x.contiguous(), do_downsample=self.out_after_downsample)
            # print_cuda_memory(f'经过第{i}层')
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = rearrange(out, 'b d h w c -> b c d h w').contiguous()
                outs.append(out)

            if layer.downsample is not None and not self.out_after_downsample:
                x = layer.downsample(x)
                # print_cuda_memory(f'经过第{i}层的downsample')
            if i < self.num_layers - 1:
                x = rearrange(x, 'b d h w c -> b c d h w')

        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep layers frozen."""
        super(SwinTransformer2D_BRA, self).train(mode)
        self._freeze_stages()
