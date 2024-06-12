# Copyright (c) OpenMMLab. All rights reserved.
from functools import lru_cache, reduce
from operator import mul
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
import copy

from mmaction.registry import MODELS
from .swin import window_partition, window_reverse, get_window_size, compute_mask, PatchEmbed3D, PatchMerging
# from ..utils import print_cuda_memory

     
class LinearAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, skip_connect=True, scale=1.):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.scale = scale
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs * self.scale
        else:
            x = xs * self.scale
        return x
    
    
class HeadReallocation(nn.Module):
    def __init__(self, num_frames, num_theads=1):
        super(HeadReallocation, self).__init__()
        self.num_frames = num_frames
        self.num_theads = num_theads

        logger = MMLogger.get_current_instance()
        logger.info( f'HeadReallocation, num_frames: {self.num_frames}, num_theads: {self.num_theads}')
    
    def shift_pos(self, x_pos):
        # x_pos is (Frames, nH, Wd*Wh*Ww, Wd*Wh*Ww)
        if self.num_theads < 1: # 小于1不shift
            return x_pos
        
        out = x_pos.clone()

        if self.num_theads == 8:
            out[1:, 0:2, :, :] = x_pos[:-1, 0:2, :, :]  # shift left
            out[:-1, 2:4, :, :] = x_pos[1:, 2:4, :, :]  # shift right
            out[2:, 4:6, :, :] = x_pos[:-2, 4:6, :, :]  # shift left2
            out[:-2, 6:8, :, :] = x_pos[2:, 6:8, :, :]  # shift right2
        elif self.num_theads == 16:
            out[1:, 0:4, :, :] = x_pos[:-1, 0:4, :, :]  # shift left
            out[:-1, 4:8, :, :] = x_pos[1:, 4:8, :, :]  # shift right
            out[2:, 8:12, :, :] = x_pos[:-2, 8:12, :, :]  # shift left2
            out[:-2, 12:16, :, :] = x_pos[2:, 12:16, :, :]  # shift right2
        else:
            out[1:, 0, :, :] = x_pos[:-1, 0, :, :]  # shift left
            if self.num_theads > 1:
                out[:-1, 1, :, :] = x_pos[1:, 1, :, :]  # shift right
                if self.num_theads > 2:
                    out[2:, 2, :, :] = x_pos[:-2, 2, :, :]  # shift left2
                    if self.num_theads > 3:
                        out[:-2, 3, :, :] = x_pos[2:, 3, :, :]  # shift right2

        return out


    def forward(self, x):
        # x is (BT, num_heads, N, C//num_heads)
        if self.num_theads == 0:
            return x

        bt, num_heads, n, c_ = x.shape
        feat = x

        # 不分head shift
        feat = feat.view(bt // self.num_frames, self.num_frames, num_heads, n, c_)
        out = feat.clone() 
        # print(out.shape) # e([16, 8, 12, 197, 64])
        if self.num_theads < 1:
            out[:, 1:, 0, :, :c_//2] = feat[:, :-1, 0, :, :c_//2]  # shift left
            out[:, :-1, 0, :, c_//2:] = feat[:, 1:, 0, :, c_//2:]  # shift right
        elif self.num_theads == 8:
            out[:, 1:, 0:2, :, :] = feat[:, :-1, 0:2, :, :]  # shift left
            out[:, :-1, 2:4, :, :] = feat[:, 1:, 2:4, :, :]  # shift right
            out[:, 2:, 4:6, :, :] = feat[:, :-2, 4:6, :, :]  # shift left2
            out[:, :-2, 6:8, :, :] = feat[:, 2:, 6:8, :, :]  # shift right2
        elif self.num_theads == 16:
            out[:, 1:, 0:4, :, :] = feat[:, :-1, 0:4, :, :]  # shift left
            out[:, :-1, 4:8, :, :] = feat[:, 1:, 4:8, :, :]  # shift right
            out[:, 2:, 8:12, :, :] = feat[:, :-2, 8:12, :, :]  # shift left2
            out[:, :-2, 12:16, :, :] = feat[:, 2:, 12:16, :, :]  # shift right2
        else:
            out[:, 1:, 0, :, :] = feat[:, :-1, 0, :, :]  # shift left
            if self.num_theads > 1:
                out[:, :-1, 1, :, :] = feat[:, 1:, 1, :, :]  # shift right
                if self.num_theads > 2:
                    out[:, 2:, 2, :, :] = feat[:, :-2, 2, :, :]  # shift left2
                    if self.num_theads > 3:
                        out[:, :-2, 3, :, :] = feat[:, 2:, 3, :, :]  # shift right2

        out = out.view(bt, num_heads, n, c_)

        return out


class WindowAttention2D(BaseModule):
    """Window based multi-head self attention (W-MSA) module with relative
    position bias. It supports both of shifted and non-shifted window.

    Args:
        embed_dims (int): Number of input channels.
        window_size (Sequence[int]): The temporal length, height and
            width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool):  If True, add a learnable bias to query,
            key, value. Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop (float): Dropout ratio of attention weight. Defaults to 0.0.
        proj_drop (float): Dropout ratio of output. Defaults to 0.0.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_spatial_windows: int,  # TODO
                 num_heads: int,
                 num_frames: int,  # TODO
                 window_size: Sequence[int],
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 init_cfg: Optional[Dict] = None,
                 zero_cfg=dict()) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_spatial_windows = num_spatial_windows

        self.zero_cfg = zero_cfg
        # NOTE 2d情形一般window_size[0] = 1
        window_size = (num_frames, window_size[1], window_size[2])
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        # # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) *
                        (2 * window_size[2] - 1), num_heads))

        # get pair-wise relative position index for
        # each token inside the window
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(
            coords_d,
            coords_h,
            coords_w,
        ))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:, None, :]
        # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * \
            (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

        relative_position_index = relative_position_index.view(window_size[0], window_size[1]*window_size[2], window_size[0], window_size[1]*window_size[2]).permute(
            0, 2, 1, 3).reshape(window_size[0]*window_size[0], window_size[1]*window_size[2], window_size[1]*window_size[2])[::window_size[0], :, :]

        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.shift_ops = HeadReallocation(num_frames=num_frames, num_theads=zero_cfg.get('num_theads'))

        self.T_Adapter_in = LinearAdapter(embed_dims, scale=zero_cfg.get('adapter_scale', 1), mlp_ratio=0.25)
        self.S_Adapter = LinearAdapter(embed_dims, scale=zero_cfg.get('adapter_scale', 1), mlp_ratio=0.25)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                batch_size=8, frame_len=8) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): Input feature maps of shape
                :meth:`(B*num_windows, N, C)`.
            mask (torch.Tensor, optional): (0/-inf) mask of shape
                :meth:`(num_windows, N, N)`. Defaults to None.
        """
        B_, N, C = x.shape

        x = self.T_Adapter_in(x)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        k = self.shift_ops(k)
        v = self.shift_ops(v)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:].reshape(-1), :].reshape(
            frame_len, N, N, -1)  # 8frames ,Wd*Wh*Ww,Wd*Wh*Ww,nH

        relative_position_bias = relative_position_bias.permute(
            0, 3, 1, 2).contiguous()  # Frames, nH, Wd*Wh*Ww, Wd*Wh*Ww

        if self.zero_cfg.get('shift_pos', True):
            # print('shift_ops')
            relative_position_bias = self.shift_ops.shift_pos(relative_position_bias)

        # print('309 attn shape', attn.shape)
        attn = attn.view(batch_size, frame_len, -1, self.num_heads, N, N).permute(
            0, 2, 1, 3, 4, 5) + relative_position_bias.unsqueeze(0).unsqueeze(1)  # B_, nH, N, N

        # (batch_size, self.num_spatial_windows, frame_len, self.num_heads, N, N)
        # print('313 attn shape', attn.shape) # [B, 64(num_windows), 16(T), num_heads, 7*7(N), 7*7])

        attn = attn.permute(0, 2, 1, 3, 4, 5).view(-1, self.num_heads, N, N) # 这里刚好permute回去

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # if self.shift and self.shift_type == 'tsm':
        #     x = self.shift_op_back(attn @ v, batch_size,
        #                            frame_len).transpose(1, 2).reshape(B_, N, C)
        # else:
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        x = self.S_Adapter(x)

        return x


class Mlp(BaseModule):
    """Multilayer perceptron.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden features.
            Defaults to None.
        out_features (int, optional): Number of output features.
            Defaults to None.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
        init_cfg (dict, optional): Config dict for initialization.
            Defaults to None.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_cfg: Dict = dict(type='GELU'),
                 drop: float = 0.,
                 init_cfg: Optional[Dict] = None,
                 zero_cfg=dict()) -> None:
        super().__init__(init_cfg=init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.zero_cfg = zero_cfg

        self.MLP_Adapter = LinearAdapter(in_features, scale=self.zero_cfg.get('adapter_scale', 1), mlp_ratio=0.25)
        self.MLP_Adapter_out = LinearAdapter(out_features, scale=self.zero_cfg.get('adapter_scale', 1), mlp_ratio=0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        x = self.MLP_Adapter(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.MLP_Adapter_out(x)
        return x
    
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
                 zero_cfg=dict(),
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp


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
            'zero_cfg': zero_cfg
        }
        self.attn = WindowAttention2D(**_attn_cfg)

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        _mlp_cfg = {
            'in_features': embed_dims,
            'hidden_features': int(embed_dims * mlp_ratio),
            'act_cfg': act_cfg,
            'drop': drop,
            'zero_cfg': zero_cfg
        }
        self.mlp = Mlp(**_mlp_cfg)

    def forward_part1(self, x: torch.Tensor,
                      mask_matrix: torch.Tensor) -> torch.Tensor:
        """Forward function part1."""
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
                 window_size: Sequence[int] = (8, 7, 7),
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
                 zero_cfg=dict(),
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.with_cp = with_cp


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
                'zero_cfg': zero_cfg
            }

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
class SwinTransformer2D_Zero(BaseModule):
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
        with_cp: bool = False,
        zero_cfg=dict(),
        num_theads=(0, 0, 0, 0),
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
            
            now_zero_cfg = copy.deepcopy(zero_cfg)
            now_zero_cfg['num_theads'] = num_theads[i]
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
                'zero_cfg': now_zero_cfg
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

        ## initialize Adapter
        for n, m in self.named_modules():
            if 'Adapter' in n or 'adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)


        ## freeze some parameters
        for name, param in self.named_parameters():
            if 'relative_position_bias_table' not in name and 'norm3' not in name and 'Adapter' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        for name, param in self.named_parameters():
            logger.info('{}: {}'.format(name, param.requires_grad))
        num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.parameters())
        logger.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    

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
        super(SwinTransformer2D_Zero, self).train(mode)