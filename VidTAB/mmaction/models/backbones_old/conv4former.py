# Copyright (c) OpenMMLab. All rights reserved.
# from functools import partial
# from itertools import chain
# from typing import Sequence
from itertools import chain
from typing import Sequence

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint as cp
from mmcv.cnn.bricks import build_norm_layer, DropPath, build_activation_layer
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model, print_log
from mmengine.logging import MMLogger
from mmaction.registry import MODELS

from ..utils import inflate_conv2d_params  # , print_cuda_memory
from ..common import TemporalConv, tcs_shift_NCHW, tps_shift_NCHW, tsm_shift_NCHW, temporal_rearrange_NCHW, temporal_difference_NCHW, ME

'''
以更接近Uniformer的方式改造conv2former，去掉了conv2former3d中一些没什么卵用的模块
'''

# 后面考虑换成overlapped的方式
# def conv_3xnxn(inp, oup, kernel_size=3, stride=3):
#     return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (2, stride, stride), (1, 1, 1))


# def conv_1xnxn(inp, oup, kernel_size=3, stride=3):
#     return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 1, 1))

# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """

#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         self.proj1 = conv_3xnxn(in_chans, embed_dim//2, kernel_size=3, stride=2)
#         self.norm1= nn.BatchNorm3d(embed_dim//2)
#         self.act=nn.GELU()
#         self.proj2 = conv_1xnxn(embed_dim//2, embed_dim, kernel_size=3, stride=2)
#         self.norm2 = nn.BatchNorm3d(embed_dim)

#     def forward(self, x):
#         x = self.proj1(x)
#         x= self.norm1(x)
#         x=self.act(x)
#         x = self.proj2(x)
#         x = self.norm2(x)
#         return x

# class Downsample(nn.Module):
#     """ Image to Patch Embedding
#     """

#     def __init__(self, in_embed_dim, out_embed_dim, patch_size):
#         super().__init__()
#         self.proj = conv_1xnxn(in_embed_dim, out_embed_dim, kernel_size=3, stride=2)
#         self.norm=nn.LayerNorm(out_embed_dim)

#     def forward(self, x):
#         x = x.permute(0, 4, 1, 2, 3)
#         x = self.proj(x)  # B, C, T, H, W
#         x = x.permute(0, 2, 3, 4, 1)
#         x=self.norm(x)
#         return x

class LocalRefiner(BaseModule):
    def __init__(self, in_channels, num_frames, refine_cfg, init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.refine_cfg = refine_cfg
        self.tc = TemporalConv(in_channels, n_segment=num_frames,
                                             dwconv_cfg=dict(kernel_size=refine_cfg.get('kernel_size', 3), padding=refine_cfg.get('padding', 1)))
    
    def forward(self, x):
        # input shape: NT C H W
        x = self.tc(x)
        if self.refine_cfg.get('use_act'):
            return F.gelu(x)
        else:
            return x
    
class GlobalRefiner(BaseModule):
    def __init__(self, in_channels, num_frames, refine_cfg, init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.refine_cfg = refine_cfg
        assert in_channels % refine_cfg.get(
                    'time_groups') == 0, f"{in_channels} % {refine_cfg.get('time_groups')} != 0"
        time_dim = in_channels // refine_cfg.get('time_groups') * num_frames
        self.fc = nn.Linear(time_dim, time_dim)
    
    def forward(self, x):
        # input shape: NT C H W
        NT, C, H, W = x.shape
        N, T = NT // self.num_frames, self.num_frames
        x = x.reshape(N, T*C // self.refine_cfg.get('time_groups'), self.refine_cfg.get(
                    'time_groups')*H*W).permute(0, 2, 1).contiguous()
        x = self.fc(x).permute(0, 2, 1).contiguous().view(NT, C, H, W)
        
        if self.refine_cfg.get('use_act'):
            return F.gelu(x)
        else:
            return x


class LocalGlobalRefiner(BaseModule):
    def __init__(self, in_channels, num_frames, refine_cfg, init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.refine_cfg = refine_cfg
        assert self.refine_cfg.get('local_ratio') > 0 and self.refine_cfg.get('local_ratio') < 1
        self.local_channels = int(self.refine_cfg.get('local_ratio') * in_channels)
        self.global_channels = in_channels - self.local_channels
        self.local_refiner = LocalRefiner(self.local_channels, num_frames, refine_cfg, init_cfg)
        self.global_refiner = GlobalRefiner(self.global_channels, num_frames, refine_cfg, init_cfg)

    def forward(self, x):
        # NT C H W
        x = torch.cat((self.local_refiner(x[:, :self.local_channels, :, :]), self.global_refiner(x[:, self.local_channels:, :, :])), dim=1)
        return x

class ConvMod3d(BaseModule):
    def __init__(self, in_channels, num_frames, height, width, norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'), dwconv_cfg=dict(kernel_size=11, padding=5),
                 shift_cfg=dict(), pos_cfg=dict(), rearrange_cfg=dict(), refine_cfg=dict()):
        super().__init__()
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]

        self.shift_cfg = shift_cfg
        self.pos_cfg = pos_cfg
        self.rearrange_cfg = rearrange_cfg
        self.refine_cfg = refine_cfg

        if refine_cfg.get('type', '') in ['tc', 'local', 'L']:
            # a.shape: NT C H W
            self.attn_refiner = LocalRefiner(in_channels, num_frames, refine_cfg)
        elif refine_cfg.get('type', '') in ['fc', 'global', 'G']:
            self.attn_refiner = GlobalRefiner(in_channels, num_frames, refine_cfg)
        elif refine_cfg.get('type', '') in ['local_global', 'LG']:
            self.attn_refiner = LocalGlobalRefiner(in_channels, num_frames, refine_cfg)
        elif refine_cfg.get('type'):
            raise NotImplementedError(
                f"Not support refine_type: {refine_cfg.get('type')}!")
        else:
            self.attn_refiner = None

        if pos_cfg.get('type', '') == 'tc':
            self.time_pos = TemporalConv(in_channels, n_segment=num_frames)

        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.attn = nn.Sequential(
            build_activation_layer(act_cfg),
            nn.Conv2d(in_channels,
                      in_channels,
                      dwconv_cfg.get('kernel_size'),
                      padding=dwconv_cfg.get('padding'),
                      groups=in_channels)
        )
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        NT, C, H, W = x.shape
        N, T = NT // self.num_frames, self.num_frames
        rearrange_type = self.rearrange_cfg.get('type', None)
        pos_type = self.pos_cfg.get('type', None)

        x = self.norm(x)

        if self.shift_cfg.get('type', '') == 'tcs':
            x = tcs_shift_NCHW(x, self.num_frames,
                               shift_div=self.shift_cfg.get('shift_div', 8))
        elif self.shift_cfg.get('type', '') == 'td':
            x = temporal_difference_NCHW(
                x, self.num_frames, shift_div=self.shift_cfg.get('shift_div', 8))
        elif self.shift_cfg.get('type') is not None:
            raise NotImplementedError(
                f"Not support shift_type: {self.shift_cfg.get('type')}!")

        q, v = self.q(x), self.v(x)

        if rearrange_type is None:
            a = self.attn(q)
        else:
            a = self.attn(temporal_rearrange_NCHW(q, rearrange_type, self.rearrange_cfg.get('rearrange_length'), (N, T, C, H, W),
                                                  self.rearrange_cfg.get('window_size', None), inverse=False))
            a = temporal_rearrange_NCHW(a, rearrange_type, self.rearrange_cfg.get('rearrange_length'), (N, T, C, H, W), self.rearrange_cfg.get(
                'window_size', None), inverse=True)

        if self.attn_refiner is not None:
            if self.refine_cfg.get('use_res'):
                a = a + self.attn_refiner(a)
            else:
                a = self.attn_refiner(a)
  
        if pos_type is None:
            x = a * v
        else:
            raise NotImplementedError(f"Not support pos_type: {pos_type}!")

        x = self.proj(x)

        return x


class MLP2d(BaseModule):
    def __init__(self, in_channels, num_frames, height, width, mlp_ratio=4., norm_cfg=dict(type='LN2d', eps=1e-6), act_cfg=dict(type='GELU'), pos_cfg=dict()):
        super().__init__()

        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.act = build_activation_layer(act_cfg)
        mid_channels = int(in_channels * mlp_ratio)
        self.fc1 = nn.Conv2d(in_channels, mid_channels, 1)
        if pos_cfg.get('use_time', False):
            self.time_pos = TemporalConv(
                mid_channels, n_segment=num_frames)
        else:
            self.time_pos = None
        self.fc2 = nn.Conv2d(mid_channels, in_channels, 1)

    def forward(self, x):

        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        if self.time_pos is not None:
            x = x + self.time_pos(x)
        x = self.fc2(x)

        return x


class PositionMLP2d(BaseModule):
    def __init__(self, in_channels, num_frames, height, width,  mlp_ratio=4., norm_cfg=dict(type='LN2d', eps=1e-6), act_cfg=dict(type='GELU'), pos_cfg=dict()):
        super().__init__()

        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.act = build_activation_layer(act_cfg)
        mid_channels = int(in_channels * mlp_ratio)
        self.fc1 = nn.Conv2d(in_channels, mid_channels, 1)
        if pos_cfg.get('use_time', False):
            self.time_pos = TemporalConv(
                mid_channels, n_segment=num_frames)
        else:
            self.time_pos = None
        self.pos = nn.Conv2d(mid_channels, mid_channels, 3,
                             padding=1, groups=mid_channels)
        self.fc2 = nn.Conv2d(mid_channels, in_channels, 1)

    def forward(self, x):

        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        if self.time_pos is None:
            x = x + self.act(self.pos(x))
        else:
            x = x + self.act(self.pos(self.time_pos(x)))
        x = self.fc2(x)

        return x


class Conv4FormerBlock(BaseModule):
    """Conv4Former Block.

    Args:
        in_channels (int): The number of input channels.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
    """

    def __init__(self,
                 in_channels,
                 num_frames, height, width,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 dwconv_cfg=dict(kernel_size=11, padding=5),
                 shift_cfg=dict(),
                 attn_pos_cfg=dict(),
                 attn_rearrange_cfg=dict(),
                 attn_refine_cfg=dict(),
                 mlp_pos_cfg=dict(),
                 mlp_ratio=4.,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 with_cp=False,
                 block_pos_cfg=dict(type='divided')):
        super().__init__()
        self.with_cp = with_cp
        self.in_channels = in_channels
        self.attn = ConvMod3d(in_channels, num_frames, height, width, norm_cfg,
                              act_cfg, dwconv_cfg, shift_cfg, attn_pos_cfg, attn_rearrange_cfg, attn_refine_cfg)
        self.block_pos_cfg = block_pos_cfg
        if block_pos_cfg.get('type') is not None:
            if block_pos_cfg.get('type') == 'joint':
                self.pos_embed = nn.Conv3d(
                    in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
            elif block_pos_cfg.get('type') == 'divided':
                self.time_pos_embed = TemporalConv(
                    in_channels, n_segment=num_frames)
                self.pos_embed = nn.Conv2d(
                    in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
            elif block_pos_cfg.get('type') == 'me_divided':
                self.time_pos_embed = nn.Sequential(
                    ME(in_channels, num_segments=num_frames),
                    TemporalConv(in_channels, n_segment=num_frames))
                self.pos_embed = nn.Conv2d(
                    in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
            elif block_pos_cfg.get('type') == 'spatial_only':
                self.pos_embed = nn.Conv2d(
                    in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
            else:
                raise NotImplementedError(
                    f"Not support block pos type:{block_pos_cfg.get('type')}!")
            self.mlp = MLP2d(in_channels, num_frames, height,
                             width, mlp_ratio, norm_cfg, act_cfg, mlp_pos_cfg)
        else:
            raise NotImplementedError("先不用position mlp，太慢了！")
            self.mlp = PositionMLP2d(
                in_channels, num_frames, height, width, mlp_ratio, norm_cfg, act_cfg, mlp_pos_cfg)

        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)), requires_grad=True)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):

        N, C, T, H, W = x.shape
        if self.block_pos_cfg.get('type', '') == 'joint':
            x = x + self.pos_embed(x)

        x = x.permute(0, 2, 1, 3, 4).contiguous().view(N*T, C, H, W)  # 转成2d处理

        if 'divided' in self.block_pos_cfg.get('type', ''):
            # TODO 或许可以考虑换成两个残差连接
            x = x + self.pos_embed(self.time_pos_embed(x))
        elif self.block_pos_cfg.get('type') == 'spatial_only':
            x = x + self.pos_embed(x)
        else:
            raise NotImplementedError(
                f"Not support block pos type:{self.block_pos_cfg.get('type')}!")

        def _inner_forward(x):
            x = x + \
                self.drop_path(
                    self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
            x = x + \
                self.drop_path(
                    self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        x = x.view(N, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()  # 恢复3d
        # print_cuda_memory(f'经过一个Conv2FormerBlock3d后')
        return x


@MODELS.register_module()
class Conv4Former(BaseModule):
    """Conv4Former.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``Conv2Former.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict
    """  # noqa: E501
    arch_settings = {
        'N': {
            'block': Conv4FormerBlock,
            'depths': [2, 2, 8, 2],
            'channels': [64, 128, 256, 512]
        },
        'T': {
            'block': Conv4FormerBlock,
            'depths': [3, 3, 12, 3],
            'channels': [72, 144, 288, 576]
        },
        'S': {
            'block': Conv4FormerBlock,
            'depths': [4, 4, 32, 4],
            'channels': [72, 144, 288, 576]
        },
        'B': {
            'block': Conv4FormerBlock,
            'depths': [4, 4, 34, 4],
            'channels': [96, 192, 384, 768]
        },
        'L': {
            'block': Conv4FormerBlock,
            'depths': [4, 4, 48, 4],
            'channels': [128, 256, 512, 1024]
        },
    }

    def __init__(self,
                 arch='T',
                 pretrained=None,
                 in_channels=3,
                 input_size=(32, 224, 224),  # T H W
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 dwconv_cfg=dict(kernel_size=11, padding=5),
                 act_cfg=dict(type='GELU'),
                 attn_pos_cfg=dict(),
                 attn_rearrange_cfg=dict(),
                 attn_refine_cfg=dict(),
                 mlp_pos_cfg=dict(use_time=False),
                 block_pos_cfg=dict(type='divided'),
                 even_shift_cfg=dict(),
                 odd_shift_cfg=dict(),
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Conv3d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        self.pretrained = pretrained
        self.in_channels = in_channels
        if isinstance(stem_patch_size, int):
            stem_patch_size = (1, stem_patch_size, stem_patch_size)
        self.stem_patch_size = stem_patch_size
        self.norm_cfg = norm_cfg
        # self.sts_cfg = sts_cfg
        self.act_cfg = act_cfg
        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        self._make_stem_layer()

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        num_frames = input_size[0] // stem_patch_size[0]
        height = input_size[1] // stem_patch_size[1]
        width = input_size[2] // stem_patch_size[2]
        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(  # 注意这里为了方便写死了ln3d哈，本质和ln2d一样的
                    build_norm_layer(dict(type='LN3d', eps=1e-6),
                                     self.channels[i - 1])[1],
                    nn.Conv3d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=(1, 2, 2),
                        stride=(1, 2, 2)),
                )
                self.downsample_layers.append(downsample_layer)
                height = height // 2
                width = width // 2

            stage = Sequential(*[
                arch['block'](
                    in_channels=channels,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    dwconv_cfg=dwconv_cfg,
                    act_cfg=act_cfg,
                    attn_pos_cfg=attn_pos_cfg,
                    attn_rearrange_cfg=attn_rearrange_cfg,
                    attn_refine_cfg=attn_refine_cfg,
                    mlp_pos_cfg=mlp_pos_cfg,
                    shift_cfg=even_shift_cfg if j % 2 == 0 else odd_shift_cfg,
                    layer_scale_init_value=layer_scale_init_value,
                    with_cp=with_cp,
                    block_pos_cfg=block_pos_cfg) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(
                    dict(type='LN3d', eps=1e-6), channels)[1]
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def _make_stem_layer(self):
        if isinstance(self.stem_patch_size, int):
            self.stem_patch_size = (
                1, self.stem_patch_size, self.stem_patch_size)
        stem = nn.Sequential(
            nn.Conv3d(
                self.in_channels,
                self.channels[0],
                kernel_size=self.stem_patch_size,
                stride=self.stem_patch_size),
            build_norm_layer(dict(type='LN3d', eps=1e-6), self.channels[0])[1],
        )
        self.downsample_layers.append(stem)

    def forward(self, x):
        # N C T H W
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-3, -2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    # The output of LayerNorm2d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(norm_layer(x).contiguous())
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

    def _load_2dcheckpoint(self):
        """Load checkpoint from a file or URI.
        """
        logger = MMLogger.get_current_instance()
        checkpoint = _load_checkpoint(self.pretrained, map_location='cpu', logger=logger)
        # OrderedDict is a subclass of dict
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f'No state_dict found in checkpoint file {self.pretrained}')
        if 'state_dict' in checkpoint:
            state_dict2d = checkpoint['state_dict']
            print_log(
                f"Checkpoint存在多个keys: {checkpoint.keys()}, 可能引发显存泄露！", logger=logger, level=logging.WARNING)
            del_keys = set()
            for k in checkpoint.keys():
                if k != 'state_dict':
                    del_keys.add(k)
            for k in del_keys:
                # 释放显存，说实话为时已晚了，上面_load_checkpoint就已经带来副作用了，这里就是个安慰
                del checkpoint[k]
        else:
            state_dict2d = checkpoint

        for name, module in self.named_modules():
            if isinstance(module, nn.Conv3d):
                inflate_conv2d_params(
                    module, state_dict2d, f'backbone.{name}', inflate_init_type='center', delete_old=True)
                print_log(
                    f"Load backbone.{name} by `inflate_conv2d_params`.", logger=logger)

        _load_checkpoint_to_model(self, state_dict2d, False, logger, [
                                  (r'^backbone\.', ''), (r'attn\.a\.0', 'attn.q'), (r'attn\.a\.2', 'attn.attn.1')])

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        super().init_weights()  # 不管三七二十一先初始化了再说
        for m in self.modules():  # noqa lxh: 不确定是否对齐了, 源码好像没动layernorm
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(self.pretrained, str):
            self._load_2dcheckpoint()
        elif self.pretrained is not None:
            raise TypeError('pretrained must be a str or None')

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(Conv4Former, self).train(mode)
        self._freeze_stages()
