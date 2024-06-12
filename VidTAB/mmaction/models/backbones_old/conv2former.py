# Copyright (c) OpenMMLab. All rights reserved.
# from functools import partial
# from itertools import chain
# from typing import Sequence
from itertools import chain
from typing import Sequence
from collections import OrderedDict
import re
import logging

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import DropPath, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.logging import MMLogger
from mmaction.registry import MODELS

from ..common import LayerNorm2d, tcs_shift_NCHW, tps_shift_NCHW, temporal_difference_NCHW

# 这边的实现是当作2d recognizer，隔壁conv2former3d是当作3d recognizer


class PositionMLP(BaseModule):
    def __init__(self, in_channels, mlp_ratio=4., norm_cfg=dict(type='LN2d', eps=1e-6), act_cfg=dict(type='GELU')):
        super().__init__()

        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.act = build_activation_layer(act_cfg)
        mid_channels = int(in_channels * mlp_ratio)
        self.fc1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.pos = nn.Conv2d(mid_channels, mid_channels, 3,
                             padding=1, groups=mid_channels)
        self.fc2 = nn.Conv2d(mid_channels, in_channels, 1)

    def forward(self, x):

        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x


class SpatioConvMod(BaseModule):
    def __init__(self, in_channels, norm_cfg=dict(type='LN2d', eps=1e-6),
                    act_cfg=dict(type='GELU'), dwconv_cfg=dict(kernel_size=11, padding=5), shift_cfg=dict()):
        super().__init__()

        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.a = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            build_activation_layer(act_cfg),
            nn.Conv2d(in_channels,
                      in_channels,
                      dwconv_cfg.get('kernel_size'),
                      padding=dwconv_cfg.get('padding'),
                      groups=in_channels)
        )
        self.shift_cfg = shift_cfg

        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):

        x = self.norm(x)

        if self.shift_cfg.get('type', '') == 'tcs':
            x = tcs_shift_NCHW(x, self.shift_cfg.get('num_frames'), shift_div=self.shift_cfg.get('shift_div', 8))
        elif self.shift_cfg.get('type', '') == 'tps':
            x = tps_shift_NCHW(x, self.shift_cfg.get('num_frames'), invert=False, shift_div=self.shift_cfg.get('ratio', 1), stride=self.shift_cfg.get('stride', 1))
        elif self.shift_cfg.get('type', '') == 'td':
            x = temporal_difference_NCHW(x, self.shift_cfg.get('num_frames'), shift_div=self.shift_cfg.get('shift_div', 8))

        a = self.a(x)
        x = a * self.v(x)

        if self.shift_cfg.get('type', '') == 'tps':
            x = tps_shift_NCHW(x, self.shift_cfg.get('num_frames'), invert=True, shift_div=self.shift_cfg.get('ratio', 1), stride=self.shift_cfg.get('stride', 1))

        x = self.proj(x)

        return x

class Conv2FormerBlock(BaseModule):
    """Conv2Former Block.

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
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 dwconv_cfg=dict(kernel_size=11, padding=5),
                 shift_cfg=dict(),
                 mlp_ratio=4.,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.in_channels = in_channels
        self.attn = SpatioConvMod(in_channels, norm_cfg, act_cfg, dwconv_cfg, shift_cfg)
        self.mlp = PositionMLP(in_channels, mlp_ratio, norm_cfg, act_cfg)
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)), requires_grad=True)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):

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

        return x
    
class StackTempoSpatioConvMod(BaseModule):
    """
    stack_type: 'local'对应way4，'global'对应way1
    """
    def __init__(self, in_channels, num_frames, norm_cfg=dict(type='LN2d', eps=1e-6), act_cfg=dict(type='GELU'),
             window_size=(7, 7), stack_type='local'):
        super().__init__()

        self.in_channels = in_channels
        self.num_frames = num_frames
        self.stack_type = stack_type
        self.t_h = int(math.sqrt(num_frames))
        self.t_w = self.num_frames // self.t_h
        assert self.t_w == self.t_h, f"t_w({self.t_w}) != t_h({self.t_h})"  # 目前先用9, 16这样能整数开方的输入吧，方便
        self.window_size = window_size

        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.a = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            build_activation_layer(act_cfg),
            nn.Conv2d(in_channels, in_channels, 11,
                      padding=5, groups=in_channels)
        )

        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)



    def forward(self, x):
        # x: N*T C H W
        NT, C, H, W = x.shape
        N, T = NT // self.num_frames, self.num_frames
        
        if self.stack_type == 'local': # way4 stack
            x = x.view(N, self.t_h, self.t_w, C, H//self.window_size[0], self.window_size[0], W//self.window_size[1], self.window_size[1]).permute(
                0, 3, 4, 1, 5, 6, 2, 7).contiguous().view(N, C, self.t_h*H, self.t_w*W)
        elif self.stack_type == 'global': # way1 stack
            x = x.view(N, self.t_h, self.t_w, C, H, W).permute(0, 3, 1, 4, 2, 5).contiguous().view(N, C, self.t_h*H, self.t_w*W)
        else:
            raise NotImplementedError(f"Not support stack_type: {self.stack_type}!")

        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)

        if self.stack_type == 'local': # way4 复原
            x = x.view(N, C, H//self.window_size[0], self.t_h, self.window_size[0], W//self.window_size[1],
                    self.t_w, self.window_size[1]).permute(0, 3, 6, 1, 2, 4, 5, 7).contiguous().view(N*T, C, H, W)
        elif self.stack_type == 'global': # way1 复原
            x = x.view(N, C, self.t_h, H, self.t_w, W).permute(0, 2, 4, 1, 3, 5).contiguous().view(N*self.t_h*self.t_w, C, H, W)
        else:
            raise NotImplementedError(f"Not support stack_type: {self.stack_type}!")

        return x


class STSConv2FormerBlock(BaseModule):
    """Conv2Former Block.

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
                 num_frames=8,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 with_cp=False,
                 attn_type='Spatial',
                 sts_cfg=dict()):
        super().__init__()
        self.with_cp = with_cp
        self.in_channels = in_channels
        if attn_type == 'Spatial':
            self.attn = SpatioConvMod(in_channels, norm_cfg, act_cfg)
        elif attn_type == 'StackTempoSpatial':
            self.attn = StackTempoSpatioConvMod(
                in_channels, num_frames, norm_cfg, act_cfg, **sts_cfg)
        else:
            raise NotImplementedError(f"Not support attn_type: {attn_type}")
        self.mlp = PositionMLP(in_channels, mlp_ratio, norm_cfg, act_cfg)
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)), requires_grad=True)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):

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

        return x


@MODELS.register_module()
class Conv2Former(BaseModule):
    """Conv2Former.

    A PyTorch implementation of : `Conv2Former: A Simple Transformer-Style ConvNet for Visual Recognition
    <https://arxiv.org/abs/2211.11943.pdf>`_

    Modified from the `official repo
    <https://github.com/HVision-NKU/Conv2Former>`_

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
            'block': STSConv2FormerBlock,
            'depths': [2, 2, 8, 2],
            'channels': [64, 128, 256, 512]
        },
        'T': {
            'block': STSConv2FormerBlock,
            'depths': [3, 3, 12, 3],
            'channels': [72, 144, 288, 576]
        },
        'S': {
            'block': STSConv2FormerBlock,
            'depths': [4, 4, 32, 4],
            'channels': [72, 144, 288, 576]
        },
        'B': {
            'block': STSConv2FormerBlock,
            'depths': [4, 4, 34, 4],
            'channels': [96, 192, 384, 768]
        },
        'L': {
            'block': STSConv2FormerBlock,
            'depths': [4, 4, 48, 4],
            'channels': [128, 256, 512, 1024]
        },
    }

    def __init__(self,
                 arch='T',
                 attn_type='Spatial',
                 pretrained=None,
                 num_frames=8,
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 dwconv_cfg=dict(kernel_size=11, padding=5),
                 sts_cfg=dict(window_size=(7, 7)),
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        self.pretrained = pretrained
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.stem_patch_size = stem_patch_size
        self.norm_cfg = norm_cfg
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

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    LayerNorm2d(self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                arch['block'](
                    in_channels=channels,
                    num_frames=num_frames,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    dwconv_cfg=dwconv_cfg,
                    layer_scale_init_value=layer_scale_init_value,
                    with_cp=with_cp,
                    attn_type=attn_type,
                    sts_cfg=sts_cfg) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)[1]
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def _make_stem_layer(self):
        stem = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.channels[0],
                kernel_size=self.stem_patch_size,
                stride=self.stem_patch_size),
            build_norm_layer(self.norm_cfg, self.channels[0])[1],
        )
        self.downsample_layers.append(stem)

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    # The output of LayerNorm2d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(norm_layer(x).contiguous())
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(
                self,
                self.pretrained,
                strict=False,
                logger=logger,
                revise_keys=[(r'^backbone\.', '')])

        elif self.pretrained is None:
            for m in self.modules():  # noqa lxh: 不确定是否对齐了, 源码好像没动layernorm
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    trunc_normal_(m.weight, std=.02)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        else:
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
        super(Conv2Former, self).train(mode)
        self._freeze_stages()
