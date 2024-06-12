# Copyright (c) OpenMMLab. All rights reserved.
# from functools import partial
# from itertools import chain
# from typing import Sequence

from collections import OrderedDict
import re
import logging
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import DropPath, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner.checkpoint import load_checkpoint, _load_checkpoint, load_state_dict
from mmengine.logging import MMLogger, print_log
from mmaction.registry import MODELS

from ..utils import inflate_2dcnn_weights
from .conv2former import SpatioConvMod, PositionMLP, Conv2Former

# NOTE: 这版写法可能有个问题，就是LayerNorm全用的2d的，参考timesformer


class SpatioTemporalAdapter(BaseModule):
    def __init__(self, in_channels, mid_channels=384, dwconv_cfg=dict(kernel_size=(3, 1), padding=(1, 0))):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=dwconv_cfg.get('kernel_size', (3, 1)),
                      padding=dwconv_cfg.get('padding', (1, 0)), groups=mid_channels),
            nn.Conv2d(mid_channels, in_channels, 1),       
        )
    
    def forward(self, x):
        NT, C, H, W = x.shape
        N = NT // self.num_frames
        x = x.view(N, self.num_frames, C, H*W).permute(0, 2, 1, 3).contiguous()
        x = self.adapter(x)
        x = x.permute(0, 2, 1, 3).contiguous().view(N*self.num_frames, C, H, W)
        return x


class TempoConvMod(BaseModule):
    def __init__(self, in_channels, num_frames, norm_cfg=dict(type='LN2d', eps=1e-6), dwconv_cfg=dict(kernel_size=(7, 1), padding=(3, 0)), act_cfg=dict(type='GELU')):
        super().__init__()
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.a = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            build_activation_layer(act_cfg),
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=dwconv_cfg.get('kernel_size', (7, 1)),
                      padding=dwconv_cfg.get('padding', (3, 0)), groups=in_channels))

        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        # x: N*T C H W
        NT, C, H, W = x.shape
        N = NT // self.num_frames # 为了只在时间维上norm
        x = x.view(N, self.num_frames, C, H*W).permute(0, 3, 2, 1).contiguous().view(N*H*W, C, self.num_frames).unsqueeze(-1)
        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x) # N*H*W C T 1
        x = x.view(N, H*W, C, self.num_frames).permute(0, 3, 1, 2).contiguous().view(N*self.num_frames, C, H, W)

        return x

# class TempoConvMod(BaseModule):
#     def __init__(self, in_channels, num_frames, norm_cfg=dict(type='LN2d', eps=1e-6), dwconv_cfg=dict(kernel_size=(7, 1), padding=(3, 0)), act_cfg=dict(type='GELU')):
#         super().__init__()
#         self.in_channels = in_channels
#         self.num_frames = num_frames
#         self.norm = build_norm_layer(norm_cfg, in_channels)[1]
#         self.a = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 1),
#             build_activation_layer(act_cfg),
#             nn.Conv2d(in_channels, in_channels,
#                       kernel_size=dwconv_cfg.get('kernel_size', (7, 1)),
#                       padding=dwconv_cfg.get('padding', (3, 0)), groups=in_channels) 
#         )

#         self.v = nn.Conv2d(in_channels, in_channels, 1)
#         self.proj = nn.Conv2d(in_channels, in_channels, 1)

#     def forward(self, x):
#         # x: N*T C H W
#         NT, C, H, W = x.shape
#         N = NT // self.num_frames
#         x = x.view(N, self.num_frames, C, H*W).permute(0, 2, 1, 3).contiguous()
#         x = self.norm(x)
#         a = self.a(x)
#         x = a * self.v(x)
#         x = self.proj(x)
#         x = x.permute(0, 2, 1, 3).contiguous().view(N*self.num_frames, C, H, W)

#         return x


class TempoBottleNeckConvMod(BaseModule):
    def __init__(self, in_channels, num_frames, norm_cfg=dict(type='LN2d', eps=1e-6), dwconv_cfg=dict(kernel_size=(7, 1), padding=(3, 0), reduction=4), act_cfg=dict(type='GELU')):
        super().__init__()
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.reduction = dwconv_cfg.get('reduction', 4)
        self.inner_channels = in_channels // self.reduction
        self.a = nn.Sequential(nn.Conv2d(in_channels, self.inner_channels, 1),
                               nn.Conv2d(self.inner_channels, self.inner_channels,
                                         kernel_size=dwconv_cfg.get(
                                             'kernel_size', (7, 1)),
                                         padding=dwconv_cfg.get('padding', (3, 0)), groups=self.inner_channels),
                               build_activation_layer(act_cfg),
                               nn.Conv2d(self.inner_channels, in_channels, 1))

        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        # x: N*T C H W
        NT, C, H, W = x.shape
        N = NT // self.num_frames
        x = x.view(N, self.num_frames, C, H*W).permute(0, 2, 1, 3).contiguous()
        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)
        x = x.permute(0, 2, 1, 3).contiguous().view(N*self.num_frames, C, H, W)

        return x


class TempoDiffConvMod(BaseModule):  # TODO 这个模块后面再考虑加
    def __init__(self, in_channels, num_frames, norm_cfg=dict(type='LN2d', eps=1e-6), act_cfg=dict(type='GELU'), reduction=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.inner_channels = in_channels//reduction
        self.conv_down = nn.Conv2d(in_channels, self.inner_channels, 1)
        self.conv_act = nn.Conv2d(self.inner_channels, self.inner_channels, (7, 1),
                                  padding=(3, 0), groups=in_channels)
        self.act = build_activation_layer(act_cfg)
        self.conv_up = nn.Conv2d(self.inner_channels, in_channels, 1)

        self.v = nn.Conv2d(in_channels, self.inner_channels, 1)
        self.proj = nn.Conv2d(self.inner_channels, in_channels, 1)

    def forward(self, x):
        # x: N*T C H W
        NT, C, H, W = x.shape
        N = NT // self.num_frames
        x = self.norm(x)  # NOTE 这里norm2d不太对，没准有影响
        x_down = self.conv_down(x)
        # t feature
        reshape_x_down = x_down.view(
            (-1, self.num_frames) + x_down.size()[1:])  # n, t, c//r, h, w
        t_fea, __ = reshape_x_down.split(
            [self.num_frames-1, 1], dim=1)  # n, t-1, c//r, h, w

        # apply transformation conv to t+1 feature
        conv_x_down = self.conv2(x_down)  # nt, c//r, h, w
        # reshape fea: n, t, c//r, h, w
        reshape_conv_x_down = conv_x_down.view(
            (-1, self.num_segments) + conv_x_down.size()[1:])
        __, tPlusone_fea = reshape_conv_x_down.split(
            [1, self.num_segments-1], dim=1)  # n, t-1, c//r, h, w
        # motion fea = t+1_fea - t_fea
        # pad the last timestamp
        diff_fea = tPlusone_fea - t_fea  # n, t-1, c//r, h, w
        # pad = (0,0,0,0,0,0,0,1)
        diff_fea_pluszero = F.pad(
            diff_fea, self.pad, mode="constant", value=0)  # n, t, c//r, h, w
        diff_fea_pluszero = diff_fea_pluszero.view(
            (-1,) + diff_fea_pluszero.size()[2:])  # nt, c//r, h, w
        a = self.conv_up(diff_fea_pluszero)

        x = a * self.v(x)
        x = self.proj(x)
        x = x.permute(0, 2, 1, 3).contiguous().view(N*self.num_frames, C, H, W)

        return x


class SpatioTempoConvMod(BaseModule):
    def __init__(self, in_channels, num_frames, norm_cfg=dict(type='LN2d', eps=1e-6), dwconv_cfg=dict(kernel_size=(7, 11, 11), padding=(3, 5, 5)), act_cfg=dict(type='GELU')):
        super().__init__()

        self.in_channels = in_channels
        self.num_frames = num_frames
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.a = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1),
            build_activation_layer(act_cfg),
            nn.Conv3d(in_channels, in_channels,
                      dwconv_cfg.get('kernel_size', (7, 11, 11)),
                      padding=dwconv_cfg.get('padding', (3, 5, 5)), groups=in_channels)
        )

        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        # x: N*T C H W
        x = self.norm(x)
        NT, C, H, W = x.shape
        N = NT // self.num_frames
        x = x.view(N, self.num_frames, C, H, W).permute(
            0, 2, 1, 3, 4).contiguous()
        a = self.a(x)
        x = a * self.v(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(
            N * self.num_frames, C, H, W)
        x = self.proj(x)

        return x


class Conv3FormerBlock(BaseModule):
    """Conv3Former Block.

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
                 num_frames,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 with_cp=False,
                 attn_type='divided',
                 dwconv_cfg=dict(kernel_size=(7, 1), padding=(3, 0))):
        super().__init__()
        self.with_cp = with_cp
        self.in_channels = in_channels

        self.mlp = PositionMLP(in_channels, mlp_ratio, norm_cfg, act_cfg)
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)), requires_grad=True)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        # 为了加载预训练模型方便尽量不破坏命名
        self.attn_type = attn_type
        if attn_type == 'divided' or attn_type == 'divided_bottleneck':
            self.attn = SpatioConvMod(in_channels, norm_cfg, act_cfg)
            if attn_type == 'divided':
                self.tempo_attn = TempoConvMod(
                    in_channels, num_frames, norm_cfg, dwconv_cfg, act_cfg)
            elif attn_type == 'divided_bottleneck':
                self.tempo_attn = TempoBottleNeckConvMod(
                    in_channels, num_frames, norm_cfg, dwconv_cfg, act_cfg)
            self.layer_scale_3 = nn.Parameter(
                layer_scale_init_value * torch.ones((in_channels)), requires_grad=True)
        elif attn_type == 'joint':
            self.attn = SpatioTempoConvMod(
                in_channels, num_frames, norm_cfg, act_cfg)
        elif attn_type == 'space_only':
            self.attn = SpatioConvMod(in_channels, norm_cfg, act_cfg)
        else:
            raise NotImplementedError(f"Not support attn_type: {attn_type}!")

    def forward(self, x):

        def _inner_forward(x):
            if self.attn_type == 'divided' or self.attn_type == 'divided_bottleneck':
                x = x + \
                    self.drop_path(
                        self.layer_scale_3.unsqueeze(-1).unsqueeze(-1) * self.tempo_attn(x))
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
class Conv3Former(Conv2Former):
    arch_settings = {
        'N': {
            'block': Conv3FormerBlock,
            'depths': [2, 2, 8, 2],
            'channels': [64, 128, 256, 512]
        },
        'T': {
            'block': Conv3FormerBlock,
            'depths': [3, 3, 12, 3],
            'channels': [72, 144, 288, 576]
        },
        'S': {
            'block': Conv3FormerBlock,
            'depths': [4, 4, 32, 4],
            'channels': [72, 144, 288, 576]
        },
        'B': {
            'block': Conv3FormerBlock,
            'depths': [4, 4, 34, 4],
            'channels': [96, 192, 384, 768]
        },
        'L': {
            'block': Conv3FormerBlock,
            'depths': [4, 4, 48, 4],
            'channels': [128, 256, 512, 1024]
        },
    }
    # 'divided', 'space_only', 'joint', 'divided_bottleneck'

    def __init__(self, arch='T', pretrained2d=True, attn_type='divided', dwconv_cfg=dict(kernel_size=(7, 1), padding=(3, 0)), inflate_init_type='center', **kwargs):
        self.arch_settings[arch]['block'] = functools.partial(
            Conv3FormerBlock, attn_type=attn_type, dwconv_cfg=dwconv_cfg)
        self.attn_type = attn_type
        self.pretrained2d = pretrained2d
        self.inflate_init_type = inflate_init_type
        super().__init__(arch, **kwargs)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):  # NOTE 借鉴timesformer加载预训练的方式
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            if self.pretrained2d:
                if self.attn_type == 'space_only':
                    load_checkpoint(
                        self,
                        self.pretrained,
                        strict=False,
                        logger=logger,
                        revise_keys=[(r'^backbone\.', '')])
                elif self.attn_type == 'divided' or self.attn_type == 'divided_bottleneck' or  'divided' in self.attn_type:
                    state_dict = _load_checkpoint(self.pretrained, map_location='cpu')
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']

                    # strip prefix of state_dict
                    metadata = getattr(state_dict, '_metadata', OrderedDict())
                    for p, r in [(r'^backbone\.', '')]:
                        state_dict = OrderedDict(
                            {re.sub(p, r, k): v
                             for k, v in state_dict.items()})
                    # Keep metadata in state_dict
                    state_dict._metadata = metadata

                    # modify the key names of norm layers
                    old_state_dict_keys = list(state_dict.keys())
                    for old_key in old_state_dict_keys:
                        if 'attn' in old_key:
                            if self.attn_type == 'divided' and 'attn.a.2' in old_key:  # dwconv卷积的权重没法复用
                                continue
                            if self.attn_type == 'divided_bottleneck' and 'attn.a' in old_key:  # 整个bottleneck的权重都没法复用
                                continue
                            new_key = old_key.replace('attn.',
                                                        'tempo_attn.')
                            state_dict[new_key] = state_dict[old_key].clone()
                            print_log(f"复制{old_key}的参数到{new_key}",
                                        logger=logger, level=logging.WARNING)
                    for name, m in self.named_modules():
                        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)) and name not in state_dict.keys():
                            print_log(
                                f"手动初始化{name}的参数", logger=logger, level=logging.WARNING)
                            trunc_normal_(m.weight, std=.02)
                            nn.init.constant_(m.bias, 0)
                    
                    load_state_dict(self, state_dict,
                                    strict=False, logger=logger)
                                    
                elif self.attn_type == 'joint':  # need to inflate conv2d
                    inflate_2dcnn_weights(self, self.pretrained, logger, self.inflate_init_type, revise_keys=[
                                         (r'^backbone\.', '')])
            else:
                load_checkpoint(
                    self,
                    self.pretrained,
                    strict=False,
                    logger=logger,
                )

        elif self.pretrained is None:
            for m in self.modules():  # noqa lxh: 不确定是否对齐了, 源码好像没动layernorm
                if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                    trunc_normal_(m.weight, std=.02)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        else:
            raise TypeError('pretrained must be a str or None')



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
            build_norm_layer(dict(type='LN3d', eps=1e-6),
                             self.channels[0])[1],  # NOTE 别的地方应该都是LN2d
        )
        self.downsample_layers.append(stem)

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            if i == 0:  # N C T H W 换成 N*T C H W
                x = x.permute(0, 2, 1, 3, 4).contiguous().flatten(
                    start_dim=0, end_dim=1)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    # 只有stem layer可能会对时序维进行下采样
                    T = self.num_frames # // self.stem_patch_size[0] 设置的时候就应该考虑时序维的下采样
                    gap = x.view(x.shape[0]//T, T, x.shape[1], x.shape[2], x.shape[3]).permute(
                        0, 2, 1, 3, 4).contiguous().mean([-3, -2, -1], keepdim=True).squeeze(-1)  # TODO 如果全改成ln3d那这也要改
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    # The output of LayerNorm2d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(norm_layer(x).contiguous())
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

    def train(self, mode=True):
        super(Conv3Former, self).train(mode)
        self._freeze_stages()
