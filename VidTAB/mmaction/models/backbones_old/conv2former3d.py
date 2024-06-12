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
from ..common import TemporalConv, tcs_shift_NCHW, tps_shift_NCHW, tsm_shift_NCHW, temporal_rearrange_NCHW, temporal_difference_NCHW

'''
这边的实现是当作3d recognizer，隔壁conv2former是当作2d recognizer
3d是考虑到目前mmaction2采样对3d方便一点
这边直接按stage进行flatten转换，保证LayerNorm没问题的同时基本做的是2d操作
'''


def resize_decomposed_rel_pos(rel_pos: torch.Tensor, q_size: int) -> torch.Tensor:
    """Get relative positional embeddings according to the relative positions
    of query and key sizes.

    Args:
        rel_pos (Tensor): relative position embeddings (L, C).
        q_size (int): size of query q.

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * q_size - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        resized = F.interpolate(
            # (L, C) -> (1, C, L)
            rel_pos.transpose(0, 1).unsqueeze(0),
            size=max_rel_dist,
            mode='linear',
        )
        # (1, C, L) -> (L, C)
        resized = resized.squeeze(0).transpose(0, 1)
    else:
        resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.

    q_coords = torch.arange(q_size)[:, None]
    k_coords = torch.arange(q_size)[None, :]
    relative_coords = (q_coords - k_coords) + (q_size - 1)

    return resized[relative_coords.long()]


def get_decomposed_rel_pos_thw(q: torch.Tensor,
                                 rel_pos_t: torch.Tensor,
                                 rel_pos_h: torch.Tensor,
                                 rel_pos_w: torch.Tensor) -> torch.Tensor:
    """Spatiotemporal Relative Positional Embeddings."""
    N, T, H, W, C = q.shape

    Rt = resize_decomposed_rel_pos(rel_pos_t, T)  # T T C
    Rh = resize_decomposed_rel_pos(rel_pos_h, H)  # H H C
    Rw = resize_decomposed_rel_pos(rel_pos_w, W)  # W W C

    rel_t = torch.einsum('bthwc,tkc->bthwk', q, Rt)  # 以t为q,k
    
    rel_h = torch.einsum('bthwc,hkc->bthwk', q, Rh)  # 以h为q,k
    rel_w = torch.einsum('bthwc,wkc->bthwk', q, Rw)  # 以w为q,k
    rel_pos_embed = (
        rel_t[:, :, :, :, :, None, None] +
        rel_h[:, :, :, :, None, :, None] +
        rel_w[:, :, :, :, None, None, :])

    return rel_pos_embed.reshape(N, T*H*W, T*H*W)  # N T H W T H W

def get_decomposed_rel_pos_t(q: torch.Tensor,
                                 rel_pos_t: torch.Tensor) -> torch.Tensor:
    """Temporal Relative Positional Embeddings."""
    N, T, H, W, C = q.shape

    Rt = resize_decomposed_rel_pos(rel_pos_t, T)  # T T C

    rel_t = torch.einsum('bthwc,tkc->bthwk', q, Rt)  # 以t为q,k
    
    return rel_t.reshape(N, T*H*W, T)  # N T H W T H W


class ConvMod3d(BaseModule):
    def __init__(self, in_channels, num_frames, height, width, norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'), dwconv_cfg=dict(kernel_size=11, padding=5),
                 shift_cfg=dict(), pos_cfg=dict(), rearrange_cfg=dict()):
        super().__init__()
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]

        self.shift_cfg = shift_cfg
        self.pos_cfg = pos_cfg
        self.rearrange_cfg = rearrange_cfg

        if pos_cfg.get('type', '') == 'tc':
            self.time_pos = TemporalConv(in_channels, n_segment=num_frames)
        elif pos_cfg.get('type', '') == 'rel_thw':
            self.rel_pos_t = nn.Parameter(torch.zeros(2*num_frames-1, in_channels))
            self.rel_pos_h = nn.Parameter(torch.zeros(2*height-1, in_channels))
            self.rel_pos_w = nn.Parameter(torch.zeros(2*width-1, in_channels))
        elif pos_cfg.get('type', '') == 'rel_t':
            self.rel_pos_t = nn.Parameter(
                torch.zeros(2*num_frames-1, in_channels))

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
        elif self.shift_cfg.get('type', '') == 'tps':
            assert pos_type is None or pos_type == 'tc', "位置编码还不支持shift" # TODO
            x = tps_shift_NCHW(x, self.num_frames, invert=False, ratio=self.shift_cfg.get(
                'ratio', 1), stride=self.shift_cfg.get('stride', 1))
        elif self.shift_cfg.get('type', '') == 'tsm':
            x = tsm_shift_NCHW(x, self.num_frames,
                               shift_div=self.shift_cfg.get('shift_div', 8))
        elif self.shift_cfg.get('type', '') == 'td':
            x = temporal_difference_NCHW(x, self.num_frames, shift_div=self.shift_cfg.get('shift_div', 8))
        elif self.shift_cfg.get('type') is not None:
            raise NotImplementedError(f"Not support shift_type: {self.shift_cfg.get('type')}!")

        q, v = self.q(x), self.v(x)
       
        if rearrange_type is None:
            a = self.attn(q)
        else:
            a = self.attn(temporal_rearrange_NCHW(q, rearrange_type, (N, T, C, H, W),
                       self.rearrange_cfg.get('window_size', None), inverse=False))
            a = temporal_rearrange_NCHW(a, rearrange_type, (N, T, C, H, W), self.rearrange_cfg.get(
                'window_size', None), inverse=True)

        if pos_type is None:
            x = a * v
        elif pos_type == 'tc':
            x = a * self.time_pos(v)
        elif pos_type == 'rel_thw':
            rel_pos = get_decomposed_rel_pos_thw(q.view(N, T, C, H, W).permute(0, 1, 3, 4, 2).contiguous(), self.rel_pos_t, self.rel_pos_h, self.rel_pos_w)
            x = a * v + (rel_pos @ v.permute(0, 2, 3, 1).contiguous().view(N, T*H*W, C)).view(N, T, H, W, C).permute(0, 1, 4, 2, 3).view(N*T, C, H, W) # 太夸张了
        elif pos_type == 'rel_t':
            rel_pos = get_decomposed_rel_pos_t(q.view(N, T, C, H, W).permute(0, 1, 3, 4, 2).contiguous(), self.rel_pos_t)
            x = a * v + (rel_pos @ v.view(N, T, C, H*W).mean(dim=-1)).view(N*T, C, H, W)  # sum改成mean，不然值太大了
            # Be equal to x = a * v +  (rel_pos.repeat((1, 1, H*W)) @ v.permute(0, 2, 3, 1).contiguous().view(N, T*H*W, C)).view(N, T, H, W, C).permute(0, 1, 4, 2, 3).view(N*T, C, H, W)
        else:
            raise NotImplementedError(f"Not support pos_type: {pos_type}!")

        if self.shift_cfg.get('type', '') == 'tps':
            x = tps_shift_NCHW(x, self.num_frames, invert=True, ratio=self.shift_cfg.get(
                'ratio', 1), stride=self.shift_cfg.get('stride', 1))

            x = self.proj(x)

        return x


class PositionMLP3d(BaseModule):
    def __init__(self, in_channels, mlp_ratio=4., norm_cfg=dict(type='LN2d', eps=1e-6), act_cfg=dict(type='GELU'), pos_cfg=dict()):
        super().__init__()

        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.act = build_activation_layer(act_cfg)
        mid_channels = int(in_channels * mlp_ratio)
        self.fc1 = nn.Conv2d(in_channels, mid_channels, 1)
        if pos_cfg.get('use_time', False):
            self.time_pos = TemporalConv(
                mid_channels, n_segment=pos_cfg.get('num_frames'))
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


class Conv2FormerBlock3d(BaseModule):
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
                 num_frames, height, width,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 dwconv_cfg=dict(kernel_size=11, padding=5),
                 shift_cfg=dict(),
                 attn_pos_cfg=dict(),
                 attn_rearrange_cfg=dict(),
                 mlp_pos_cfg=dict(),
                 mlp_ratio=4.,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.in_channels = in_channels
        self.attn = ConvMod3d(in_channels, num_frames, height, width, norm_cfg,
                              act_cfg, dwconv_cfg, shift_cfg, attn_pos_cfg, attn_rearrange_cfg)
        self.mlp = PositionMLP3d(
            in_channels, mlp_ratio, norm_cfg, act_cfg, mlp_pos_cfg)
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
        # print_cuda_memory(f'经过一个Conv2FormerBlock3d后')
        return x


@MODELS.register_module()
class Conv2Former3d(BaseModule):
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
            'block': Conv2FormerBlock3d,
            'depths': [2, 2, 8, 2],
            'channels': [64, 128, 256, 512]
        },
        'T': {
            'block': Conv2FormerBlock3d,
            'depths': [3, 3, 12, 3],
            'channels': [72, 144, 288, 576]
        },
        'S': {
            'block': Conv2FormerBlock3d,
            'depths': [4, 4, 32, 4],
            'channels': [72, 144, 288, 576]
        },
        'B': {
            'block': Conv2FormerBlock3d,
            'depths': [4, 4, 34, 4],
            'channels': [96, 192, 384, 768]
        },
        'L': {
            'block': Conv2FormerBlock3d,
            'depths': [4, 4, 48, 4],
            'channels': [128, 256, 512, 1024]
        },
    }

    def __init__(self,
                 arch='T',
                 pretrained=None,
                 in_channels=3,
                 input_size=(32, 224, 224), # T H W
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 dwconv_cfg=dict(kernel_size=11, padding=5),
                 act_cfg=dict(type='GELU'),
                 attn_pos_cfg=dict(),
                 attn_rearrange_cfg=dict(),
                 mlp_pos_cfg=dict(use_time=False),
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
                    mlp_pos_cfg=mlp_pos_cfg,
                    shift_cfg=even_shift_cfg if j % 2 == 0 else odd_shift_cfg,
                    layer_scale_init_value=layer_scale_init_value,
                    with_cp=with_cp) for j in range(depth)
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
            N, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(N*T, C, H, W)
            x = stage(x)  # NOTE: 默认stage按2d recognizer处理
            x = x.view(N, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

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
        
        # a改名了记得改上 TODO
        inflate_conv2d_params(self.downsample_layers[0][0], state_dict2d,
                              'backbone.downsample_layers.0.0', inflate_init_type='center', delete_old=True)
        print_log(
            "Load backbone.downsample_layers.0.0 by `inflate_conv2d_params`.", logger=logger)
        for i in range(1, 4):
            inflate_conv2d_params(
                self.downsample_layers[i][1], state_dict2d, f'backbone.downsample_layers.{i}.1', inflate_init_type='center', delete_old=True)
            print_log(
                f"Load backbone.downsample_layers.{i}.1 by `inflate_conv2d_params`.", logger=logger)

        _load_checkpoint_to_model(self, state_dict2d, False, logger, [
                                  (r'^backbone\.', ''), (r'attn\.a\.0', 'attn.q'), (r'attn\.a\.2', 'attn.attn.1')])

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            self._load_2dcheckpoint()

        elif self.pretrained is None:
            for m in self.modules():  # noqa lxh: 不确定是否对齐了, 源码好像没动layernorm
                if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
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
        super(Conv2Former3d, self).train(mode)
        self._freeze_stages()
