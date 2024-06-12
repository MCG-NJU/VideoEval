# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcv.cnn.bricks import DropPath, build_activation_layer, build_norm_layer
from mmengine.model import ModuleList, Sequential
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner.checkpoint import _load_checkpoint, load_checkpoint
from mmengine.logging import MMLogger
from mmaction.registry import MODELS

from ..utils import inflate_conv2d_params, copy_params
from ..utils.embed import AdaptivePadding3d
from ..utils import dwconv_1xnxn, conv_1xnxn
from ..common import TemporalConv3d



class ConvNeXtBlock3d(nn.Module):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        self.inflate_style=inflate_style
        self.inflate_init_type=inflate_init_type

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back

        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def __init__(self,
                 in_channels,
                 norm_cfg=dict(type='LN3d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 inflate_kernel_size=(1, 7, 7),  # 默认不inflate 
                 decompose_conv3d=True,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.decompose_conv3d = decompose_conv3d and (inflate_kernel_size[0] != 1)

        if inflate_kernel_size[0] == 1:  # 退化成2d，速度更快
            self.tc = None
            self.ada_pad3d = None
            self.dwconv = dwconv_1xnxn(in_channels, kernel_size=7, stride=1, padding=3)  # noqa lxh: 反正convnext只有k=7情况，直接写死
        elif decompose_conv3d:
            self.tc = TemporalConv3d(in_channels, kernel_size=8)
            self.ada_pad3d = None
            self.dwconv = dwconv_1xnxn(in_channels, kernel_size=7, stride=1, padding=3)
        else:
            self.tc = None
            self.ada_pad3d = AdaptivePadding3d(kernel_size=inflate_kernel_size, padding="same")
            self.dwconv = nn.Conv3d(in_channels,
                                    in_channels,
                                    kernel_size=inflate_kernel_size,
                                    padding=0,  # noqa lxh: padding用之前的adaptive padding
                                    groups=in_channels)

        self.linear_pw_conv = linear_pw_conv
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv3d, kernel_size=1)

        self.pwconv1 = pw_conv(in_channels, mid_channels)
        self.act = build_activation_layer(act_cfg)
        self.pwconv2 = pw_conv(mid_channels, in_channels)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):

        def _inner_forward(x):
            shortcut = x
            if self.decompose_conv3d:
                x = self.tc(x)
            elif self.ada_pad3d is not None:
                x = self.ada_pad3d(x)
            x = self.dwconv(x)
            x = self.norm(x)

            if self.linear_pw_conv:
                # (N, C, T, H, W) -> (N, T, H, W, C)
                x = x.permute(0, 2, 3, 4, 1)

            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.linear_pw_conv:
                x = x.permute(0, 4, 1, 2, 3)  # permute back

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1, 1))
            x = shortcut + self.drop_path(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


@MODELS.register_module()
class ConvNeXt3d(nn.Module):
    """ConvNeXt3d.

    A PyTorch implementation of : `A ConvNet for the 2020s
    <https://arxiv.org/pdf/2201.03545.pdf>`_

    Modified from the `official repo
    <https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py>`_.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        pretrained (str | None): Name of pretrained model. Default: None.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
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
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
    }

    def __init__(self,
                 arch='tiny',
                 pretrained=None,
                 pretrained2d=True,
                 in_channels=3,
                 stem_patch_size=(2, 4, 4),
                 norm_cfg=dict(type='LN3d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=False,
                 with_cp=False,
                 inflate=(1, 1, 1, 1),
                 inflate_kernel_size=(8, 7, 7),
                 inflate_init_type='center'):
        super().__init__()

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
        self.in_channels = in_channels
        self.stem_patch_size = stem_patch_size
        self.norm_cfg = norm_cfg
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.inflate = inflate
        self.inflate_kernel_size = inflate_kernel_size
        self.inflate_init_type = inflate_init_type

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
                    build_norm_layer(norm_cfg, self.channels[i - 1])[1],
                    conv_1xnxn(self.channels[i - 1], channels, kernel_size=2, stride=2)
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock3d(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    inflate_kernel_size=self.inflate_kernel_size if self.inflate[i] else (
                        1, 7, 7),
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)[1]
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def _make_stem_layer(self):
        stem = nn.Sequential(
            nn.Conv3d(
                self.in_channels,
                self.channels[0],
                kernel_size=self.stem_patch_size,
                stride=self.stem_patch_size),
            build_norm_layer(self.norm_cfg, self.channels[0])[1]
        )
        self.downsample_layers.append(stem)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)
            else:
                # Directly load 3D model.
                load_checkpoint(
                    self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():  # noqa lxh: 不确定是否对齐了, 源码好像没动layernorm
                if isinstance(m, (nn.Conv3d, nn.Linear)):
                    trunc_normal_(m.weight, std=.02)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        else:
            raise TypeError('pretrained must be a str or None')

    def inflate_weights(self, logger: MMLogger) -> None:
        """Inflate the convnext2d parameters to convnext3d.

        The differences between convnext3d and convnext2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (MMLogger): The logger used to print
                debugging information.
        """
        state_dict_2d = _load_checkpoint(self.pretrained, map_location='cpu')
        if 'state_dict' in state_dict_2d:
            state_dict_2d = state_dict_2d['state_dict']

        inflated_param_names = []
        for name, module in self.named_modules():
            if name == 'norm3':
                copy_params(module, state_dict_2d,
                                    'norm', inflated_param_names)
            elif isinstance(module, nn.Conv3d):
                inflate_conv2d_params(
                    module, state_dict_2d, name, inflated_param_names, self.inflate_init_type)
            elif isinstance(module, (nn.Conv2d, nn.Linear, nn.LayerNorm)):
                copy_params(module, state_dict_2d, name, inflated_param_names)

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(
            state_dict_2d.keys()) - set(inflated_param_names)

        for name, param in self.named_parameters():  # 加载预训练parameters
            if name in remaining_names and param.data.shape != state_dict_2d[name].data.shape:
                print('----- name is same but shape is not same:')
                print(name, param.data.shape, state_dict_2d[name].data.shape)
                print('-----')

            if name in remaining_names and param.data.shape == state_dict_2d[name].data.shape:
                param.data.copy_(state_dict_2d[name].data)
                remaining_names.remove(name)

        if remaining_names:
            logger.info(f'These parameters in the 2d checkpoint are not loaded'
                        f': {remaining_names}')

    def forward(self, x):
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
                    # The output of LayerNorm3d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(norm_layer(x).contiguous())
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

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
        super(ConvNeXt3d, self).train(mode)
        self._freeze_stages()
