# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils import checkpoint as cp
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmaction.registry import MODELS
from ..common import LongTermTDM, TemporalConv
from .resnet import Bottleneck, ResNet
from .conv2former import Conv2Former, Conv2FormerBlock


class LongTermBlock(nn.Module):
    """Long-Term Block for TDN.

    This block is proposed in `Temporal Difference Networks for Efficient Action Recognition`

    The long-term difference module (L-TDM) and temporal convolution (tc) is embedded into 
    ResNet-Block after the first Conv2D, which turns the vanilla ResNet-Block into LongTerm-Block.

    Args:
        block (nn.Module): Residual blocks to be substituted.
        num_segments (int): Number of frame segments.
        ltdm_cfg (dict | None): Config for long-term difference module (L-TDM).
            Default: dict().
        tc_cfg (dict | None): Config for temporal convolution.
            Default: dict().
    """

    def __init__(self, block, num_segments, ltdm_cfg=dict(), tc_cfg=dict()):
        super().__init__()
        self.ltdm_cfg = deepcopy(ltdm_cfg)
        self.tc_cfg = deepcopy(tc_cfg)
        self.block = block
        self.num_segments = num_segments
        self.ltdm = LongTermTDM(block.conv1.out_channels,
                                n_segment=self.num_segments, **self.ltdm_cfg)
        self.tc = TemporalConv(block.conv1.out_channels,
                               n_segment=self.num_segments, **self.tc_cfg)

        if not isinstance(self.block, Bottleneck):
            raise NotImplementedError('LongTerm-Blocks have not been fully '
                                      'implemented except the pattern based '
                                      'on Bottleneck block.')

    def forward(self, x):
        assert isinstance(self.block, Bottleneck)

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.block.conv1(x)
            out = self.ltdm(out)
            out = self.tc(out)
            out = self.block.conv2(out)
            out = self.block.conv3(out)

            if self.block.downsample is not None:
                identity = self.block.downsample(x)

            out = out + identity

            return out

        if self.block.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.block.relu(out)

        return out


@MODELS.register_module()
class TDN(ResNet):
    """Temporal Difference Network (TDN) backbone.

    This backbone is proposed in `Temporal Difference Networks for Efficient Action Recognition`

    Embedding the short-term difference module (S-TDM) and long-term difference module (L-TDM) 
    into ResNet to instantiate TDN.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_segments (int): Number of frame segments.
        stdm_cfg (dict | None): Config for short-term difference module (S-TDM).
            Default: dict().
        ltdm_cfg (dict | None): Config for long-term difference module (L-TDM).
            Default: dict().
        tc_cfg (dict | None): Config for temporal convolution.
            Default: dict().
        **kwargs (keyword arguments, optional): Arguments for ResNet except
            ```depth```.
    """

    def __init__(self, depth, num_segments, n_length=5, stdm_cfg=dict(alpha=0.5, beta=0.5), ltdm_cfg=dict(), tc_cfg=dict(), **kwargs):
        super().__init__(depth, **kwargs)
        assert num_segments >= 3
        self.num_segments = num_segments
        # assert self.stem_type == '2d', f"Not support stem type: {self.stem_type} for TDN!!!"
        self.n_length = n_length
        self.stdm_cfg = deepcopy(stdm_cfg)
        self.ltdm_cfg = deepcopy(ltdm_cfg)
        self.tc_cfg = deepcopy(tc_cfg)
        super().init_weights()
        self.make_tdm_modeling()

    def init_weights(self):
        pass

    def _make_stdm_layer(self):
        conv1_weight = deepcopy(self.conv1.conv.weight)  # 复制ImageNet预训练权重
        kernel_size = conv1_weight.size()  # (out_channels(64), in_channels(3), 7, 7)
        # (out_channels(64), 12, 7, 7)
        new_kernel_size = kernel_size[:1] + (3 * 4,) + kernel_size[2:]
        new_kernels = conv1_weight.data.mean(
            dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        self.conv1_5 = ConvModule(
            12,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv1_5.conv.weight.data = new_kernels
        self.maxpool_diff = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer1_diff = deepcopy(self.layer1)  # noqa lxh: 这里新layer应该是和被复制的layer相互独立的

    def make_tdm_modeling(self):
        """Replace ResNet-Block with ShortTermBlock and LongTermBlock."""

        def make_ltdm_block(stage, num_segments, ltdm_cfg=dict(), tc_cfg=dict()):
            blocks = list(stage.children())
            for i, block in enumerate(blocks):
                blocks[i] = LongTermBlock(
                    block, num_segments, deepcopy(ltdm_cfg), deepcopy(tc_cfg))
            return nn.Sequential(*blocks)

        self._make_stdm_layer()
        for i in range(1, self.num_stages):  # skip the fisrt stage for stdm
            layer_name = f'layer{i + 1}'
            res_layer = getattr(self, layer_name)
            setattr(self, layer_name,
                    make_ltdm_block(res_layer, self.num_segments, self.ltdm_cfg, self.tc_cfg))

    def forward(self, x):
        """Defines the computation performed at every call.

        Args: 
            x (torch.Tensor): The input data, which shape is :
            (batch*num_segments, clip_len (n_length), c, h, w)

        Returns:
            torch.Tensor: The feature of the input samples extracted
            by the backbone.
        """
        # stdm
        x_diff = torch.cat([x[:, i, :, :, :] - x[:, i-1, :, :, :]
                           for i in range(1, self.n_length)], dim=1)
        x_diff = self.conv1_5(self.avg_diff(x_diff))
        x_diff = self.maxpool_diff(x_diff)
        # x_diff = self.maxpool_diff(1.0/1.0*x_diff) 这个1.0/1.0不知道干嘛的，应该是调参剩的

        x = self.conv1(x[:, 2, :, :, :])
        x = self.maxpool(x)

        outs = []
        # first fusion
        x = self.layer1(self.stdm_cfg['alpha']*x +
                        self.stdm_cfg['beta']*F.interpolate(x_diff, x.size()[2:]))

        # second fusion
        x_diff = self.layer1_diff(x_diff)
        x = self.stdm_cfg['alpha']*x + self.stdm_cfg['beta'] * \
            F.interpolate(x_diff, x.size()[2:])

        if 0 in self.out_indices:
            outs = [x]
        else:
            outs = []
        for i in range(1, len(self.res_layers)):
            res_layer = getattr(self, self.res_layers[i])
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)


@MODELS.register_module()
class ResNetLTDM(ResNet):
    """ResNet backbone with L-TDM.

    This backbone is proposed in `Temporal Difference Networks for Efficient Action Recognition`

    Embedding the long-term difference module (L-TDM) into ResNet.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_segments (int): Number of frame segments.
        ltdm_cfg (dict | None): Config for long-term difference module (L-TDM).
            Default: dict().
        tc_cfg (dict | None): Config for temporal convolution.
            Default: dict().
        **kwargs (keyword arguments, optional): Arguments for ResNet except
            ```depth```.
    """

    def __init__(self, depth, num_segments, use_ltdm=(1, 1, 1, 1), ltdm_cfg=dict(), tc_cfg=dict(), **kwargs):
        super().__init__(depth, **kwargs)
        assert num_segments >= 3
        self.num_segments = num_segments
        self.use_ltdm = use_ltdm
        self.ltdm_cfg = deepcopy(ltdm_cfg)
        self.tc_cfg = deepcopy(tc_cfg)
        super().init_weights()
        self.make_ltdm_modeling()

    def init_weights(self):
        pass

    def make_ltdm_modeling(self):
        """Replace ResNet-Block with LongTermBlock."""

        def make_ltdm_block(stage, num_segments, ltdm_cfg=dict(), tc_cfg=dict()):
            blocks = list(stage.children())
            for i, block in enumerate(blocks):
                blocks[i] = LongTermBlock(
                    block, num_segments, deepcopy(ltdm_cfg), deepcopy(tc_cfg))
            return nn.Sequential(*blocks)

        for i in range(self.num_stages):
            if self.use_ltdm[i]:
                layer_name = f'layer{i + 1}'
                res_layer = getattr(self, layer_name)
                setattr(self, layer_name,
                        make_ltdm_block(res_layer, self.num_segments, self.ltdm_cfg, self.tc_cfg))


class LongTermBlock4Conv2Former(nn.Module):
    """Long-Term Block for TDN with Swin2d backbone.

    This block is proposed in `Temporal Difference Networks for Efficient Action Recognition`

    The long-term difference module (L-TDM) and temporal convolution (tc) is embedded into 
    ResNet-Block after the first Conv2D, which turns the vanilla ResNet-Block into LongTerm-Block.

    Args:
        block (nn.Module): Residual blocks to be substituted.
        num_segments (int): Number of frame segments.
        ltdm_cfg (dict | None): Config for long-term difference module (L-TDM).
            Default: dict().
        tc_cfg (dict | None): Config for temporal convolution.
            Default: dict().
    """

    def __init__(self, block, num_segments, ltdm_cfg=dict(), tc_cfg=dict()):
        super().__init__()
        self.block = block
        self.num_segments = num_segments
        self.ltdm_cfg = deepcopy(ltdm_cfg)
        self.tc_cfg = deepcopy(tc_cfg)
        self.ltdm = LongTermTDM(
            block.in_channels, n_segment=self.num_segments, **self.ltdm_cfg)
        self.tc = TemporalConv(
            block.in_channels, n_segment=self.num_segments, **self.tc_cfg)

        if not isinstance(self.block, Conv2FormerBlock):
            raise NotImplementedError('LongTerm-Blocks for swin have not been fully '
                                      'implemented except the pattern based '
                                      'on Conv2Former block.')

    def forward(self, x):
        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""

            x = x + self.block.drop_path(self.block.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                                         * self.block.attn(self.tc(self.ltdm(x))))  # 加的位置还有待商榷 TODO
            x = x + self.block.drop_path(self.block.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                                         * self.block.mlp(x))
                                         
            return x

        if self.block.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


@MODELS.register_module()
class Conv2FormerLTDM(Conv2Former):
    """Conv2Former backbone with L-TDM.

    Embedding the long-term difference module (L-TDM) into Conv2Former.

    Args:
        arch_type (str): Type of swin transformer, from {'tiny', 'small', 'base', 'large'}.
        num_segments (int): Number of frame segments.
        ltdm_cfg (dict | None): Config for long-term difference module (L-TDM).
            Default: dict().
        tc_cfg (dict | None): Config for temporal convolution.
            Default: dict().
        **kwargs (keyword arguments, optional): Arguments for Conv2Former.
    """

    def __init__(self, arch, num_segments=8, use_ltdm=(1, 1, 1, 1), ltdm_cfg=dict(), tc_cfg=dict(), **kwargs):
        super().__init__(arch, **kwargs)
        assert num_segments >= 3
        self.num_segments = num_segments
        self.use_ltdm = use_ltdm
        self.ltdm_cfg = deepcopy(ltdm_cfg)
        self.tc_cfg = deepcopy(tc_cfg)
        super().init_weights()
        self.make_ltdm_modeling()

    def init_weights(self):
        pass

    def make_ltdm_modeling(self):
        """Replace Conv2FormerBlock with LongTermBlock4Conv2Former."""

        def make_ltdm_block(layer, num_segments, ltdm_cfg=dict(), tc_cfg=dict()):
            for i, block in enumerate(layer):
                layer[i] = LongTermBlock4Conv2Former(
                    block, num_segments, deepcopy(ltdm_cfg), deepcopy(tc_cfg))
            # return layer #noqa lxh

        for i in range(self.num_stages):
            if self.use_ltdm[i]:
                make_ltdm_block(
                    self.stages[i], self.num_segments, self.ltdm_cfg, self.tc_cfg)
