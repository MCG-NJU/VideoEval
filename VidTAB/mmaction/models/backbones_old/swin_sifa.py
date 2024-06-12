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

from mmaction.registry import MODELS
from .swin import window_partition, window_reverse, get_window_size, compute_mask, PatchEmbed3D, PatchMerging, Mlp
from .swin_refiner import WindowAttention2D
# from ..utils import print_cuda_memory
from ..common import DefAgg, DefCorFixW


class SIFASwinTransformerBlock3D(BaseModule):
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
                 sifa_cfg: Dict = dict(sifa_kernel=3, cor_dilation=1, cor_group=1),
                 with_cp: bool = False,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.norm_cfg = norm_cfg

        assert 0 <= self.shift_size[0] < self.window_size[
            0], 'shift_size[0] must in [0, window_size[0])'
        assert 0 <= self.shift_size[1] < self.window_size[
            1], 'shift_size[1] must in [0, window_size[0])'
        assert 0 <= self.shift_size[2] < self.window_size[
            2], 'shift_size[2] must in [0, window_size[0])'

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        _attn_cfg = {
            'embed_dims': embed_dims,
            # NOTE 输入不是224 224的时候可能要考虑用get_window_size((D, H, W), self.window_size, self.shift_size)
            'num_spatial_windows': (height // window_size[1]) * (width // window_size[2]),
            'num_heads': num_heads,
            'num_frames': num_frames,
            'window_size': window_size,
            'qkv_bias': qkv_bias,
            'qk_scale': qk_scale,
            'attn_drop': attn_drop,
            'proj_drop': drop
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

        # sifa part
        self.init_sifa(**sifa_cfg)

    def init_sifa(self, sifa_kernel=3, cor_dilation=1, cor_group=1):
        if any(i > 0 for i in self.shift_size): # noqa self.shift_size可能不准确，但是我觉得这里用来判断应该没啥微调
            self.conv = nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=[1, 3], padding=[0, 1], groups=1, bias=False)
        else:
            K = sifa_kernel
            pad_num = (cor_dilation * (K - 1) + 1) // 2
            self.off_channels_ = 2 * K * K

            # inter-frame attention
            self.def_cor = DefCorFixW(in_channels=self.embed_dims, times=self.num_frames, kernel_size=(K,K), stride=1, padding=pad_num, 
                            dilation=cor_dilation, defcor_groups=cor_group)
            self.def_agg = DefAgg(in_channels=self.embed_dims, times=self.num_frames, kernel_size=(K,K), stride=1, padding=pad_num, 
                            dilation=cor_dilation, defagg_groups=cor_group)
            self.tda_norm = build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            nn.init.constant_(self.tda_norm.bias, 0)
            nn.init.constant_(self.tda_norm.weight, 0)


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

    def forward_sifa(self, x):
        B, D, H, W, C = x.shape
        assert D == self.num_frames and H == self.height and W == self.width
        x = x.permute(0, 4, 1, 2, 3).contiguous() # NCTHW
        # x_tmp = x.clone()
        # x_tmp[:,:,1:,:,:] = x_tmp[:,:,:-1,:,:]
        # x_tmp = (torch.sigmoid(x - x_tmp) * x) + x 
        # offset = self.conv_offset(x_tmp) 不知道为啥这个栽种把这个代码注释掉了
        # offset = nn.Parameter(torch.zeros(B, self.off_channels_, D, H, W)).cuda() # 这搞成可学习的参数有毛用啊
        offset = torch.zeros(B, self.off_channels_, D, H, W).cuda()  # 他应该就是想用0，那么直接退化成了普通的帧间attention
        corre_weight = self.def_cor(x, offset)
        x_agg = self.def_agg(x, offset, corre_weight)
        mask = torch.ones(x.size()).cuda()
        mask[:,:,-1,:,:] = 0
        mask.requires_grad = False
        x_shift = x_agg.clone()
        x_shift[:,:,:-1,:,:] = x_shift[:,:,1:,:,:]
        x = x_shift * mask
        
        x = x.permute(0, 2, 3, 4, 1).contiguous() # NTHWC
        x = self.tda_norm(x)

        return x

    def forward_tc(self, x): # TODO
        B, D, H, W, C = x.shape
        x = x.permute(0, 4, 2, 3, 1).contiguous().view(B, C, H*W, D)
        x = self.conv(x)
        x = x.view(B, C, H, W, D).permute(0, 4, 2, 3, 1)
        return x

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

        # sifa part
        if any(i > 0 for i in self.shift_size): # noqa self.shift_size可能不准确，但是我觉得这里用来判断应该没啥微调
            x = x + self.forward_tc(x)
        else:
            x = x + self.forward_sifa(x)

        if self.with_cp:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class SIFABasicLayer(BaseModule):
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
                 sifa_cfg: Dict = dict(sifa_kernel=3, cor_dilation=1, cor_group=1),
                 downsample: Optional[PatchMerging] = None,
                 with_cp: bool = False,
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
                'sifa_cfg': sifa_cfg,
                'with_cp': with_cp
            }

            block = SIFASwinTransformerBlock3D(**_block_cfg)
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
class SwinTransformer2D_SIFA(BaseModule):
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
        patch_size: Union[int, Sequence[int]] = (4, 4, 4),
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
        sifa_cfg: Dict = dict(sifa_kernel=3, cor_dilation=1, cor_group=1),
        patch_norm: bool = True,
        frozen_stages: int = -1,
        with_cp: bool = False,
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
                'sifa_cfg': sifa_cfg,
                'downsample': downsample,
                'with_cp': with_cp
            }

            layer = SIFABasicLayer(**_layer_cfg)
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
            wd = self.num_frames  # 16  NOTE 时序位置编码用多一点，能涨一点点
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
        super(SwinTransformer2D_SIFA, self).train(mode)
        self._freeze_stages()
