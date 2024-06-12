from typing import Dict, List, Optional, Union
from functools import partial

from mmaction.registry import MODELS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from mmengine.logging import MMLogger

try:
    import xformers.ops as xops
except:
    print("xformers isn't installed")

try:
    from flash_attn import flash_attn_qkvpacked_func
except:
    print("flash_attn isn't installed")
try:
    from apex.normalization import FusedLayerNorm
except:
    print("Please 'pip install apex'")



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CosAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # self.scale = qk_scale or head_dim**-0.5
        # DO NOT RENAME [self.scale] (for no weight decay)
        if qk_scale is None:
            self.scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))),
                requires_grad=True)
        else:
            self.scale = qk_scale

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (
            F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))

        # torch.log(torch.tensor(1. / 0.01)) = 4.6052
        logit_scale = torch.clamp(self.scale, max=4.6052).exp()

        attn = attn * logit_scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None,
                 attn_type='origin'):
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


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 init_values=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 attn_head_dim=None,
                 cos_attn=False,
                 attn_type='origin',
                 mlp_type='origin'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        print(f'attn_type: {attn_type}, mlp_type: {mlp_type}')
        if attn_type == 'origin':
            if cos_attn:
                self.attn = CosAttention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    attn_head_dim=attn_head_dim)
            else:
                self.attn = Attention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    attn_head_dim=attn_head_dim)
        elif attn_type in ['flash_v2', 'xformer']:
            self.attn = Attention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    attn_head_dim=attn_head_dim,
                    attn_type=attn_type)
        else:
            raise NotImplementedError(f"Not support: {attn_type}")

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_frames=16,
                 tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_spatial_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        num_patches = num_spatial_patches * (num_frames // tubelet_size)

        self.img_size = img_size
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # b, c, l -> b, l, c
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)

@MODELS.register_module()
class VideoMAE(nn.Module):
    """ Modeling VideoMAE for finetuning. """

    def __init__(self,
                 pretrained: str,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dims: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: int = 4.,
                 qkv_bias: bool = True,
                 qk_scale: int = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 init_values: int = 0.,
                 use_learnable_pos_emb=False,
                 num_frames: int = 16,
                 tubelet_size: int = 2,
                 use_mean_pooling: int = True,
                 with_cp: bool = False,
                 cos_attn: bool = False,
                 attn_type: str = 'origin',
                 mlp_type: str = 'origin',
                 adaptation_type='full_finetuning'):
         
        super().__init__()
        self.pretrained = pretrained
        # num_features for consistency with other models
        self.num_features = self.embed_dims = embed_dims
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_frames=num_frames,
            tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.with_cp = with_cp

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dims))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(
                num_patches, embed_dims)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                cos_attn=cos_attn,
                attn_type=attn_type,
                mlp_type=mlp_type) for i in range(depth)
        ])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(
            embed_dims)
        self.fc_norm = norm_layer(embed_dims) if use_mean_pooling else None

        self.adaptation_type = adaptation_type
        if self.adaptation_type == 'frozen_tuning':
            from .attentive_pooler import AttentivePooler
            self.attentive_pooler = AttentivePooler(
                num_queries=1,
                embed_dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                depth=1
            )
        elif self.adaptation_type == 'adapter':
            raise NotImplementedError
        
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)


    def init_weights(self):
        logger = MMLogger.get_current_instance()
            
        if isinstance(self.pretrained, str):
            state_dict = torch.load(self.pretrained, map_location='cpu')
            if 'model' in state_dict.keys():
                state_dict = state_dict['model']
                
            if 'module' in state_dict.keys():
                new_state_dict = state_dict['module']
            else:
                new_state_dict = {}
                for k in state_dict.keys():
                    if k.startswith('encoder.'):
                        new_state_dict[k[8:]] = state_dict[k]
            msg = self.load_state_dict(new_state_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        else:
            raise TypeError('pretrained must be a str or None')

        if self.adaptation_type == 'linear_probe':
            for name, param in self.named_parameters():
                param.requires_grad_(False)
                logger.info('{}: {}'.format(name, param.requires_grad))
        elif self.adaptation_type == 'frozen_tuning':
            for name, param in self.named_parameters():
                if 'attentive_pooler' not in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)
                logger.info('{}: {}'.format(name, param.requires_grad))
        elif self.adaptation_type == 'adapter':
            raise NotImplementedError
        
        num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.parameters())
        logger.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))


    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(
                x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            if self.with_cp:
                x = cp.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.adaptation_type == 'frozen_tuning':
            # print('before:', x.shape)
            x = self.attentive_pooler(x).squeeze(1)
            # print('after:', x.shape)
            return x
        else:
            if self.fc_norm is not None:
                return self.fc_norm(x.mean(1))
            else:
                return self.norm(x[:, 0])




