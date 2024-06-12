from collections import OrderedDict
from typing import Tuple, Union
from mmcv.cnn.bricks import DropPath
from mmengine.model.weight_init import trunc_normal_
from timm.models.layers import to_2tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip
from mmengine.logging import MMLogger
from einops import rearrange
from mmaction.registry import MODELS

# class RepAdapter2D(nn.Module):
#     """ Pytorch Implemention of RepAdapter for 2d tensor"""

#     def __init__(
#             self,
#             in_features=768,
#             hidden_dim=8,
#             groups=2,
#             scale=1
#     ):
#         super().__init__()
#         self.conv_A = nn.Conv2d(in_features, hidden_dim, 1, groups=1, bias=True)
#         self.conv_B = nn.Conv2d(hidden_dim, in_features, 1, groups=groups, bias=True)
#         self.dropout = nn.Dropout(0.1)
#         self.groups = groups
#         self.scale = scale

#         nn.init.xavier_uniform_(self.conv_A.weight)
#         nn.init.zeros_(self.conv_A.bias)
#         nn.init.zeros_(self.conv_B.weight)
#         nn.init.zeros_(self.conv_B.bias)

#     def forward(self, x):
#         x = self.conv_B(self.dropout(self.conv_A(x))) * self.scale + x
#         return x
    

class ActRepAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True, scale=1.):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features) # TODO 分组先不实现
        self.act = act_layer()
        self.dropout = nn.Dropout(0.1)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.scale = scale

    def forward(self, x):
        # x is (HW+1, BT, D)
        x = self.act(x)
        xs = self.D_fc1(x)
        xs = self.dropout(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs * self.scale
        else:
            x = xs * self.scale
        return x

class ActLayNormRepAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True, scale=1.):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features) # TODO 分组先不实现
        self.act = act_layer()
        self.dropout = nn.Dropout(0.1)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.scale = scale
        self.ln = LayerNorm(D_features)

    def forward(self, x):
        # x is (HW+1, BT, D)
        x = self.act(x)
        x = self.ln(x)
        xs = self.D_fc1(x)
        xs = self.dropout(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs * self.scale
        else:
            x = xs * self.scale
        return x
    
class RepAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, skip_connect=True, scale=1.):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features) # TODO 分组先不实现
        self.dropout = nn.Dropout(0.1)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.scale = scale

    def forward(self, x):
        # x is (HW+1, BT, D)
        xs = self.D_fc1(x)
        xs = self.dropout(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs * self.scale
        else:
            x = xs * self.scale
        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    adapter_map = {'rep': RepAdapter, 'act_rep': ActRepAdapter, 'act_rep_ln': ActLayNormRepAdapter}
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_tadapter=1, num_frames=8, drop_path=0., adapter_type='rep', mlp_ratio=0.25):
        super().__init__()
        self.num_tadapter = num_tadapter
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

        self.MLP_Adapter = self.adapter_map[adapter_type](d_model, scale=scale, mlp_ratio=mlp_ratio)
        self.S_Adapter = self.adapter_map[adapter_type](d_model, scale=scale, mlp_ratio=mlp_ratio)
        self.T_Adapter = self.adapter_map[adapter_type](d_model, skip_connect=False, scale=scale, mlp_ratio=mlp_ratio)
        if num_tadapter == 2:
            self.T_Adapter_in = self.adapter_map[adapter_type](d_model, scale=scale, mlp_ratio=mlp_ratio)
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        ## x shape [HW+1, BT, D]
        n, bt, d = x.shape
        ## temporal adaptation
        xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames)
        if self.num_tadapter == 2:
            xt = self.T_Adapter(self.attention(self.T_Adapter_in(self.ln_1(xt))))
        else:
            xt = self.T_Adapter(self.attention(self.ln_1(xt)))
        xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
        x = x + self.drop_path(xt)
        ## spatial adaptation
        x = x + self.attention(self.S_Adapter(self.ln_1(x)))
        ## joint adaptation
        x = x + self.mlp(self.MLP_Adapter(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1, scale=1., drop_path=0.1, adapter_type='rep', mlp_ratio=0.25):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i], adapter_type=adapter_type, mlp_ratio=mlp_ratio) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@MODELS.register_module()
class ViT_RepAdapter_CLIP(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None, adapter_type='rep', mlp_ratio=0.25):
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=num_tadapter, scale=adapter_scale, drop_path=drop_path_rate, adapter_type=adapter_type, mlp_ratio=mlp_ratio)

        self.ln_post = LayerNorm(width)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = MMLogger.get_current_instance()
            
            ## Load OpenAI CLIP pretrained weights
            if self.layers == 12:
                logger.info(f'load model from: {self.pretrained} clip ViT-B/16')
                clip_model, _ = clip.load("ViT-B/16", device="cpu", download_root=self.pretrained)
            else:
                logger.info(f'load model from: {self.pretrained} clip ViT-L/14')
                clip_model, _ = clip.load("ViT-L/14", device="cpu", download_root=self.pretrained)
            pretrain_dict = clip_model.visual.state_dict()
            del clip_model
            del pretrain_dict['proj']
            msg = self.load_state_dict(pretrain_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            raise NotImplementedError('why do not you use the clip pretrained model?')
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        ## initialize Adapter
        for n, m in self.transformer.named_modules():
            if 'Adapter' in n or 'adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)


        ## freeze some parameters
        for name, param in self.named_parameters():
            if 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name:
                param.requires_grad = False

        for name, param in self.named_parameters():
            logger.info('{}: {}'.format(name, param.requires_grad))
        num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.parameters())
        logger.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
        

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def forward(self, x: torch.Tensor):
        ## Space-only
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
            
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0])

        x = rearrange(x, '(b t) d -> b d t',b=B,t=T)
        
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        return x
