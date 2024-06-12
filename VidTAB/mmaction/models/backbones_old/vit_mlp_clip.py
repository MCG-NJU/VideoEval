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

class MorphFC_T(nn.Module):
    def __init__(self, in_channels, num_frames, segment_dim=8, qkv_bias=True):
        super().__init__()
        self.num_frames = num_frames
        self.segment_dim = segment_dim
        self.hidden_dim = in_channels * num_frames // segment_dim
        self.mlp_t = nn.Linear(self.hidden_dim, self.hidden_dim, bias=qkv_bias)

    def forward(self, x):
        # x is (HW+1, BT, D)
        N, BT, D = x.shape
        B, T = BT // self.num_frames, self.num_frames
        S = D // self.segment_dim
        x = x.view(N, B, T, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(N, B, self.segment_dim, T * S)
        x = self.mlp_t(x).view(N, B, self.segment_dim, T, S).permute(0, 1, 3, 2, 4).reshape(N, BT, D)

        return x
    
class MLP_Adapter(nn.Module):
    def __init__(self, num_frames, D_features, D_hidden_features=384, act_layer=nn.GELU, skip_connect=True, mlp_cfg=dict()):
        super().__init__()
        self.skip_connect = skip_connect
        self.num_frames = num_frames
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        if mlp_cfg.get('type') == 'morph':
            self.mlp = MorphFC_T(in_channels=D_hidden_features, num_frames=num_frames, segment_dim=mlp_cfg.get('segment_dim'))
        else:
            raise NotImplementedError(mlp_cfg.get('type'))
        self.act = act_layer()
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (HW+1, BT, D)
        xs = self.D_fc1(x)
        xs = self.mlp(xs)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
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
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, num_adapters=1, num_frames=8, drop_path=0., mlp_cfg=dict()):
        super().__init__()
        self.num_adapters = num_adapters
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

        
        self.mlp_adapter_before = MLP_Adapter(num_frames, d_model, skip_connect=False, mlp_cfg=mlp_cfg) # 下面drop path已经有个残差了
        if num_adapters == 2:
            self.mlp_adapter_after = MLP_Adapter(num_frames, d_model, skip_connect=False, mlp_cfg=mlp_cfg) # 下面drop path已经有个残差了
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        ## x shape [HW+1, BT, D]
        n, bt, d = x.shape
        ## spatial temporal adaptation
        x = x + self.drop_path(self.mlp_adapter_before(self.ln_1(x)))
        x = x + self.attention(self.ln_1(x))
        if self.num_adapters == 2:
            x = x + self.drop_path(self.mlp_adapter_after(self.ln_1(x)))
        ## mlp
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_adapters=1, drop_path=0.1, mlp_cfg=dict()):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, num_adapters, num_frames, dpr[i], mlp_cfg=mlp_cfg) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@MODELS.register_module()
class ViT_MLP_CLIP(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate,
                  num_adapters=1, pretrained=None, mlp_cfg=dict(type='morph', segment_dim=8)):
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

        self.transformer = Transformer(num_frames, width, layers, heads, num_adapters=num_adapters, drop_path=drop_path_rate, mlp_cfg=mlp_cfg)

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

        ## initialize ST-Adapter
        for n, m in self.transformer.named_modules():
            if 'adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            logger.info(f"{n}:{n2} is initialized with zero.")
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## freeze some parameters
        for name, param in self.named_parameters():
            if 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name and 'adapter' not in name:
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
