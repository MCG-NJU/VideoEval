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
from ..common import STDHAv2


class LinearAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, skip_connect=True, scale=1.):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features) # TODO 分组先不实现
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.scale = scale
        

    def forward(self, x):
        # x is (HW+1, BT, D)
        xs = self.D_fc1(x)
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
    adapter_map = {'linear': LinearAdapter}
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_adapters=4,
                 num_frames=8, adapter_type='linear', mlp_ratio=0.25, zero_cfg=dict()):
        super().__init__()
        self.num_adapters = num_adapters
        self.zero_cfg = zero_cfg
        self.attn = STDHAv2(embed_dim=d_model, num_heads=n_head, num_frames=num_frames, 
                            shift_pattern=self.zero_cfg.get('shift_pattern', 'kv'),
                            changed_dim=self.zero_cfg.get('changed_dim', 128),
                            tc_cfg=self.zero_cfg.get('tc_cfg', (3,)))

        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

        logger = MMLogger.get_current_instance()
        logger.info(f'num_adapters:{num_adapters}, mlp_ratio:{mlp_ratio} scale:{scale} zero_cfg: {zero_cfg}')

        self.MLP_Adapter = self.adapter_map[adapter_type](d_model, scale=scale, mlp_ratio=mlp_ratio)
        if self.num_adapters >= 4:
            self.MLP_Adapter_out = self.adapter_map[adapter_type](d_model, scale=scale, mlp_ratio=mlp_ratio)

        self.S_Adapter = self.adapter_map[adapter_type](d_model, scale=scale, mlp_ratio=mlp_ratio)
        self.T_Adapter_in = self.adapter_map[adapter_type](d_model, scale=scale, mlp_ratio=mlp_ratio)

        self.num_frames = num_frames
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.gradcam_spatial = nn.Identity() 
        # self.gradcam_temporal = nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # x[:, :, :256] = self.gradcam_temporal(x[:, :, :256]) # 7 theads
        # x[:, :, 256:] = self.gradcam_spatial(x[:, :, 256:])
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # x shape [HW+1, BT, D]
        # spatial adaptation with shift
        x = x + self.S_Adapter(self.attention(self.T_Adapter_in(self.ln_1(x))))
        # joint adaptation
        if self.num_adapters >= 4:
            x = x + self.MLP_Adapter_out(self.mlp(self.MLP_Adapter(self.ln_2(x))))
        else:
            x = x + self.mlp(self.MLP_Adapter(self.ln_2(x)))
        return x

    
class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_adapters=4, scale=1., adapter_type='linear', mlp_ratio=0.25, zero_cfg=dict()):
        super().__init__()
        self.width = width
        self.layers = layers
        # dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(d_model=width, n_head=heads, attn_mask=attn_mask, scale=scale, num_adapters=num_adapters, num_frames=num_frames, adapter_type=adapter_type, mlp_ratio=mlp_ratio, zero_cfg=zero_cfg) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@MODELS.register_module()
class ViT_Zero_CLIP_v2(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, num_adapters=4, adapter_scale=0.5, pretrained=None, adapter_type='linear', mlp_ratio=0.25, zero_cfg=dict()):
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

        self.transformer = Transformer(num_frames=num_frames, width=width, layers=layers, heads=heads, num_adapters=num_adapters, scale=adapter_scale, adapter_type=adapter_type, mlp_ratio=mlp_ratio, zero_cfg=zero_cfg)

            
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
        logger = MMLogger.get_current_instance()
            
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            
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
            # raise NotImplementedError('why do not you use the clip pretrained model?')
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
    

    def forward(self, x: torch.Tensor):
        ## Space-only
        B, C, T, H, W = x.shape
        assert T == self.num_frames, f"{T} != {self.num_frames}"
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
        x = x[:, 0] # 取cls token ND

        x = self.ln_post(x)

        x = rearrange(x, '(b t) d -> b d t',b=B,t=T)
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        return x