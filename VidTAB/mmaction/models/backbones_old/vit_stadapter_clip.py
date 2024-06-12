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
import math
from ..common import XShiftMultiheadAttention_ablation
from ..common import FlashAttention_pytorch
class ST_Adapter(nn.Module): # vit是桶状的，每个stage的dim都相同，所以你这里用ratio和给一个固定值是一样的
    def __init__(self, num_frames, D_features, D_hidden_features=384, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        self.num_frames = num_frames
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.dwconv = nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=(3, 1), padding=(1, 0), groups=D_hidden_features)
        self.act = act_layer()
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (HW+1, BT, D)
        xs = self.D_fc1(x)
        N, BT, D = xs.shape # D_fc1会改变通道数，所以要写后面
        # assert BT % self.num_frames == 0, BT
        B, T = BT // self.num_frames, self.num_frames
        xs = xs.view(N, B, T, D).permute(1, 3, 2, 0).contiguous() # B D T N
        xs = self.dwconv(xs).permute(3, 0, 2, 1).contiguous().view(N, BT, D)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class DividedST_Adapter(nn.Module): # vit是桶状的，每个stage的dim都相同，所以你这里用ratio和给一个固定值是一样的
    def __init__(self, num_frames, D_features, D_hidden_features=384, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        self.num_frames = num_frames
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.dwconv_s = nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=(3, 3), padding=(1, 1), groups=D_hidden_features)
        self.dwconv_t = nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=(3, 1), padding=(1, 0), groups=D_hidden_features)
        self.act = act_layer()
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (HW+1, BT, D)
        x_down = self.D_fc1(x)
        N, BT, D = x_down.shape # D_fc1会改变通道数，所以要写后面
        # assert BT % self.num_frames == 0, BT
        B, T = BT // self.num_frames, self.num_frames
        cls_token = x_down[0:1, :, :]
        H = W = int(math.sqrt(N-1))
        xs = x_down[1:, :, :].view(H, W, BT, D).permute(2, 3, 0, 1).contiguous()
        xs = self.act(self.dwconv_s(xs)).permute(2, 3, 0, 1).reshape(H*W, BT, D)
        xs = torch.cat([cls_token, xs], dim=0)
        xs = xs.view(N, B, T, D).permute(1, 3, 2, 0).contiguous() # B D T N
        xs = self.act(self.dwconv_t(xs)).permute(3, 0, 2, 1).contiguous().view(N, BT, D)
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
    adapter_map = {'divided_st': DividedST_Adapter, 'st': ST_Adapter, 'st_stdha': ST_Adapter} 
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, num_adapters=1, num_frames=8, drop_path=0., adapter_type='st', use_flash=False):
        super().__init__()
        self.num_adapters = num_adapters
        if adapter_type != 'st_stdha':
            if use_flash:
                self.attn = FlashAttention_pytorch(d_model, n_head)
            else:
                self.attn = nn.MultiheadAttention(d_model, n_head)
        else:
            self.attn = XShiftMultiheadAttention_ablation(embed_dim=d_model, num_heads=n_head, 
                num_frames=num_frames, shift_div=12, shift_pattern='kv',
                 ops_type='stdha', shift_stride=1,
                 long_shift_div=-1,
                 long_shift_right=False,
                 )
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head
        self.use_flash = use_flash
        
        self.st_adapter_before = self.adapter_map[adapter_type](num_frames, d_model, skip_connect=False, D_hidden_features=192 if num_adapters == 4 else 384)
        if num_adapters == 4:
            self.st_adapter_before2 = self.adapter_map[adapter_type](num_frames, d_model, skip_connect=True, D_hidden_features=192 if num_adapters == 4 else 384)
        if num_adapters >= 2:
            self.st_adapter_after = self.adapter_map[adapter_type](num_frames, d_model, skip_connect=False, D_hidden_features=192 if num_adapters == 4 else 384)
        if num_adapters == 4:
            self.st_adapter_after2 = self.adapter_map[adapter_type](num_frames, d_model, skip_connect=True, D_hidden_features=192 if num_adapters == 4 else 384)

        self.num_frames = num_frames
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        if self.use_flash:
            if x.dtype != torch.float16:
                old_type = x.dtype
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    x = self.attn(x.half(), is_causal=False)
                return x.to(old_type)
        else:
            raise NotImplementedError
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        ## x shape [HW+1, BT, D]
        n, bt, d = x.shape
        ## spatial temporal adaptation
        x = x + self.st_adapter_before(x)
        if self.num_adapters == 4:
            x = x + self.st_adapter_before2(self.attention(self.ln_1(x)))
        else:
            x = x + self.attention(self.ln_1(x))

        if self.num_adapters >= 2:
            x = x + self.st_adapter_after(x)
        ## mlp
        if self.num_adapters == 4:
            x = x + self.st_adapter_after2(self.mlp(self.ln_2(x)))
        else:
            x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_adapters=1, drop_path=0.1, adapter_type='st', use_flash=False):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, num_adapters, num_frames, dpr[i], adapter_type=adapter_type, use_flash=use_flash) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@MODELS.register_module()
class ViT_ST_Adapter_CLIP(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, num_adapters=1, pretrained=None, adapter_type='st', use_flash=False):
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

        self.transformer = Transformer(num_frames, width, layers, heads, num_adapters=num_adapters, drop_path=drop_path_rate, adapter_type=adapter_type, use_flash=use_flash)

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
            if 'st_adapter' in n:
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
