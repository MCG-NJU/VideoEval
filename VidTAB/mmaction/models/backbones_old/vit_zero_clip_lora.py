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
from ..common import XShiftMultiheadAttention_lora




class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock_RepXViT(nn.Module):
    def __init__(self, d_model: int, n_head: int, num_frames=8, xvit_cfg=dict()):
        super().__init__()
        self.xvit_cfg = xvit_cfg
        if xvit_cfg is None:
            self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head)
        else:
            self.attn = XShiftMultiheadAttention_lora(embed_dim=d_model, num_heads=n_head, 
                    num_frames=num_frames, shift_div=self.xvit_cfg.get('shift_div', 12), shift_pattern=self.xvit_cfg.get('shift_pattern', 'kv'),
                    divide_head=self.xvit_cfg.get('divide_head', False), shift_stride=self.xvit_cfg.get('shift_stride', 1),
                    long_shift_div=self.xvit_cfg.get('long_shift_div', -1),
                    long_shift_right=self.xvit_cfg.get('long_shift_right', False),
                    lora_cfg=self.xvit_cfg.get('lora_cfg', None)
                    )
        self.ln_1 = LayerNorm(d_model)

        self.mlp_lora_cfg = self.xvit_cfg.get('mlp_lora_cfg', None)
        if self.mlp_lora_cfg is not None:
            mlp_lora_width = self.mlp_lora_cfg.get('width')
            if 'up' in self.mlp_lora_cfg.get('type'):
                self.lora_mlpU_down = nn.Linear(d_model, mlp_lora_width, bias=False) 
                nn.init.constant_(self.lora_mlpU_down.weight, 0)
                self.lora_mlpU_up = nn.Linear(mlp_lora_width, d_model*4, bias=False) 
                nn.init.normal_(self.lora_mlpU_up.weight)
            if 'down' in self.mlp_lora_cfg.get('type'):
                self.lora_mlpD_down = nn.Linear(d_model*4, mlp_lora_width, bias=False) 
                nn.init.constant_(self.lora_mlpD_down.weight, 0)
                self.lora_mlpD_up = nn.Linear(mlp_lora_width, d_model, bias=False) 
                nn.init.normal_(self.lora_mlpD_up.weight)
            if self.mlp_lora_cfg.get('type') not in ['up', 'down', 'up_down']:
                raise NotImplementedError(self.mlp_lora_cfg.get('type'))
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = None
        self.n_head = n_head

        logger = MMLogger.get_current_instance()
        logger.info(f'xvit_cfg: {xvit_cfg}')
        self.num_frames = num_frames


    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def compute_mlp(self, x: torch.Tensor):
        if self.mlp_lora_cfg is None:
            return self.mlp(x)
        else:
            if 'up' in self.mlp_lora_cfg.get('type'):
                x = self.mlp.c_fc(x) + self.lora_mlpU_up(self.lora_mlpU_down(x))
            else:
                x = self.mlp.c_fc(x)
            x = self.mlp.gelu(x)
            if 'down' in self.mlp_lora_cfg.get('type'):
                x = self.mlp.c_proj(x) + self.lora_mlpD_up(self.lora_mlpD_down(x))
            else:
                x = self.mlp.c_proj(x)
            return x

    def forward(self, x: torch.Tensor):
        # x shape [HW+1, BT, D]
        x = x + self.attention(self.ln_1(x))
        x = x + self.compute_mlp(self.ln_2(x))
        return x

    
class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, xvit_cfg=dict()):
        super().__init__()
        self.width = width
        self.layers = layers
        # dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock_RepXViT(width, heads, num_frames, xvit_cfg=xvit_cfg) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@MODELS.register_module()
class ViT_Zero_CLIP_LoRA(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, pretrained=None,  xvit_cfg=dict(), final_ta_cfg=dict(), frozen=True):
        super().__init__()
        self.frozen = frozen
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

        self.transformer = Transformer(num_frames, width, layers, heads, xvit_cfg=xvit_cfg)

        self.final_ta_cfg = final_ta_cfg
        if self.final_ta_cfg.get('use_it', False):
            raise NotImplementedError
            
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
            raise NotImplementedError('why do not you use the clip pretrained model?')
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')


        # if self.final_ta_cfg.get('use_it', False): # NOTE 复用最后一层预训练权重
        #     msg = self.final_ta.load_state_dict(self.transformer.resblocks[-1].state_dict(), strict=False)
        #     logger.info('Missing keys for final_ta: {}'.format(msg.missing_keys))
        #     logger.info('Unexpected keys for final_ta: {}'.format(msg.unexpected_keys))
        #     logger.info(f"=> loaded successfully for final_ta'{self.pretrained}'")
        ## initialize Adapter
        # for n, m in self.transformer.named_modules():
        #     if 'LoRA' in n or 'lora' in n:
        #         if 'up' in n:
        #             nn.init.constant_(m.weight, 0)
        #             nn.init.constant_(m.bias, 0)
        #         elif 'down' in n:
        #             nn.init.normal_(m.weight, mean=0.0, std=1.0)
        #             nn.init.constant_(m.bias, 0)
        #         else:
        #             raise NotImplementedError



        ## freeze some parameters
        for name, param in self.named_parameters():
            if self.final_ta_cfg.get('use_it', False) and self.final_ta_cfg.get('type') == 'full_tuning' and 'final_ta' in name:
                param.requires_grad = True
            elif 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name and 'lora' not in name and 'LoRA' not in name:
                if self.frozen:
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
