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
from ..common import XShiftMultiheadAttention
from .vit_aim_clip import ResidualAttentionBlock as AIM_Block
    
class TemporalShift(nn.Module):
    def __init__(self, num_frames, n_head, n_div=8):
        super(TemporalShift, self).__init__()
        self.num_frames = num_frames
        self.fold_div = n_div
        self.n_head = n_head
        print(
            f'=> Using channel shift, num_frames: {self.num_frames}, n_head: {self.n_head}, fold_div: {self.fold_div}')

    def forward(self, x):
        # x is (HW+1, BT, D)

        n, bt, c = x.shape

        feat = x
        feat = feat.view(n, bt // self.num_frames,
                         self.num_frames, self.n_head,  c // self.n_head)
        out = feat.clone() # TODO 为了和 XViT对齐可以改成zero

        fold = c // self.n_head // self.fold_div # NOTE fold记得除以self.n_head
        out[:, :, 1:, :, :fold] = feat[:, :, :-1, :, :fold]  # shift left
        out[:, :, :-1, :, fold:2*fold] = feat[:,
                                              :, 1:, :, fold:2*fold]  # shift right

        out = out.view(n, bt, c)

        return out


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (HW+1, BT, D)
        xs = self.D_fc1(x)
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


class XViT_Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_tadapter=1, num_frames=8, drop_path=0., xvit_cfg=dict()):
        super().__init__()
        self.num_tadapter = num_tadapter
        self.xvit_cfg = xvit_cfg
        if self.xvit_cfg.get('use_real_xvit', False):
            self.attn = XShiftMultiheadAttention(embed_dim=d_model, num_heads=n_head, 
                num_frames=num_frames, shift_div=self.xvit_cfg.get('shift_div', 4),
                 divide_head=self.xvit_cfg.get('divide_head', True), shift_pattern=self.xvit_cfg.get('shift_pattern', 'kv'))
        else:
            self.time_shift = TemporalShift(
                num_frames=num_frames, n_head=n_head, n_div=self.xvit_cfg.get('shift_div', 4))
            self.attn = nn.MultiheadAttention(d_model, n_head) # 使用简化版本

        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head
            
        self.num_frames = num_frames


        self.T_Adapter_in = Adapter(d_model)

                
        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        self.S_Adapter = Adapter(d_model)
        self.scale = scale

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def attention_shift(self, x: torch.Tensor):
        # print('use attention')
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def attention_shift_simple(self, x: torch.Tensor):
        # print('use attention')
        kv = self.time_shift(x)
        return self.attn(x, kv, kv, need_weights=False, attn_mask=None)[0]
    

    def forward(self, x: torch.Tensor):
        # x shape [HW+1, BT, D]
        # n, bt, d = x.shape
        # temporal adaptation

        # if self.use_time_shift:
        #     if self.num_tadapter == 2:
        #         xt = self.T_Adapter(self.time_shift(
        #             self.T_Adapter_in(self.ln_1(x))))
        #     else:
        #         xt = self.T_Adapter(self.time_shift(self.ln_1(x)))
        #     x = x + self.drop_path(xt)
        # spatial adaptation
        if self.xvit_cfg.get('use_real_xvit', False):
            x = x + self.S_Adapter(self.attention_shift(self.T_Adapter_in(self.ln_1(x))))
        else:
            x = x + self.S_Adapter(self.attention_shift_simple(self.T_Adapter_in(self.ln_1(x))))
        # joint adaptation
        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.drop_path(self.scale *
                                              self.MLP_Adapter(xn))
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1, scale=1., drop_path=0.1, xvit_cfg=dict()):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        block_list = []
        for i in range(layers):
            if i == (layers - 1) and xvit_cfg.get('last_AIM', False):
                print(f'Use AIM layer at {i} !')
                block_list.append(AIM_Block(width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i]))
            else:
                print(f'Use XViT layer at {i} !')
                block_list.append(XViT_Block(width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i], xvit_cfg=xvit_cfg))

        self.resblocks = nn.Sequential(*block_list)

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@MODELS.register_module()
class ViT_XViT_CLIP(nn.Module):
    # ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int,
                 drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None, xvit_cfg=dict()):
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, num_frames, width))

        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=num_tadapter,
                                       scale=adapter_scale, drop_path=drop_path_rate, xvit_cfg=xvit_cfg)

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

            # Load OpenAI CLIP pretrained weights
            if self.layers == 12:
                logger.info(
                    f'load model from: {self.pretrained} clip ViT-B/16')
                clip_model, _ = clip.load(
                    "ViT-B/16", device="cpu", download_root=self.pretrained)
            else:
                logger.info(
                    f'load model from: {self.pretrained} clip ViT-L/14')
                clip_model, _ = clip.load(
                    "ViT-L/14", device="cpu", download_root=self.pretrained)
            pretrain_dict = clip_model.visual.state_dict()
            del clip_model
            del pretrain_dict['proj']
            msg = self.load_state_dict(pretrain_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            raise NotImplementedError(
                'why do not you use the clip pretrained model?')
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        # initialize S_Adapter
        for n, m in self.transformer.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        # initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        # initialize MLP_Adapter
        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        # freeze some parameters
        for name, param in self.named_parameters():
            if 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name:
                param.requires_grad = False

        for name, param in self.named_parameters():
            logger.info('{}: {}'.format(name, param.requires_grad))
        num_param = sum(p.numel()
                        for p in self.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.parameters())
        logger.info('Number of total parameters: {}, tunable parameters: {}'.format(
            num_total_param, num_param))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def forward(self, x: torch.Tensor):
        # Space-only
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0],
                      1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
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
        
        x = rearrange(x, '(b t) d -> b d t', b=B, t=T)

        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        return x
