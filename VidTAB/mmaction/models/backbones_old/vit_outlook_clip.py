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


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_tadapter=1, num_frames=8, drop_path=0., use_time_attn=False, use_space_attn=True, outlook_type='extra_token_only'):
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


        self.use_time_attn = use_time_attn
        self.outlook_type = outlook_type
        if self.use_time_attn:
            self.T_Adapter = Adapter(d_model, skip_connect=False)
            if num_tadapter == 2:
                self.T_Adapter_in = Adapter(d_model)
        self.num_frames = num_frames


        self.use_space_attn = use_space_attn
        if use_space_attn:
            self.S_Adapter = Adapter(d_model)

        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        
        self.scale = scale

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        
        # self.input_vis = nn.Identity()
        # self.appearance_vis = nn.Identity()
        # self.motion_vis = nn.Identity()

    def attention(self, x: torch.Tensor):
        # print('use attention')
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    
    def attention_outlook(self, x: torch.Tensor):
        # print('use attention outlook')
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        if self.outlook_type == 'extra_token_only':
            # print(f'use {self.outlook_type}')
            # x : [HW+1, BT, D]
            n, bt, d = x.shape  # class attention
            b = bt // self.num_frames
            qkv = x.view(n, b, self.num_frames, d).permute(0, 2, 1, 3).contiguous().view(n*self.num_frames, b, d)
            qkv = qkv[0:self.num_frames, :, :] # 只使用cls token query
            
            # 返回 [T, B, D]
            outlook_cls_token = self.attn(
                qkv, qkv, qkv, need_weights=False, attn_mask=self.attn_mask)[0]
            outlook_cls_token = outlook_cls_token.permute(1, 0, 2).contiguous().view(1, bt, d)
            
            return torch.cat([x, outlook_cls_token], dim=0)
        else:
            raise NotImplementedError
            # x : [HW+1, BT, D]
            n, bt, d = x.shape  # class attention
            b = bt // self.num_frames
            qkv = x.view(n, b, self.num_frames, d).permute(0, 2, 1, 3).contiguous().view(n*self.num_frames, b, d)
            q = qkv[0:self.num_frames, :, :] # 只使用cls token query TODO 也可以考虑再用一个cls token，时空解耦
            kv = qkv[self.num_frames:, :, :]
            
            # 返回 [T, B, D]
            outlook_cls_token = self.attn(
                q, kv, kv, need_weights=False, attn_mask=self.attn_mask, average_attn_weights=False)[0]  # TODO 后面可以考虑加相对位置编码
            outlook_cls_token = outlook_cls_token.permute(1, 0, 2).contiguous().view(1, bt, d)
            x[0, :, :] = outlook_cls_token # 换回去
            return x

    def forward(self, x: torch.Tensor):
        # x shape [HW+1, BT, D]
        n, bt, d = x.shape
        # temporal adaptation

        if self.use_time_attn:
            if self.num_tadapter == 2:
                xt = self.T_Adapter(self.attention_outlook(
                    self.T_Adapter_in(self.ln_1(x))))
            else:
                xt = self.T_Adapter(self.attention_outlook(self.ln_1(x)))
            if 'extra' in self.outlook_type: # 需要考虑最后一个多出来的token
                x = torch.cat([x, torch.zeros(1, bt, d).to(x.device)], dim=0)
            x = x + self.drop_path(xt)
        # spatial adaptation
        if self.use_space_attn:
            x = x + self.S_Adapter(self.attention(self.ln_1(x)))
        if 'extra' in self.outlook_type: # 需要考虑最后一个多出来的token
            x = x[:n, :, :] # 去掉outlook token
        # joint adaptation
        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.drop_path(self.scale *
                                              self.MLP_Adapter(xn))
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1, scale=1., drop_path=0.1, outlook_type='extra_token_only'):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask,                    # TODO 用了time shift就不用attn了
                                       scale, num_tadapter, num_frames, dpr[i], use_time_attn=True, use_space_attn=True, outlook_type=outlook_type) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@MODELS.register_module()
class ViT_Outlook_CLIP(nn.Module):
    # ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int,
                 drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None, outlook_type='extra_token_only'):
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
                                       scale=adapter_scale, drop_path=drop_path_rate, outlook_type=outlook_type)

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