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
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import load_checkpoint

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
        raise NotImplementedError('fold这里的实现有bug，fold应该放在resize之后计算')
        fold = c // self.fold_div
        feat = x
        feat = feat.view(n, bt // self.num_frames,
                         self.num_frames, self.n_head,  c // self.n_head)
        out = feat.clone()

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


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_tadapter=1,
                  num_frames=8, drop_path=0., motion_cfg=dict(), use_frame_attn=True, use_space_attn=True, use_time_shift=False):
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
        self.motion_cfg = motion_cfg


        self.use_frame_attn = use_frame_attn
        self.use_time_attn = False # NOTE 写死不用
        if self.use_time_attn or self.use_frame_attn:
            self.T_Adapter = Adapter(d_model, skip_connect=False)
            if num_tadapter == 2:
                self.T_Adapter_in = Adapter(d_model)
        self.num_frames = num_frames
        self.use_time_shift = use_time_shift
        if use_time_shift:
            self.time_shift = TemporalShift(
                num_frames=num_frames, n_head=n_head)
        else:
            self.time_shift = nn.Identity()

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


    def get_loc_map(self, shape):
        tenHorizontal = torch.linspace(-1.0, 1.0, shape[2]).view(
            1, 1, shape[2], 1).expand(shape[0], shape[1], -1, -1)
        tenVertical = torch.linspace(-1.0, 1.0, shape[1]).view(
            1, shape[1], 1, 1).expand(shape[0], -1, shape[2], -1)
        return torch.cat([tenHorizontal, tenVertical], -1).cuda()
    
    
    def attention_motion(self, x: torch.Tensor):
        # print('use attention motion')
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        # x : [HW+1, BT, D]
        n, bt, d = x.shape  # 帧间注意力
        kv = x.clone().view(n, bt // self.num_frames, self.num_frames, d)
        kv[:, :, :-1, :] = x.view(n, bt // self.num_frames,
                                  self.num_frames, d)[:, :, 1:, :]
        kv = kv.view(n, bt, d)
        x_appearance, attn_maps = self.attn(
            x, kv, kv, need_weights=True, attn_mask=self.attn_mask, average_attn_weights=False)  # TODO 后面可以考虑加相对位置编码

        # x = self.input_vis(x)
        # x_appearance = self.appearance_vis(x_appearance)
        # attn_maps = self.motion_vis(attn_maps)
        if self.motion_cfg.get('use_motion'): # NOTE 记得算的时候排除cls embed
            raise NotImplementedError("uuuuu")
            
            loc_embed = self.loc_embed(loc_map)
            loc_next_embed = (attn_motion @ loc_embed.view(B_, N, self.num_heads, C //
                              self.num_heads).permute(0, 2, 1, 3)).transpose(1, 2).reshape(B_, N, self.motion_dims)
            # TODO 单向的话最后一位motion是没用的，或者考虑改成双向的
            x_motion = self.motion_proj(loc_next_embed - loc_embed)

            if self.motion_cfg.get('use_multihead_motion', False):
                pass
            x_motion = attn_maps
            return x_appearance + x_motion
        else:
            return x_appearance

    def forward(self, x: torch.Tensor):
        # x shape [HW+1, BT, D]
        n, bt, d = x.shape
        # temporal adaptation

        if self.use_frame_attn:
            # print('use frame attn')
            if self.num_tadapter == 2:
                x_motion = self.T_Adapter(self.attention_motion(
                    self.T_Adapter_in(self.time_shift(self.ln_1(x)))))
            else:
                x_motion = self.T_Adapter(self.attention_motion(self.time_shift(self.ln_1(x))))
            if self.motion_cfg.get('type', 'series') == 'series':
                x = x + self.drop_path(x_motion)
                # raise NotImplementedError(self.motion_cfg.get('type'))
            elif self.motion_cfg.get('type', 'series') != 'parallel':
                raise NotImplementedError(self.motion_cfg.get('type'))
            
        # spatial adaptation
        if self.use_space_attn:
            if self.use_frame_attn and self.motion_cfg.get('type', 'series') == 'parallel':
                x = x + self.S_Adapter(self.attention(self.ln_1(x))) + self.drop_path(x_motion)
            else:
                x = x + self.S_Adapter(self.attention(self.ln_1(x)))

        # joint adaptation
        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.drop_path(self.scale *
                                              self.MLP_Adapter(xn))
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1, scale=1., drop_path=0.1, motion_cfg=dict(), use_time_shift=False, allow_time_space=True):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask,                    
                                       scale, num_tadapter, num_frames, dpr[i], motion_cfg, use_frame_attn= i % 2 != 0, use_space_attn=True if (allow_time_space or i % 2 == 0) else False, use_time_shift=use_time_shift) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


@MODELS.register_module()
class ViT_Motion_CLIP(BaseModule):
    # ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int,
                 drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None, motion_cfg=dict(), use_time_shift=False, allow_time_space=True, detection_type=None, with_cls_token=True, frozen_backbone=True):
        super().__init__()
        self.frozen_backbone = frozen_backbone
        self.with_cls_token = with_cls_token
        self.detection_type = detection_type
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)

        self.width = width
        self.patch_size = patch_size
        scale = width ** -0.5
        self.layers = layers
        if self.with_cls_token:
            self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, num_frames, width))

        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=num_tadapter,
                                       scale=adapter_scale, drop_path=drop_path_rate, motion_cfg=motion_cfg, use_time_shift=use_time_shift, allow_time_space=allow_time_space)

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
            if '.pt' in self.pretrained:
                load_checkpoint(
                self,
                self.pretrained,
                strict=False,
                logger=logger,
                revise_keys=[(r'^backbone\.', '')])
            else:
                self.apply(_init_weights)
                
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

                # initialize S_Adapter
                for n, m in self.transformer.named_modules():
                    if 'S_Adapter' in n:
                        for n2, m2 in m.named_modules():
                            if 'D_fc2' in n2:
                                if isinstance(m2, nn.Linear):
                                    nn.init.constant_(m2.weight, 0)
                                    nn.init.constant_(m2.bias, 0)

        elif self.pretrained is None:
            raise NotImplementedError(
                'why do not you use the clip pretrained model?')
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')


        if self.frozen_backbone:
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
        if self.with_cls_token:
            x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0],
                        1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        
        if self.with_cls_token:
            x = x + self.positional_embedding.to(x.dtype)
        else:
            if self.detection_type is not None and (H != 224 or W != 224):
                pos_embed = self.positional_embedding[1:, :].to(x.dtype).permute(0, 1).reshape((self.width, self.input_resolution // self.patch_size, self.input_resolution // self.patch_size)).unsqueeze(0)
                x = x + F.interpolate(pos_embed, (H // self.patch_size, W // self.patch_size)).view(1, self.width, H*W // self.patch_size**2).permute(0, 2, 1)
            else:
                x = x + self.positional_embedding[1:, :].to(x.dtype)

        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.with_cls_token:
            if self.detection_type is None:
                x = self.ln_post(x[:, 0])

                x = rearrange(x, '(b t) d -> b d t', b=B, t=T)

                x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head
            elif self.detection_type == 'simple':
                x = self.ln_post(x[:, 1:]) # B (HW) D
                BT, HW, D = x.shape
                H = W = int(math.sqrt(HW))
                x = x.view(B, BT//B, H, W, D).permute(0, 4, 1, 2, 3)
            else:
                raise NotImplementedError(self.detection_type)
        else:
            x = self.ln_post(x)
            BT, HW, D = x.shape
            x = x.view(B, BT//B, H // self.patch_size, W // self.patch_size, D).permute(0, 4, 1, 2, 3)

        return x
