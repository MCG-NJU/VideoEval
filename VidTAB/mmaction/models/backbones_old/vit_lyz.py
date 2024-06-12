import os
from collections import OrderedDict
import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from mmcv.cnn.bricks import DropPath, build_norm_layer
from mmengine.logging import MMLogger
from mmaction.registry import MODELS
from mmengine.model import BaseModule
from typing import Optional

class Local_MHRA(BaseModule):
    """Local MHRA.

    Args:
        d_model (int): Number of input channels.
        dw_reduction (float): Downsample ratio of input channels.
            Defaults to 1.5.
        pos_kernel_size (int): Kernel size of local MHRA.
            Defaults to 3.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        d_model: int,
        dw_reduction: float = 1.5,
        pos_kernel_size: int = 3,
        init_cfg: Optional[dict] = None,
        use_ln=False
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        re_d_model = int(d_model // dw_reduction)
        self.pos_embed = nn.Sequential(
            build_norm_layer(dict(type='LN3d', eps=1e-6), d_model)[1] if use_ln else nn.BatchNorm3d(d_model),
            nn.Conv3d(d_model, re_d_model, kernel_size=1, stride=1, padding=0),
            nn.Conv3d(
                re_d_model,
                re_d_model,
                kernel_size=pos_kernel_size,
                stride=(1, 1, 1),
                padding=(pos_kernel_size[0] // 2, pos_kernel_size[1] // 2, pos_kernel_size[2] // 2),
                groups=re_d_model),
            nn.Conv3d(re_d_model, d_model, kernel_size=1, stride=1, padding=0),
        )

        # init zero
        # logger.info('Init zero for Conv in pos_emb')
        nn.init.constant_(self.pos_embed[3].weight, 0)
        nn.init.constant_(self.pos_embed[3].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos_embed(x)

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, drop_path=0., attn_mask=None, dropout=0., dpe_cfg=None, num_frames=0):
        super().__init__()

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # logger.info(f'Droppath: {drop_path}')
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("drop1", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            ("drop2", nn.Dropout(dropout)),
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.num_frames = num_frames

        self.dpe_cfg = dpe_cfg
        if dpe_cfg is not None:
            self.dpe1 = Local_MHRA(d_model, dw_reduction=dpe_cfg.get('dw_reduction', 1.5), pos_kernel_size=dpe_cfg.get('pos_kernel_size', (3, 1, 1)), use_ln=dpe_cfg.get('use_ln', False))
            if dpe_cfg.get('double_dpe', False):
                self.dpe2 = Local_MHRA(d_model, dw_reduction=dpe_cfg.get('dw_reduction', 1.5), pos_kernel_size=dpe_cfg.get('pos_kernel_size', (3, 1, 1)), use_ln=dpe_cfg.get('use_ln', False))

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        if self.dpe_cfg is not None:
            tmp_x = x[1:, :, :]
            LT, B, C = tmp_x.shape
            L = LT // self.num_frames
            T = self.num_frames
            H = W = int(L**0.5)
            # print((H, W, T, B, C))
            tmp_x = tmp_x.view(H, W, T, B, C).permute(3, 4, 2, 0, 1).contiguous()
            tmp_x = tmp_x + self.drop_path1(self.dpe1(tmp_x))
            tmp_x = tmp_x.view(B, C, LT).permute(2, 0, 1).contiguous()
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        x = x + self.drop_path1(self.attention(self.ln_1(x)))

        if self.dpe_cfg is not None and self.dpe_cfg.get('double_dpe', False):
            tmp_x = x[1:, :, :]
            LT, B, C = tmp_x.shape
            L = LT // self.num_frames
            H = W = int(L**0.5)
            tmp_x = tmp_x.view(H, W, T, B, C).permute(3, 4, 2, 0, 1).contiguous()
            tmp_x = tmp_x + self.drop_path2(self.dpe2(tmp_x))
            tmp_x = tmp_x.view(B, C, LT).permute(2, 0, 1).contiguous()
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        x = x + self.drop_path2(self.mlp(self.ln_2(x)))

        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, drop_path=0., checkpoint_num=0, dropout=0., dpe_cfg=None, num_frames=0):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]
        self.resblocks = nn.ModuleList()
        for idx in range(layers):
            self.resblocks.append(ResidualAttentionBlock(width, heads, drop_path=dpr[idx], dropout=dropout, dpe_cfg=dpe_cfg, num_frames=num_frames))
        self.checkpoint_num = checkpoint_num

    def forward(self, x):
        for idx, blk in enumerate(self.resblocks):
            if idx < self.checkpoint_num:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


@MODELS.register_module()
class ViT_lyz(nn.Module):
    def __init__(
        self, input_resolution, patch_size, width, layers, heads, output_dim=None, 
        kernel_size=1, num_frames=8, drop_path=0, checkpoint_num=0, dropout=0.,
        temp_embed=True, pretrained=None, pretrained_type='ViCLIP', freeze_backbone=False,
        dpe_cfg=None
    ):
        super().__init__()
        self.pretrained = pretrained
        self.pretrained_type = pretrained_type
        self.freeze_backbone = freeze_backbone
        self.dpe_cfg = dpe_cfg
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.conv1 = nn.Conv3d(
            3, width, 
            (kernel_size, patch_size, patch_size), 
            (kernel_size, patch_size, patch_size), 
            (0, 0, 0), bias=False
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        if temp_embed:
            self.temporal_positional_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
        
        self.transformer = Transformer(
            width, layers, heads, drop_path=drop_path, checkpoint_num=checkpoint_num,
            dropout=dropout, dpe_cfg=dpe_cfg, num_frames=num_frames)

        self.ln_post = nn.LayerNorm(width)
        if output_dim is not None:
            self.proj = nn.Parameter(torch.empty(width, output_dim))
        else:
            self.proj = None
        
        self.dropout = nn.Dropout(dropout)

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        logger.info(f"pretrained: {self.pretrained}, pretrained type: {self.pretrained_type}")
        if self.pretrained is not None:
            state_dict = torch.load(self.pretrained, map_location='cpu')
            if 'state_dict' in state_dict.keys():
                state_dict = state_dict['state_dict']

            new_state_dict = {}
            if self.pretrained_type == 'k710':
                for k in state_dict.keys():
                    if 'backbone.' in k:
                        new_state_dict[k[9:]] = state_dict[k]
            elif self.pretrained_type == 'ViCLIP':
                for k in state_dict.keys():
                    if 'vision_encoder.' in k and '.proj' not in k:
                        new_state_dict[k[15:]] = state_dict[k]
            elif self.pretrained_type == 'CLIP':
                new_state_dict = state_dict
            else:
                raise NotImplementedError(f"Not support pretrained type: {self.pretrained_type}")

            load_state_dict_lyz(self, new_state_dict, input_resolution=self.input_resolution, patch_size=self.patch_size, center=True, logger=logger, strict=(self.pretrained_type != 'CLIP' and self.dpe_cfg is None))

        for name, param in self.named_parameters():
            if self.freeze_backbone:
                param.requires_grad_(False)
            logger.info('{}: {}'.format(name, param.requires_grad))
        num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.parameters())
        logger.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # temporal pos
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        if hasattr(self, 'temporal_positional_embedding'):
            if x.size(1) == 1:
                # This is a workaround for unused parameter issue
                x = x + self.temporal_positional_embedding.mean(1)
            else:
                x = x + self.temporal_positional_embedding
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)

        # if masking_prob > 0.0:
        #     x = self.mask_tokens(x, masking_prob)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  #BND -> NBD
        x = self.transformer(x)

        x = self.ln_post(x)

        x = self.dropout(x[0])

        x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # for I3D head
        # if self.proj is not None:
        #     x = self.dropout(x[0]) @ self.proj
        # else:
        #     x = x.permute(1, 0, 2)  #NBD -> BND

        return x


def inflate_weight(weight_2d, time_dim, center=True):
    
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


def load_state_dict_lyz(model, state_dict, input_resolution=224, patch_size=16, center=True, logger=None, strict=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if k == 'temporal_positional_embedding':
                continue # 后面插值
            if len(state_dict_3d[k].shape) <= 2:
                logger.info(f'Ignore: {k}')
                continue
            logger.info(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            logger.info(f'Init center: {center}')
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)

    if model.pretrained_type != 'CLIP':
        pos_embed_checkpoint = state_dict['positional_embedding']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = (input_resolution // patch_size) ** 2
        orig_size = int((pos_embed_checkpoint.shape[-2] - 1) ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            raise NotImplementedError(f"{orig_size} {new_size}")
            assert old_num_frames == model.num_frames, "还没实现同时对时间和空间都插值（当然也不是不能实现）"
            logger.info(f'Pos_emb from {orig_size} to {new_size}')
            extra_tokens = pos_embed_checkpoint[:1]
            pos_tokens = pos_embed_checkpoint[1:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(0, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
            state_dict['positional_embedding'] = new_pos_embed

        temp_embed_checkpoint = state_dict['temporal_positional_embedding']
        old_num_frames = temp_embed_checkpoint.shape[1]

        if old_num_frames != model.num_frames:
            # raise NotImplementedError(f"{old_num_frames} {model.num_frames}")
            new_temp_embed_checkpoint = torch.nn.functional.interpolate(
                temp_embed_checkpoint.unsqueeze(0), size=(model.num_frames, temp_embed_checkpoint.shape[2]), mode='bicubic', align_corners=False).squeeze(0)
            state_dict['temporal_positional_embedding'] = new_temp_embed_checkpoint
            logger.info(f'tempo pos_emb from {temp_embed_checkpoint.shape} to {new_temp_embed_checkpoint.shape}')

    message = model.load_state_dict(state_dict, strict=strict)
    logger.info(f"Load pretrained weights: {message}")




if __name__ == '__main__':
    model = ViT_lyz(
            224, 14, 1024, 24, 16, output_dim=None, 
        kernel_size=1, num_frames=8, drop_path=0, checkpoint_num=0, dropout=0.,
        temp_embed=True,
    )