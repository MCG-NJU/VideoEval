from collections import OrderedDict
from turtle import forward
from typing import Tuple, Union
from mmengine.model.weight_init import trunc_normal_
from mmcv.cnn.bricks import DropPath
from timm.models.layers import to_2tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from mmengine.logging import MMLogger
from einops import rearrange
from functools import partial
from mmaction.registry import MODELS


class HeadReallocation(nn.Module):
    def __init__(self, num_frames, num_theads=1):
        super(HeadReallocation, self).__init__()
        self.num_frames = num_frames
        self.num_theads = num_theads

        logger = MMLogger.get_current_instance()
        logger.info( f'HeadReallocation, num_frames: {self.num_frames}, num_theads: {self.num_theads}')
    
        
    def forward(self, x):
        # x is (B, num_heads, N, C//num_heads)
        
        bt, num_heads, n, c_ = x.shape
        feat = x

        # 不分head shift
        feat = feat.view(bt // self.num_frames, self.num_frames, num_heads, n, c_)
        out = feat.clone() 
        # print(out.shape) # e([16, 8, 12, 197, 64])

        out[:, 1:, :self.num_theads, :, :] = feat[:, :-1, :self.num_theads, :, :]  # shift left
        out[:, :-1, self.num_theads:2*self.num_theads, :, :] = feat[:, 1:, self.num_theads:2*self.num_theads, :, :]  # shift right

        out = out.view(bt, num_heads, n, c_)

        return out


class LinearAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, skip_connect=True, scale=1.):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.scale = scale
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs * self.scale
        else:
            x = xs * self.scale
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True, num_frames=None, xvit_cfg=None, adapter_scale=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.xvit_cfg = xvit_cfg
        self.attn_adapter_type = xvit_cfg.get('attn_adapter_type')
        if self.attn_adapter_type == 'share':
            self.T_Adapter_in = LinearAdapter(dim, scale=adapter_scale, mlp_ratio=0.25)
        elif self.attn_adapter_type == 'qkv':
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.T_Adapter_in_q = LinearAdapter(dim, scale=adapter_scale, mlp_ratio=0.25)
            self.T_Adapter_in_k = LinearAdapter(dim, scale=adapter_scale, mlp_ratio=0.25)
            self.T_Adapter_in_v = LinearAdapter(dim, scale=adapter_scale, mlp_ratio=0.25)
        # elif self.attn_adapter_type == 'qv':
        #     self.T_Adapter_in_q = LinearAdapter(dim, scale=adapter_scale, mlp_ratio=0.25)
        #     self.T_Adapter_in_v = LinearAdapter(dim, scale=adapter_scale, mlp_ratio=0.25)
        else:
            raise NotImplementedError(self.attn_adapter_type)

        self.shift_ops = HeadReallocation(num_frames=num_frames, num_theads=xvit_cfg.get('num_theads'))

    def forward(self, x):
        B, N, C = x.shape
        if self.attn_adapter_type == 'share':
            x = self.T_Adapter_in(x)
            if self.with_qkv:
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, C//num_heads)
                q, k, v = qkv[0], qkv[1], qkv[2]
            else:
                qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                q, k, v  = qkv, qkv, qkv
        elif self.attn_adapter_type == 'qkv':
            q = self.T_Adapter_in_q(x)
            k = self.T_Adapter_in_k(x)
            v = self.T_Adapter_in_v(x) 
            q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.k(k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # elif self.attn_adapter_type == 'qv':
        #     q = self.T_Adapter_in_q(x)
        #     k = x
        #     v = self.T_Adapter_in_v(x)
        else:
            raise NotImplementedError(self.attn_adapter_type)
        

        k = self.shift_ops(k)
        v = self.shift_ops(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x




class Block(nn.Module):

    def __init__(self, dim, num_frames, num_heads, mlp_ratio=4., scale=0.5, num_tadapter=1, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, xvit_cfg=None):
        super().__init__()
        self.num_frames = num_frames
        self.num_tadapter = num_tadapter
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            num_frames=num_frames, xvit_cfg=xvit_cfg, adapter_scale=scale)

        logger = MMLogger.get_current_instance()
        logger.info(f'num_tadapter:{num_tadapter}, scale:{scale} xvit_cfg: {xvit_cfg}')


        self.MLP_Adapter = LinearAdapter(dim, scale=scale, mlp_ratio=0.25)
        if self.num_tadapter == 2:
            self.MLP_Adapter_out = LinearAdapter(dim, scale=scale, mlp_ratio=0.25)
        if self.num_tadapter != -1:
            self.S_Adapter = LinearAdapter(dim, scale=scale, mlp_ratio=0.25)
        

        self.num_frames = num_frames
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        ## x in shape [BT, HW+1, D]
        # bt, n, d = x.shape
        # ## temporal adaptation
        # xt = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        # if self.num_tadapter == 2:
        #     xt = self.T_Adapter(self.attn(self.T_Adapter_in(self.norm1(xt))))
        # else:
        #     xt = self.T_Adapter(self.attn(self.norm1(xt)))
        # xt = rearrange(xt, '(b n) t d ->(b t) n d', n=n)
        # x = x + self.drop_path(xt)
        # ## spatial adaptation
        # x = x + self.S_Adapter(self.attn(self.norm1(x)))
        # ## joint adaptation
        # xn = self.norm2(x)
        # x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
        # return x

        if self.num_tadapter != -1:
            x = x + self.S_Adapter(self.attn(self.norm1(x)))
        else:
            x = x + self.attn(self.norm1(x))
        # joint adaptation
        if self.num_tadapter == 2:
            x = x + self.MLP_Adapter_out(self.mlp(self.MLP_Adapter(self.norm2(x))))
        else:
            x = x + self.mlp(self.MLP_Adapter(self.norm2(x)))
        return x
        
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

@MODELS.register_module()
class ViT_Zero_IN21k(nn.Module):
    def __init__(self, img_size=224, num_frames=8, patch_size=16, in_chans=3, embed_dim=768, depth=12, adapter_scale=0.5, num_tadapter=1,
                 num_heads=12, mlp_ratio=4., patch_embedding_bias=True, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), pretrained=None, xvit_cfg=None):
        super().__init__()
        self.attn_adapter_type = xvit_cfg.get('attn_adapter_type')
        self.qkv_bias = qkv_bias
        self.num_tadapter = num_tadapter
        self.pretrained = pretrained
        self.depth = depth
        self.num_frames = num_frames
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, bias=patch_embedding_bias)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_frames=num_frames, num_heads=num_heads, mlp_ratio=mlp_ratio, scale=adapter_scale, num_tadapter=num_tadapter, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, xvit_cfg=xvit_cfg)
            for i in range(self.depth)])
        self.ln_post = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

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
            logger.info(f'load model from: {self.pretrained}')
            ## Load ImageNet pretrained weights
            state_dict = torch.load(self.pretrained)
            state_dict['ln_post.weight'] = state_dict['norm.weight']
            state_dict['ln_post.bias'] = state_dict['norm.bias']
            msg = self.load_state_dict(state_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        ## initialize Adapter
        for n, m in self.blocks.named_modules():
            if '.qkv' in n and self.attn_adapter_type == 'qkv':
                self.blocks[int(n.split('.')[0])].attn.q.weight.data.copy_(m.weight[:self.embed_dim, :].detach().clone())
                if self.qkv_bias:
                    self.blocks[int(n.split('.')[0])].attn.q.bias.data.copy_(m.bias[:self.embed_dim].detach().clone())
                self.blocks[int(n.split('.')[0])].attn.k.weight.data.copy_(m.weight[self.embed_dim:2*self.embed_dim, :].detach().clone())
                if self.qkv_bias:
                    self.blocks[int(n.split('.')[0])].attn.k.bias.data.copy_(m.bias[self.embed_dim:2*self.embed_dim].detach().clone())
                self.blocks[int(n.split('.')[0])].attn.v.weight.data.copy_(m.weight[2*self.embed_dim:, :].detach().clone())
                if self.qkv_bias:
                    self.blocks[int(n.split('.')[0])].attn.v.bias.data.copy_(m.bias[2*self.embed_dim:].detach().clone())
                logger.info(f"split qkv:'{n}'")
        
            if 'Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        if self.attn_adapter_type == 'qkv':
            for block in self.blocks:
                del block.attn.qkv

        ## freeze some parameters
        for name, param in self.named_parameters():
            if 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name:
                param.requires_grad = False

        for name, param in self.named_parameters():
            logger.info('{}: {}'.format(name, param.requires_grad))
        num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.parameters())
        logger.info('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, num_param))
    

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.patch_embed(x)  # shape = [BT, HW, D]
        x = torch.cat([self.cls_token.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) 
        x = x + self.pos_embed.to(x.dtype)

        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
        
        for blk in self.blocks:
            x = blk(x)  # [BT, HW+1, D]

        x = self.ln_post(x)
        x = x[:, 0]
        x = rearrange(x, '(b t) c -> b c t',b=B,t=T)
        
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        return x