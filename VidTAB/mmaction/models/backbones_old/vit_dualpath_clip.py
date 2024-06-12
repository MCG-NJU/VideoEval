from collections import OrderedDict
import torch
from torch import nn
import math
from torchvision import transforms
from collections import OrderedDict
import torch
from torch import nn
import clip
from mmengine.logging import MMLogger
from einops import rearrange
from mmaction.registry import MODELS

class TemporalPositionalEmbedding(nn.Module):
    def __init__(self, channels):
        super(TemporalPositionalEmbedding, self).__init__()
        self.channels = channels
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    def forward(self, tensor):
        if len(tensor.shape) == 3:
            batch_size, N, channels = tensor.shape
            resize_shape = int(math.sqrt(N-1))
            num_frame_per_width = 4
            width = int(16 / num_frame_per_width)    # resized frame size
            temp = torch.zeros(batch_size, 16, width, width).to(tensor.device)
            for i in range(num_frame_per_width ** 2):
                temp[:, i, :, :] = i + 1
            temp = temp.reshape(batch_size, num_frame_per_width, num_frame_per_width, width, width)
            temp = temp.permute(0, 1, 3, 2, 4).reshape(batch_size, 16, 16)
            resize = transforms.Resize((resize_shape, resize_shape))
            temp = resize(temp)
            emb = temp.view(batch_size, -1)[0]
            emb = torch.cat([torch.tensor([0.0]).view(1).to(tensor.device), emb])
            emb = torch.einsum("i,j->ij", emb, self.inv_freq.to(tensor.device))   # [N, D]
            emb = torch.stack((emb.sin(), emb.cos()), dim = -1)
            emb = torch.flatten(emb, -2, -1)
            return emb.repeat(batch_size, 1, 1)
        else:
            batch_size, Tt, N, channels = tensor.shape
            resize_shape = int(math.sqrt(N-1))
            # emb = torch.zeros(batch_size, N, channels)
            num_frame_per_width = 4
            width = int(16 / num_frame_per_width)    # resized frame size
            temp = torch.zeros(batch_size, 16 * Tt, width, width).to(tensor.device)
            for i in range((num_frame_per_width ** 2) * Tt):
                temp[:, i, :, :] = i + 1
            temp = temp.reshape(batch_size, Tt, num_frame_per_width, num_frame_per_width, width, width)
            temp = temp.permute(0, 1, 2, 4, 3, 5).reshape(batch_size, Tt, 16, 16)
            resize = transforms.Resize((resize_shape, resize_shape))
            temp = resize(temp) # [B, Tt, root(N), root(N)]
            emb = temp.view(batch_size, Tt, -1)[0]  # [B, Tt, N]
            emb = torch.cat([torch.tensor([[0.0]]*Tt).to(tensor.device), emb], dim = 1) #[Tt, N]
            emb = emb.view(-1)  # [TtxN]
            emb = torch.einsum("i,j->ij", emb, self.inv_freq.to(tensor.device))   # [N, D/2]
            emb = torch.stack((emb.sin(), emb.cos()), dim = -1)
            emb = torch.flatten(emb, -2, -1).reshape(Tt, N, -1)
            return emb.repeat(batch_size, 1, 1, 1)


class Adapter(nn.Module):
    def __init__(self,
                 d_model: int = 768,
                 bottleneck: int = 128,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="out"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.GELU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
        else:
            raise NotImplementedError

    def forward(self, x, add_residual=False, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

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
    def __init__(self, d_model: int, n_head: int, num_frames: int, attn_mask: torch.Tensor = None, num_t_adapters: int = 2):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.num_frames = num_frames
        self.num_t_adapters = num_t_adapters
        
        if num_t_adapters == 2:
            self.t_adapter_attn_b = Adapter(d_model=d_model, bottleneck=128, dropout=0.1, adapter_layernorm_option=None)
        self.s_adapter_attn = Adapter(d_model=d_model, bottleneck=128, dropout=0.1, adapter_layernorm_option=None)
        self.t_adapter_attn = Adapter(d_model=d_model, bottleneck=128, dropout=0.1, adapter_layernorm_option=None)
        self.s_adapter_mlp = Adapter(d_model=d_model, bottleneck=128, dropout=0.1, adapter_layernorm_option=None)
        self.t_adapter_mlp = Adapter(d_model=d_model, bottleneck=128, dropout=0.1, adapter_layernorm_option=None)

        # self.t_adapter_attn_b = Adapter(d_model=d_model, bottleneck=int(0.25 * d_model), dropout=0.1, adapter_layernorm_option=None)
        # self.s_adapter_attn = Adapter(d_model=d_model, bottleneck=int(0.25 * d_model), dropout=0.1, adapter_layernorm_option=None)
        # self.t_adapter_attn = Adapter(d_model=d_model, bottleneck=int(0.25 * d_model), dropout=0.1, adapter_layernorm_option=None)
        # self.s_adapter_mlp = Adapter(d_model=d_model, bottleneck=int(0.25 * d_model), dropout=0.1, adapter_layernorm_option=None)
        # self.t_adapter_mlp = Adapter(d_model=d_model, bottleneck=int(0.25 * d_model), dropout=0.1, adapter_layernorm_option=None)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        T = self.num_frames
        Ts = 8
        Tt = T - Ts
        B, N, D = x.shape[1] // (T), x.shape[0], x.shape[2]
        x_t_residual = x.reshape(N, B, T, D)[:, :, 0:Tt, :].reshape(N, -1, D)
        x_s_residual = x.reshape(N, B, T, D)[:, :, Tt:, :].reshape(N, -1, D)
        x = self.ln_1(x)
        x = x.reshape(N, B, T, D)
        x_s = x[:, :, Tt:, :].reshape(N, -1, D)  # [N+1, B*Ts, D]
        x_t = x[:, :, 0:Tt, :].reshape(N, -1, D) # [N+1, B*Tt, D]
        if self.num_t_adapters == 2:
            x_t = self.t_adapter_attn_b(x_t, add_residual=False)
        x_t = self.attention(x_t)
        x_t = self.t_adapter_attn(x_t)
        x_t = x_t + x_t_residual
        x_t_residual2 = x_t
        x_t = self.ln_2(x_t)
        x_t = self.mlp(x_t)
        x_t = self.t_adapter_mlp(x_t) + x_t_residual2

        x_s_adapt = self.s_adapter_attn(x_s)
        x_s = self.attention(x_s) + x_s_adapt + x_s_residual
        x_s_residual2 = x_s
        x_s = self.ln_2(x_s)
        x_s_mlp = self.s_adapter_mlp(x_s)
        x_s = self.mlp(x_s) + x_s_mlp + x_s_residual2
        x = torch.cat([x_t.reshape(N, B, -1, D), x_s.reshape(N, B, -1, D)], dim=2).reshape(N, -1, D)
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, num_frames: int, attn_mask: torch.Tensor = None, num_t_adapters: int=2):
        super().__init__()
        self.width = width
        self.layers = layers
        self.num_t_adapters = num_t_adapters
        self.num_frames = num_frames
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, num_frames, attn_mask, num_t_adapters=num_t_adapters) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

@MODELS.register_module()
class ViT_DualPath_CLIP(nn.Module):
    def __init__(self,
                input_resolution: int,
                patch_size: int,
                width: int,
                layers: int,
                heads: int,
                output_dim: int,
                num_frames: int,
                num_t_adapters: int=2,
                pretrained: str=None,
                ):
        super().__init__()
        # adapter_list = [None, 'w-adapter', 'protuning', 'vpt', 'st-adapter', 'adaptformer', 'aim']
        # if adapter not in adapter_list:
        #     raise ValueError("Warning: Check adapt method!")
        self.pretrained = pretrained
        assert num_frames == 32, "目前只支持32帧的输入！"
        self.num_frames = num_frames // 16 + 8 # NOTE 因为rearrange和dualpath，需要进行转换

        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, num_frames=self.num_frames, attn_mask=None, num_t_adapters=num_t_adapters)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


        self.head = nn.Sequential(OrderedDict([
          ('linear1', nn.Linear(output_dim * 2, output_dim)),
          ('gelu', nn.GELU()) # 留一个给I3D发挥
        #   ('linear2', nn.Linear(output_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()),
        ]))
            # nn.Linear(output_dim * 2, num_classes) if num_classes > 0 else nn.Identity()

        # trunc_normal_(self.head.weight, std=.02)

        self.num_t_adapters = num_t_adapters
        

        self.temporal_positional_embedding = TemporalPositionalEmbedding(width)
        self.spatial_positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))


    def prepare_data(self, samples: torch.Tensor):
        # x is (B, C, T, H, W)
        B, C, T, H, W = samples.shape
        assert T == (self.num_frames - 8) * 16, f"num_frames={self.num_frames}不正确，T={T}"
        block_width = 4
        num_temporal_frame = int(T / (block_width ** 2))
        spatial_stride = int(T / 8)
        resize = transforms.Resize((H, W))
        samples_t = samples
        samples_s = samples[:, :, 0::spatial_stride, :, :]

        samples_t = samples_t.reshape(B, C, num_temporal_frame, int(block_width ** 2), H, W)
        samples_t = samples_t.reshape(B, C, num_temporal_frame, block_width, block_width, H, W)
        samples_t = samples_t.permute(0, 1, 2, 3, 5, 4, 6)
        samples_t = samples_t.reshape(B * C * num_temporal_frame, block_width * H, block_width * W)
        samples_t = resize(samples_t).reshape(B, C, num_temporal_frame, H, W)

        return torch.cat([samples_t, samples_s], dim=2)


    def forward(self, x: torch.Tensor):
        x = self.prepare_data(x).clone().detach() #  B C T H W, T = Tt + Ts
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        GB, N, D = x.shape
        Ts = 8
        Tt = int(self.num_frames - Ts)
        x = x.reshape(int(GB / self.num_frames), self.num_frames, N, D)

        # with torch.cuda.amp.autocast(dtype=torch.float32): # maybe it makes trainning more stable
        x[:, 0:Tt, :, :] = x[:, 0:Tt, :, :] + self.temporal_positional_embedding(x[:, 0:Tt, :, :]) + self.spatial_positional_embedding.expand(int(GB / self.num_frames), Tt, -1, -1)
        x[:, Tt:, :, :] = (x[:, Tt:, :, :].reshape(-1, N, D) + self.positional_embedding.to(x.dtype)).reshape(int(GB / self.num_frames), 8, N, D)
        x = x.reshape(-1, N, D)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD    [B*T, N+1, D]


        # x_temp = x[:, 1:, :]    # [B*T, N, D]
        x = self.ln_post(x[:, 0, :])    # [B*T, 1, D]

        if self.proj is not None:
            x = x @ self.proj


        x = x.reshape(-1, Ts + Tt, x.shape[-1])

        # x_temp = x_temp.reshape(-1, Ts + Tt, x_temp.shape[1], x_temp.shape[-1]) # [B, T, N, D]
        # x_temp = x_temp[:, 0:Tt, :, :].reshape(-1, x_temp.shape[2], x_temp.shape[-1])  # [B * Tt, N, D]
        # cls_t = x[:, 0:Tt, :].reshape(-1, 1, cls_t.shape[-1])   # [B * Tt, D]
        # attn_t = torch.bmm(x_temp, cls_t.permute(0, 2, 1)).reshape(-1, Tt, x_temp.shape[1])  # [B * Tt, N]
        # attn_t = attn_t.reshape(-1, Tt, int(math.sqrt(attn_t.shape[1])), int(math.sqrt(attn_t.shape[1])))

        x = torch.cat([x[:, 0:Tt, :].mean(1), x[:, Tt:, :].mean(1)], dim=-1)

        out = self.head(x)

        return out

    def init_weights(self):
        if isinstance(self.pretrained, str):
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
            # del pretrain_dict['proj']
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

        # adapter自己初始化了
        # freeze some parameters
        for name, param in self.named_parameters():
            if 'class_embedding' in name or 'spatial_positional_embedding' in name or 'positional_embedding'  in name:
                param.requires_grad = True
            elif name in msg.missing_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, param in self.named_parameters():
            logger.info('{}: {}'.format(name, param.requires_grad))
        num_param = sum(p.numel()
                        for p in self.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.parameters())
        logger.info('Number of total parameters: {}, tunable parameters: {}'.format(
            num_total_param, num_param))








