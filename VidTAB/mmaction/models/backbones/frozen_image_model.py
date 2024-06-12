from collections import OrderedDict
from mmengine.model.weight_init import trunc_normal_
import torch
from torch import nn
import clip
from mmengine.logging import MMLogger
from einops import rearrange
from mmaction.registry import MODELS
from transformers import AutoModel
from .attentive_pooler import AttentivePooler
from .eva_clip import create_model


@MODELS.register_module()
class Frozen_Image_Model(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, image_model_name: str, num_frames: int):
        super().__init__()
        
        self.num_frames = num_frames
        self.image_model_name = image_model_name

        if image_model_name in ['siglip-so400m-patch14-384', 'clip-vit-large-patch14']:
            self.image_model = AutoModel.from_pretrained(f"/mnt/petrelfs/share_data/lixinhao/models/{image_model_name}").vision_model
            self.width = self.image_model.config.hidden_size
            self.heads = self.image_model.config.num_attention_heads
        elif image_model_name in ['dinov2-large', 'dinov2-giant']:
            self.image_model = AutoModel.from_pretrained(f"/mnt/petrelfs/share_data/lixinhao/models/{image_model_name}")
            self.width = self.image_model.config.hidden_size
            self.heads = self.image_model.config.num_attention_heads
        elif image_model_name in ["EVA01-CLIP-g-14-plus"]:
            self.image_model = create_model(model_name=image_model_name, pretrained="yourpath/EVA/EVA01_CLIP_g_14_plus_psz14_s11B.pt", force_custom_clip=True).visual
            del self.image_model.head
            self.width = 1408
            self.heads = 16
        else:
            raise NotImplementedError(image_model_name)

        self.temporal_embedding = nn.Parameter(torch.zeros(1, self.num_frames, 1, self.width))
            
        self.attentive_pooler = AttentivePooler(
            num_queries=1,
            embed_dim=self.width,
            num_heads=self.heads,
            mlp_ratio=4.0,
            depth=1
        )

        
    def init_weights(self):


        logger = MMLogger.get_current_instance()

        ## freeze some parameters
        for name, param in self.named_parameters():
            param.requires_grad = True
            if 'temporal_embedding' not in name and 'ln_post' not in name and 'attentive_pooler' not in name:
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

        if self.image_model_name in ['siglip-so400m-patch14-384']:
            x = self.image_model(pixel_values=x, interpolate_pos_encoding=True)['last_hidden_state']
        elif self.image_model_name in ["EVA01-CLIP-g-14-plus"]:
            x = self.image_model.forward_features(x, return_all_features=True)
        else:
            x = self.image_model(pixel_values=x)['last_hidden_state']
        
        x = rearrange(x, '(b t) n d -> b t n d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, 'b t n d -> b (t n) d')

        x = self.attentive_pooler(x)
        x = x[:, 0, :]

        x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # for I3D head

        return x
