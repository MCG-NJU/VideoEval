# Copyright (c) OpenMMLab. All rights reserved.
from .aagcn import AAGCN
from .c2d import C2D
from .c3d import C3D
from .Harmful_Contentilenet_v2 import Harmful_ContentileNetV2
from .Harmful_Contentilenet_v2_tsm import Harmful_ContentileNetV2TSM
from .mvit import MViT
from .resnet import ResNet
from .resnet2plus1d import ResNet2Plus1d
from .resnet3d import ResNet3d, ResNet3dLayer
from .resnet3d_csn import ResNet3dCSN
from .resnet3d_slowfast import ResNet3dSlowFast
from .resnet3d_slowonly import ResNet3dSlowOnly
from .resnet_audio import ResNetAudio
from .resnet_omni import OmniResNet
from .resnet_tin import ResNetTIN
from .resnet_tsm import ResNetTSM
from .stgcn import STGCN
from .swin import SwinTransformer3D
from .swin_tps import SwinTransformer2D_TPS
from .swin_sifa import SwinTransformer2D_SIFA
from .swin_refiner import SwinTransformer2D_Refiner
from .swin_ifa import SwinTransformer2D_IFA
from .swin_bra import SwinTransformer2D_BRA
from .tanet import TANet
from .timesformer import TimeSformer, TimeSformerWithAdaptiveSampler
from .uniformer import UniFormer
from .uniformerv2 import UniFormerV2
from .vit_mae import VisionTransformer
from .x3d import X3D
from .convnext import ConvNeXt
from .convnext3d import ConvNeXt3d
from .conv2former import Conv2Former
from .conv2former_tsm import Conv2FormerTSM
from .conv3former import Conv3Former
from .conv4former import Conv4Former
from .conv2former3d import Conv2Former3d
from .tdn import TDN, ResNetLTDM, Conv2FormerLTDM
from .morphmlp import MorphMLP
from .resnet_tsm_selfy import ResNetSELFY
from .c2d_resnet_sifa import C2D_ResNet_SIFA
from .vit_aim_clip import ViT_AIM_CLIP
from .vit_stadapter_clip import ViT_ST_Adapter_CLIP
from .vit_motion_clip import ViT_Motion_CLIP
from .vit_dualpath_clip import ViT_DualPath_CLIP
from .vit_shift_clip import ViT_Shift_CLIP
from .vit_outlook_clip import ViT_Outlook_CLIP
from .vit_tdn_clip import ViT_TDN_CLIP
from .vit_linear_clip import ViT_Linear_CLIP
from .vit_mlp_clip import ViT_MLP_CLIP
from .vit_bypass_clip import ViT_Bypass_CLIP
from .vit_repadapter_clip import ViT_RepAdapter_CLIP
from .vit_xvit_clip import ViT_XViT_CLIP
from .vit_zero_clip import ViT_Zero_CLIP
from .vit_zero_clip_eval import ViT_Zero_CLIP_eval
from .vit_zero_clip_baseline import ViT_Zero_CLIP_baseline
from .vit_zero_clip_ablation import ViT_Zero_CLIP_ablation
from .vit_zero_clip_lora import ViT_Zero_CLIP_LoRA
from .vit_zero_clip_shift import ViT_Zero_CLIP_shift
from .vit_zero_clip_one import ViT_Zero_CLIP_one
from .vit_zero_clip_tokenshift import ViT_Zero_CLIP_token_shift
from .vit_zero_in21k import ViT_Zero_IN21k
from .vit_evl_clip import ViT_EVL_CLIP
from .swin_zero import SwinTransformer2D_Zero
from .swin_stdha import SwinTransformer2D_STDHA
from .swin_zero_fix import SwinTransformer2D_Zero_fix
from .vit_zero_clip_debug import ViT_Zero_CLIP_Debug
from .vit_clip import ViT_CLIP
from .vit_zero_clip_v2 import ViT_Zero_CLIP_v2
from .vit_zero_clip_epoch import ViT_Zero_CLIP_epoch
from .vit_zero_clip_rep import ViT_Zero_CLIP_rep
from .vit_zero_clip_rank import ViT_Zero_CLIP_rank
from .vit_lyz import ViT_lyz
from .vit_lyz_divided import ViT_lyz_divided

__all__ = [
    'AAGCN', 'C2D', 'C3D', 'MViT', 'Harmful_ContentileNetV2', 'Harmful_ContentileNetV2TSM',
    'OmniResNet', 'ResNet', 'ResNet2Plus1d', 'ResNet3d', 'ResNet3dCSN',
    'ResNet3dLayer', 'ResNet3dSlowFast', 'ResNet3dSlowOnly', 'ResNetAudio',
    'ResNetTIN', 'ResNetTSM', 'STGCN', 'SwinTransformer3D', 'TANet',
    'TimeSformer', 'TimeSformerWithAdaptiveSampler',
    'VisionTransformer', 'X3D', 'ConvNeXt', 'ConvNeXt3d',
    'UniFormer', 'UniFormerV2', 'Conv2Former', 'Conv2FormerTSM',
    'TDN', 'ResNetLTDM', 'Conv2FormerLTDM', 'Conv3Former', 'Conv2Former3d',
    'SwinTransformer2D_TPS', 'SwinTransformer2D_Refiner', 'Conv4Former', 'MorphMLP',
    'ResNetSELFY', 'SwinTransformer2D_SIFA', 'C2D_ResNet_SIFA', 'SwinTransformer2D_IFA',
    'SwinTransformer2D_BRA', 'ViT_AIM_CLIP', 'ViT_ST_Adapter_CLIP', 'ViT_Motion_CLIP',
    'ViT_DualPath_CLIP', 'ViT_Shift_CLIP', 'ViT_Outlook_CLIP', 'ViT_TDN_CLIP', 'ViT_Linear_CLIP',
    'ViT_MLP_CLIP', 'ViT_Bypass_CLIP', 'ViT_RepAdapter_CLIP', 'ViT_XViT_CLIP', 'ViT_Zero_CLIP',
    'ViT_Zero_CLIP_eval', 'ViT_Zero_CLIP_baseline', 'ViT_Zero_CLIP_ablation', 'ViT_Zero_CLIP_LoRA',
    'ViT_Zero_CLIP_shift', 'ViT_Zero_CLIP_one', 'ViT_Zero_CLIP_token_shift', 'ViT_Zero_IN21k', 'ViT_EVL_CLIP',
    'SwinTransformer2D_Zero_fix', 'SwinTransformer2D_STDHA', 'ViT_lyz'
]
