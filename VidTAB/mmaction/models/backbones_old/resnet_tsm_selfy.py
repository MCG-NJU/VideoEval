# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from typing import Tuple, Union
from mmaction.registry import MODELS
from .resnet_tsm import ResNetTSM

import torch.nn as nn
import torch.nn.functional as F
import torch as tr
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
try:
    from spatial_correlation_sampler import SpatialCorrelationSampler
except Exception as e:
    print(e)
    print("如果不用selfy可以不用管")




class STSSTransformation(nn.Module):
    def __init__(self, d_in, d_hid, num_segments, window=(5,9,9), use_corr_sampler=False):
        super(STSSTransformation, self).__init__()
        self.num_segments = num_segments
        self.window = window
        assert window[1] == window[2]
        self.use_corr_sampler = use_corr_sampler
        
        self.use_corr_sampler = use_corr_sampler
        if use_corr_sampler:
            try:
                from spatial_correlation_sampler import SpatialCorrelationSampler
                pass # TODO 先不用了
                # self.correlation_sampler = SpatialCorrelationSampler(1, window[1], 1, 0, 1)
            except:
                print("[Warning] SpatialCorrelationSampler cannot be used.")
                
            
        # Resize spatial resolution to 14x14
        self.downsample = nn.Sequential(
            nn.Conv2d(d_in, d_hid, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_hid),
            nn.ReLU(inplace=True)
        )
        
    def _L2normalize(self, x, d=1):
        eps = 1e-6
        norm = x ** 2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        
        return (x / norm)

    def _corr_abs_to_rel(self, corr, h, w):
        # Naive implementation of spatial correlation sampler
        
        max_d = self.window[1] // 2

        b,c,s = corr.size()        
        corr = corr.view(b,h,w,h,w)

        w_diag = tr.zeros((b,h,h,self.window[1],w),device='cuda')
        for i in range(max_d+1):
            if (i==0):
                w_corr_offset = tr.diagonal(corr, offset=0, dim1=2, dim2=4)       
                w_diag[:,:,:,max_d] = w_corr_offset
            else:
                w_corr_offset_pos = tr.diagonal(corr, offset=i, dim1=2, dim2=4) 
                w_corr_offset_pos = F.pad(w_corr_offset_pos, (i,0))
                w_diag[:,:,:,max_d-i] = w_corr_offset_pos
                w_corr_offset_neg = tr.diagonal(corr, offset=-i, dim1=2, dim2=4) 
                w_corr_offset_neg = F.pad(w_corr_offset_neg, (0,i))
                w_diag[:,:,:,max_d+i] = w_corr_offset_neg

        hw_diag = tr.zeros((b,self.window[1],w,self.window[1],h), device='cuda') 
        for i in range(max_d+1):
            if (i==0):
                h_corr_offset = tr.diagonal(w_diag, offset=0, dim1=1, dim2=2)
                hw_diag[:,:,:,max_d] = h_corr_offset
            else:
                h_corr_offset_pos = tr.diagonal(w_diag, offset=i, dim1=1, dim2=2) 
                h_corr_offset_pos = F.pad(h_corr_offset_pos, (i,0))
                hw_diag[:,:,:,max_d-i] = h_corr_offset_pos
                h_corr_offset_neg = tr.diagonal(w_diag, offset=-i,dim1=1, dim2=2) 
                h_corr_offset_neg = F.pad(h_corr_offset_neg, (0,i))     
                hw_diag[:,:,:,max_d+i] = h_corr_offset_neg 

        hw_diag = hw_diag.permute(0,3,1,4,2).contiguous()
        hw_diag = hw_diag.view(-1, self.window[1], self.window[1], h, w)      

        return hw_diag         

    def _correlation(self, feature1, feature2):
        feature1 = self._L2normalize(feature1) # btl, c, h, w
        feature2 = self._L2normalize(feature2) # btl, c, h, w

        if self.use_corr_sampler:
            corr = self.correlation_sampler(feature1, feature2)
        else:
            b, c, h, w = feature1.size()
            feature1 = rearrange(feature1, 'b c h w -> b c (h w)')
            feature2 = rearrange(feature2, 'b c h w -> b c (h w)')
            corr = tr.einsum('bcn,bcm->bnm',feature2, feature1)
            corr = self._corr_abs_to_rel(corr, h, w)

        return corr
        
    def forward(self, x):
        # resize spatial resolution to 14x14
        x = self.downsample(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.num_segments)
        
        x_pre = repeat(x, 'b t c h w -> (b t l) c h w', l=self.window[0])
        x_post = F.pad(x, (0,0,0,0,0,0,self.window[0]//2,self.window[0]//2), 'constant', 0).unfold(1,self.window[0],1)
        x_post = rearrange(x_post, 'b t c h w l -> (b t l) c h w')     
        
        stss = self._correlation(x_pre, x_post)   
        stss = rearrange(stss, '(b t l) u v h w -> b t h w 1 l u v', t=self.num_segments, l=self.window[0])
        
        return stss

    
class STSSExtraction(nn.Module):
    def __init__(self, num_segments, window=(5,9,9), chnls=(4,16,64,64)):
        super(STSSExtraction, self).__init__()
        self.num_segments = num_segments
        self.window = window
        self.chnls = chnls
        
        self.conv0 = nn.Sequential(
            nn.Conv3d(1, chnls[0], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[0]),
            nn.ReLU(inplace=True))
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(chnls[0], chnls[1], kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[1]),
            nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(chnls[1], chnls[2], kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[2]),
            nn.ReLU(inplace=True))
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(chnls[2], chnls[3], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,0,0), bias=False),
            nn.BatchNorm3d(chnls[3]),
            nn.ReLU(inplace=True))    
        
    def forward(self, x):
        b,t,h,w,_,l,u,v = x.size()
        x = rearrange(x, 'b t h w 1 l u v -> (b t h w) 1 l u v')
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = rearrange(x, '(b t h w) c l 1 1 -> (b l) c t h w', t=t, h=h, w=w)
        
        return x
    
    
class STSSIntegration(nn.Module):
    def __init__(self, d_in, d_out, num_segments, window=(5,9,9), chnls=(64,64,64,64)):
        super(STSSIntegration, self).__init__()
        self.num_segments = num_segments
        self.window = window
        self.chnls = chnls
        
        self.conv0 = nn.Sequential(
            nn.Conv3d(d_in, chnls[0], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[0]),
            nn.ReLU(inplace=True))
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(chnls[0], chnls[1], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[1]),
            nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(chnls[1], chnls[2], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[2]),
            nn.ReLU(inplace=True))
        
        self.conv3_fuse = nn.Sequential(
            Rearrange('(b l) c t h w -> b (l c) t h w', l=self.window[0]),
            nn.Conv3d(chnls[2]*self.window[0], chnls[3], kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(chnls[3]),
            nn.ReLU(inplace=True)
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(chnls[3], d_out, kernel_size=1, stride=(1,2,2), padding=(0,0,0), output_padding=(0,1,1), bias=False),
            nn.BatchNorm3d(d_out),
            Rearrange('b c t h w -> (b t) c h w')
        )
        
        
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3_fuse(x)
        x = self.upsample(x)
        
        return x
    
    
class SELFYBlock(nn.Module):
    def __init__(self,
                 d_in,
                 d_hid,
                 num_segments=8,
                 window=(5,9,9),
                 ext_chnls=(4,16,64,64),
                 int_chnls=(64,64,64,64)
                ):
        super(SELFYBlock, self).__init__()
        
        self.stss_transformation = STSSTransformation(
            d_in,
            d_hid,
            num_segments=num_segments,
            window=window,
        )
        
        self.stss_extraction = STSSExtraction(
            num_segments=num_segments,
            window = window,
            chnls = ext_chnls
        )
        
        self.stss_integration = STSSIntegration(
            ext_chnls[-1],
            d_in,
            num_segments=num_segments,
            window = window,
            chnls = int_chnls
        )
        
        
    def forward(self, x):
        identity = x
        out = self.stss_transformation(x)
        out = self.stss_extraction(out)
        out = self.stss_integration(out)
        
        out = out + identity
        out = F.relu(out)
        
        return out
    

@MODELS.register_module()
class ResNetSELFY(ResNetTSM):
    def __init__(self,
                 depth,
                 num_segments,
                 use_selfy=True,
                 selfy_window=[5, 9, 9],
                 selfy_ext_chnls=[4,16,64,64],
                 selfy_int_chnls=[64,64,64,64],
                 zero_init_residual=True,
                 **kwargs):
        super().__init__(depth, num_segments=num_segments, **kwargs)
        self.use_selfy = use_selfy
        if use_selfy: 
            self.selfy = SELFYBlock(
                128*4,
                128,
                num_segments=num_segments,
                window=selfy_window,
                ext_chnls=selfy_ext_chnls,
                int_chnls=selfy_int_chnls
            )
            for m in self.selfy.modules():       
                if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            
            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if zero_init_residual: 
                # 只对SELFY模块做初始化，因为其它的已经加载预训练模型了
                nn.init.constant_(self.selfy.stss_integration.upsample[-2].weight, 0)
    

    def forward(self, x: torch.Tensor) \
            -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            Union[torch.Tensor or Tuple[torch.Tensor]]: The feature of the
                input samples extracted by the backbone.
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if self.use_selfy and i == 1:
                x = self.selfy(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)