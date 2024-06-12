# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model.weight_init import normal_init
from torch import Tensor, nn
# import torch.utils.checkpoint as cp
import json

from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from .base import BaseHead


@MODELS.register_module()
class PretrainedHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 spatial_type: str = 'avg',
                 dropout_ratio: float = 0.5,
                 init_std: float = 0.01,
                 must_use_fp32: bool=False, # nqa lxh
                 with_cp=False,
                 pretrained=None, # nqa lxh
                 pretrained_type='k710',
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.must_use_fp32 = must_use_fp32
        self.with_cp = with_cp

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None
        self.pretrained = pretrained
        self.pretrained_type = pretrained_type

        if self.pretrained is not None:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            if self.pretrained_type == 'k710':
                state_dict = torch.load(self.pretrained, map_location='cpu')
                logger.info(f'Load pretrained cls head from {self.pretrained}')
                new_state_dict = {}
                if 'state_dict' in state_dict.keys():
                    state_dict = state_dict['state_dict']
                if self.num_classes == 400:
                    new_state_dict['fc_cls.weight'] = state_dict['cls_head.fc_cls.weight'][:400]
                    new_state_dict['fc_cls.bias'] = state_dict['cls_head.fc_cls.bias'][:400]
                elif self.num_classes == 600 or self.num_classes == 700:
                    map_path = f'yourpath/mix_kinetics_new/label_mixto{num_classes}.json'
                    logger.info(f'Load label map from {map_path}')
                    with open(map_path) as f:
                        label_map = json.load(f)
                    new_state_dict['fc_cls.weight'] = state_dict['cls_head.fc_cls.weight'][label_map]
                    new_state_dict['fc_cls.bias'] = state_dict['cls_head.fc_cls.bias'][label_map]
                else:
                    raise NotImplementedError(f"Not support num_classes: {self.num_classes}!")
                self.load_state_dict(new_state_dict, strict=True)
            else:
                raise NotImplementedError(f"Not support pretrained_type: {self.pretrained_type}!")
        else:
            normal_init(self.fc_cls, std=self.init_std)
            
    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        pass

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
  
        if self.must_use_fp32:
            raise NotImplementedError
            with torch.cuda.amp.autocast(dtype=torch.float32):
                cls_score = self.fc_cls(x.to(torch.float32))
        else:
            cls_score = self.fc_cls(x)
        # [N, num_classes]

        return cls_score
