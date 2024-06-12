# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.structures import LabelData
from torch import Tensor
from mmengine.model.weight_init import normal_init
from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from mmaction.evaluation import top_k_accuracy
from mmaction.utils import (ConfigType, LabelList, OptConfigType,
                            OptMultiConfig, SampleList)
from .base import BaseHead

from typing import Tuple, Union



@MODELS.register_module()
class DualI3DHead(BaseHead):
    """Classification head for I3D with dual fc.

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
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
            self.dropout_dual = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
            self.dropout_dual = None

        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.fc_cls_dual = nn.Linear(self.in_channels, self.num_classes)
        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

        # self.pool = pooling_arch(input_dim=512 * block[0].expansion)
        # self.pool_dual = pooling_arch(input_dim=512 * block[0].expansion)

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)
        normal_init(self.fc_cls_dual, std=self.init_std)

    def forward(self, two_x: Tensor, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        x, x_dual = two_x[0], two_x[1]

        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
            x_dual = self.avg_pool(x_dual)

        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
            x_dual = self.dropout_dual(x_dual)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        x_dual = x_dual.view(x_dual.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        cls_score_dual = self.fc_cls_dual(x_dual)
        # [N, num_classes]
        return [cls_score, cls_score_dual]


    def loss_by_feat(self, two_cls_scores: Union[Tensor, Tuple[Tensor]],
                     data_samples: SampleList) -> dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        cls_scores, cls_scores_dual = two_cls_scores[0], two_cls_scores[1]
        labels = [x.gt_labels.item for x in data_samples]
        labels = torch.stack(labels).to(cls_scores.device)
        labels = labels.squeeze()

        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if cls_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_scores.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_scores.device)

            top_k_acc_dual = top_k_accuracy(cls_scores_dual.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc_dual):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_scores_dual.device)

        if self.label_smooth_eps != 0:
            if cls_scores.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_scores, labels)
        loss_cls_dual = self.loss_cls(cls_scores_dual, labels)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        if isinstance(loss_cls_dual, dict):
            for k in loss_cls_dual.keys():
                losses[k+'_dual'] = loss_cls_dual[k]
        else:
            losses['loss_cls_dual'] = loss_cls_dual # 只要key里面带loss最后都会被累加损失
        return losses

    def predict_by_feat(self, two_cls_scores: Tensor,
                        data_samples: SampleList) -> LabelList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (Tensor): Classification scores, has a shape
                    (num_classes, )
            data_samples (List[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_labels`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        cls_scores = two_cls_scores[0] + two_cls_scores[1] # 直接加，我感觉除不除2都无所谓吧
        num_segs = cls_scores.shape[0] // len(data_samples)
        cls_scores = self.average_clip(cls_scores, num_segs=num_segs)
        pred_labels = cls_scores.argmax(dim=-1, keepdim=True).detach()

        for data_sample, score, pred_lable in zip(data_samples, cls_scores,
                                                  pred_labels):
            prediction = LabelData(item=score)
            pred_label = LabelData(item=pred_lable)
            data_sample.pred_scores = prediction
            data_sample.pred_labels = pred_label
        return data_samples