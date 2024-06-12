# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.structures import LabelData
from torch import Tensor

from mmaction.evaluation import top_k_accuracy
from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, LabelList, OptConfigType,
                            OptMultiConfig, SampleList)


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel, BertTokenizer, BertModel


class Bert(nn.Module):
    """ Finetuned DistilBERT module """

    def __init__(self, bert_type="bert-base-uncased"):
        super(Bert, self).__init__()
        # self.bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        # self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert_type = bert_type
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_type)
        self.bert = BertModel.from_pretrained(bert_type)
        self.cls_token = self.bert_tokenizer.cls_token_id
        self.sep_token = self.bert_tokenizer.sep_token_id

    def forward(self, tokens):
        attention_mask = (tokens > 0).float()
        embds = self.bert(tokens, attention_mask=attention_mask)[0]
        return embds



class AnswerModel(nn.Module):
    """
    Answer embedding module
    """

    def __init__(self, out_dim=512, bert_type="bert-base-uncased"):
        super(AnswerModel, self).__init__()
        self.bert = Bert(bert_type=bert_type)
        self.linear_text = nn.Linear(768, out_dim)

    def forward(self, answer):
        if len(answer.shape) == 3:
            bs, nans, lans = answer.shape
            answer = answer.view(bs * nans, lans)
            answer = self.bert(answer)
            answer = answer[:, 0, :]
            answer = answer.view(bs, nans, 768)
        else:
            answer = self.bert(answer)
            answer = answer[:, 0, :]
        answer = self.linear_text(answer)
        return answer



class AnswerHead(BaseModule, metaclass=ABCMeta):
    """Head for Video QuestionAnswer.
    """

    def __init__(self,
                 in_channels: int,
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: OptMultiConfig = None) -> None:
        super(AnswerHead, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.loss_cls = MODELS.build(loss_cls)
        
        # answer modules
        self.answer_model = AnswerModel(out_dim=in_channels)
        self.answer_embeddings = None
        
    def get_answer_embedding(self, answer):
        answer = self.answer_model(answer)
        return answer
        
    def _compute_answer_embedding(self, a2v):
        self.answer_embeddings = self.get_answer_embedding(a2v)

    
    def forward(self, fusion_proj, answer) -> Tensor:
        """Defines the computation performed at every call."""
                # 下面是head部分，上面是neck，负责把fusion_proj和answer传下来就行了
        answer_proj = (
            self.get_answer_embedding(answer)
            if answer is not None
            else self.answer_embeddings
        )
        if question is not None and answer_proj.device != question.device:
            answer_proj = answer_proj.to(question.device)
        if answer is not None:
            return fusion_proj, answer_proj
        else:
            return fusion_proj @ answer_proj.t()

    def loss(self, feats: Union[Tensor, Tuple[Tensor]],
             data_samples: SampleList, **kwargs) -> dict:
        """Perform forward propagation of head and loss calculation on the
        features of the upstream network.

        Args:
            feats (Tensor or Tuple[Tensor]): Features from upstream network.
            data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        cls_scores = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples)

    def loss_by_feat(self, cls_scores: Union[Tensor, Tuple[Tensor]],
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
        if self.label_smooth_eps != 0:
            if cls_scores.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_scores, labels)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses

    def predict(self, feats: Union[Tensor, Tuple[Tensor]],
                data_samples: SampleList, **kwargs) -> LabelList:
        """Perform forward propagation of head and predict recognition results
        on the features of the upstream network.

        Args:
            feats (Tensor or Tuple[Tensor]): Features from upstream network.
            data_samples (List[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        cls_scores = self(feats, **kwargs)
        return self.predict_by_feat(cls_scores, data_samples)


