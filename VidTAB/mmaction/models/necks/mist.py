# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModel
from mmengine.model.weight_init import constant_init, normal_init, xavier_init

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, OptConfigType, SampleList

from ..common import ISTA

class EncoderVid(nn.Module):
    def __init__(self, feat_dim, bbox_dim, feat_hidden, pos_hidden, input_dropout_p=0.3):
        
        super(EncoderVid, self).__init__()
        self.dim_feat = feat_dim
        self.dim_bbox = bbox_dim
        self.dim_hidden = feat_hidden
        self.input_dropout_p = input_dropout_p

        input_dim = feat_dim

        input_dim += pos_hidden
        self.bbox_conv = nn.Sequential(
            nn.Conv2d(self.dim_bbox, pos_hidden, kernel_size=1),
            nn.BatchNorm2d(pos_hidden),
            nn.ReLU(),
            nn.Conv2d(pos_hidden, pos_hidden, kernel_size=1),
            nn.BatchNorm2d(pos_hidden),
            nn.ReLU(),
            
        )

        self.tohid = nn.Sequential(
            nn.Linear(feat_dim+pos_hidden, feat_hidden),
            nn.ELU(inplace=True))
        
        # self.roi_conv = nn.Sequential(
        #     nn.Conv1d(feat_dim, feat_hidden, kernel_size=3, padding=1),
        #     nn.ELU(inplace=True)
        # )

        # self.roi_conv = nn.Sequential(
        #     nn.Conv2d(4, 4, kernel_size=1),
        #     nn.BatchNorm2d(4),
        #     nn.ReLU(),
        # )


    def forward(self, video_o):
        bsize, num_segments, numf, numr, fdim =  video_o.shape
       
        video_o = video_o.view(bsize, num_segments*numf, numr, fdim)
        roi_feat = video_o[:,:,:, :self.dim_feat]
        roi_bbox = video_o[:,:,:, self.dim_feat:(self.dim_feat+self.dim_bbox)]
        bbox_pos = self.bbox_conv(roi_bbox.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        bbox_features = torch.cat([roi_feat, bbox_pos], dim=-1)

        bbox_feat = self.tohid(bbox_features)
        
        return bbox_feat

@MODELS.register_module()
class MIST(BaseModel):
    """MIST neck.

    This module is proposed in `MIST : Multi-modal Iterative Spatial-Temporal Transformer for Long-form Video Question Answering`
    """

    def __init__(self,
                video_dim=512,
                word_dim=768,
                num_ista_layers=2,
                num_ista_heads=8,
                embed_dim=512,
                hidden_dim=2048,
                dropout=0.1,
                Q=20,
                T=20,
                topk=2,
                topj=12,
                num_segments=8,
                probe=False
                ) -> None:
        """
        :param video_dim: dimension of the input video features
        :param word_dim: dimension of the input question features
        :param num_ista_layers: number of transformer layers
        :param num_ista_heads: number of transformer heads
        :param embed_dim: dimension for the transformer and final embedding
        :param hidden_dim: hidden dimension in the transformer
        :param dropout: dropout rate in the transformer
        :param Q: maximum number of tokens in the question
        :param T: maximum number of video features
        :param topk: number of segments to select
        :param topj: number of objects to select
        :param num_segments: number of segments per video
        :param probe: whether or not to freeze all parameters but the heads
        """
        super(MIST, self).__init__()
        # positional and modality encoding
        self.topk = topk
        self.topj = topj
        self.num_segments = num_segments
        self.numf = int(32 / self.num_segments)
        self.Q = Q
        T = 32 + (16) * self.topk * self.numf
        self.position = Embeddings(embed_dim, Q, T, dropout, True)
        self.T = T
        self.frame_position_embedding = PositionEmbeddings(512, 32, True)
        self.question_position_embedding = PositionEmbeddings(512, Q, True)
        self.token_type_embedding = TokenTypeEmbeddings(512, 3)

        d_pos = 128
        self.encode_vid = EncoderVid(feat_dim=video_dim,
                                     bbox_dim=5,
                                     feat_hidden=embed_dim,
                                     pos_hidden=d_pos)

        # video and question fusion modules
        self.ISTA = [ISTA(feature_dim=video_dim, word_dim=word_dim, Q=Q, N=num_ista_layers,
                          d_model=embed_dim, dropout=dropout, d_ff=hidden_dim, h=num_ista_heads, topk=self.topk, topj=self.topj)]
        for _ in range(1):
            self.ISTA.append(
                ISTA(feature_dim=embed_dim, word_dim=embed_dim, Q=Q, N=num_ista_layers,
                     embed_dim=embed_dim, dropout=dropout, d_ff=hidden_dim, h=num_ista_heads, topk=self.topk, topj=self.topj)
            )
        self.ISTA = nn.ModuleList(self.ISTA)

        # answer prediction
        self.vqproj = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim))


        self.config = DistilBertConfig.from_pretrained(
            "distilbert-base-uncased",
            n_layers=num_ista_layers,
            dim=embed_dim,
            dropout=dropout,
            hidden_dim=hidden_dim,
            attention_dropout=dropout,
            n_heads=num_ista_heads,
        )
        print('lxh self.config.initializer_range', self.config.initializer_range)
        # self.ttrans = Transformer(self.config)

        # weight initialization
        self.apply(self._init_weights)

        # pretrained DistilBERT language model
        self.bert = Bert()
        self.clip, _ = clip.load("ViT-B/32")



    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()




    def get_clip_txt_embedding(self, question):
        bsize = question.size(0)
        question_clip, word_clip = self.clip.encode_text(question.squeeze(dim=1))

        question_clip = question_clip / question_clip.norm(dim=-1, keepdim=True)   # [bsize, CLIP_dim]
        question_clip = question_clip.view(bsize, -1, 1).float()  # [bsize, 1, CLIP_dim]

        word_clip = word_clip / word_clip.norm(dim=-1, keepdim=True)   # [bsize, num_word, CLIP_dim]
        word_clip = word_clip.view(bsize, -1, 1).float()  # [bsize, num_word, CLIP_dim]
        return question_clip, word_clip

    def forward(
        self,
        video,
        question=None,
        question_clip=None,
        answer=None,
        text_mask=None
    ):
        """
        :param video: video features
        :param question: [bs, Q]
        :param labels: [bs, Q] used for masked language modeling
        :param answer: [batch_size, amax_words, 300] used for contrastive loss training, otherwise precomputed at the vocabulary level
        :param video_mask: [bs, T]
        :param text_mask: [bs, Q]
        """
        video_o, video_f = region_
        # video_o: [bs, num_clip * num_frame, num_object, 512]
        # video_f: [bs, num_clip * num_frame, 512])
        bsize, _, numr, fdim = video_o.size()
        num_segments, numf = self.num_segments, self.numf

        # embed video and question
        video_o = video_o.view(bsize, num_segments, numf, numr, fdim)
        video_o = self.encode_vid(video_o).view(bsize, num_segments, numf, numr, -1)

        q_feat, w_feat = self.get_clip_txt_embedding(question_clip)

        video_f_norm = video_f / video_f.norm(dim=-1, keepdim=True)
        video_clip = video_f_norm.view(bsize, num_segments, numf, -1)
        seg_feat = torch.mean(video_clip, dim=-2)

        question = self.bert(question)
        if question.shape[1] < self.Q:
            question = torch.cat(
                [
                    question,
                    torch.zeros(
                        question.shape[0],
                        self.Q - question.shape[1],
                        question.shape[2],
                    ).cuda(),
                ],
                1,
            )
            text_mask = torch.cat(
                [
                    text_mask,
                    torch.zeros(
                        text_mask.shape[0], self.Q - text_mask.shape[1]
                    ).cuda(),
                ],
                1,
            )


        # perform ISTA layers
        out_list = []
        for ista in self.ISTA:
            attended_vq, question, seg_feat = ista(q_feat, text_mask, question, seg_feat, video_o)
            out_list.append(attended_vq)

        # final answer prediction
        fusion_proj = torch.sum(torch.stack([out[:, 0, :] for out in out_list], dim=-1), dim=-1)
        fusion_proj = self.vqproj(fusion_proj)

        # TODO
