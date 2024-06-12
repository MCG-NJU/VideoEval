# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Callable, List, Optional, Union

import h5py
import clip
import torch
import numpy as np
import pandas as pd
import os.path as osp

from torch.utils.data.dataloader import default_collate
from mmengine.logging import MMLogger
from mmengine.fileio import exists
from mmengine.dataset import BaseDataset, COLLATE_FUNCTIONS
from mmaction.utils import ConfigType
from mmaction.registry import DATASETS



def transform_bb(roi_bbox, width, height):
    dshape = list(roi_bbox.shape)
    tmp_bbox = roi_bbox.reshape([-1, 4])
    relative_bbox = tmp_bbox / np.asarray([width, height, width, height])
    relative_area = (tmp_bbox[:, 2] - tmp_bbox[:, 0] + 1) * \
                    (tmp_bbox[:, 3] - tmp_bbox[:, 1] + 1)/ (width*height)
    relative_area = relative_area.reshape(-1, 1)
    bbox_feat = np.hstack((relative_bbox, relative_area))
    dshape[-1] += 1
    bbox_feat = bbox_feat.reshape(dshape)

    return bbox_feat

@DATASETS.register_module() 
class VideoQADataset(BaseDataset, metaclass=ABCMeta):
    """Base class for VideoQA datasets.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict, optional): Path to a directory where
            videos are held. Defaults to None.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Defaults to 0.
        modality (str): Modality of data. Support ``RGB``, ``Flow``, ``Pose``,
            ``Audio``. Defaults to ``RGB``.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 data_prefix: Optional[ConfigType] = dict(prefix=''),
                 test_mode: bool = False,
                 dataset_name: str = 'agqa',
                 multi_choice: int = 0,
                 modality: str = 'feature',
                 video_feature_file: Optional[str] = None,
                 frame_size_file: Optional[str] = None,
                 vocab_file: Optional[str] = None,
                 candidate_answer_file: Optional[str] = None,
                 feat_dim: int = 512,
                 feat_frame_num: int = 32,
                 feat_patch_size: int = 224,
                 feat_grid_num: int = 4,
                 qmax_words: int = 20,
                 amax_words: int = 10,
                 **kwargs) -> None:

        self.dataset_name = dataset_name
        self.multi_choice = multi_choice
        self.modality = modality
        self.data_root = data_prefix['root']

        logger = MMLogger.get_current_instance()

        if video_feature_file is not None:
            self.video_feature_file = osp.join(self.data_root, video_feature_file)
            logger.info(f"video_feature_file: {self.video_feature_file}")
        if frame_size_file is not None:
            self.frame_size_file = osp.join(self.data_root, frame_size_file)
            logger.info(f"frame_size_file: {self.frame_size_file}")
        if vocab_file is not None:
            self.vocab_file = osp.join(self.data_root, vocab_file)
            with open(self.vocab_file, 'r') as f:
                self.answer2id = json.load(f)
            logger.info(f"vocab_file: {self.vocab_file}")
        if candidate_answer_file is not None:
            self.candidate_answer_file = osp.join(self.data_root, candidate_answer_file)
            logger.info(f"candidate_answer_file: {self.candidate_answer_file}")

        self.qmax_words = qmax_words
        self.amax_words = amax_words
        self.feat_dim = feat_dim
        self.feat_frame_num = feat_frame_num
        self.feat_patch_size = feat_patch_size
        self.feat_grid_num = feat_grid_num
        # self.use_frame = True
        # self.use_mot = False

        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)
        


    def get_video_feature(self, video_feat):
        # generate bbox coordinates of patches
        patch_bbox = []
        patch_size = self.feat_patch_size # 224
        grid_num = self.feat_grid_num # 4
        width, height = patch_size * grid_num, patch_size * grid_num
        assert video_feat.shape == (self.feat_frame_num, grid_num*grid_num+1, self.feat_dim), f"video_feat.shape: {video_feat.shape} but: {(self.feat_frame_num, grid_num*grid_num+1, self.feat_dim)}"

        for j in range(grid_num):
            for i in range(grid_num):
                patch_bbox.append([i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size])
        roi_bbox = np.tile(np.array(patch_bbox), (self.feat_frame_num, 1)).reshape(self.feat_frame_num, grid_num*grid_num, -1)  # [frame_num, bbox_num, -1]
        bbox_feat = transform_bb(roi_bbox, width, height)
        bbox_feat = torch.from_numpy(bbox_feat).type(torch.float32)

        roi_feat = video_feat[:, 1:, :]  # [frame_num, 16, dim]
        roi_feat = torch.from_numpy(roi_feat).type(torch.float32)
        region_feat = torch.cat((roi_feat, bbox_feat), dim=-1)
   
        frame_feat = video_feat[:, 0, :]
        frame_feat = torch.from_numpy(frame_feat).type(torch.float32)

        # print('Sampled feat: {}'.format(region_feat.shape))
        return region_feat, frame_feat

    def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        logger = MMLogger.get_current_instance()
        if self.dataset_name == 'agqa':
            if self.modality != 'feature':
                raise NotImplementedError(self.modality)
            else:
                exists(self.ann_file)
                annos = pd.read_json(self.ann_file)
                data_list = []

                video_feats = {}

                with h5py.File(self.video_feature_file, 'r') as fp:
                    vids = fp['ids']
                    feats = fp['features']
                    logger.info(f"features' shape: {feats.shape}")  # v_num, clip_num, feat_dim
                    for idx, (vid, feat) in enumerate(zip(vids, feats)):
                        print(vid.decode('utf-8'), len(feats))
                        video_feats[vid.decode('utf-8')] = feat
  

                for index in range(len(annos)):
                    cur_sample = annos.iloc[index]
                    raw_vid_id = str(cur_sample["video_id"])
                    # print(raw_vid_id)
                    # frame_size = self.frame_size[vid_id]
                    region_feat, frame_feat = self.get_video_feature(video_feats[raw_vid_id])
                    answer_txt = cur_sample["answer"]
                    answer_id = self.answer2id.get(answer_txt, -1)
                    data_list.append({
                            "video_id": raw_vid_id, 
                            'region_feat': region_feat,
                            'frame_feat': frame_feat,
                            # "width": frame_size['width'], "height": frame_size['height'],
                            "question_id": str(cur_sample['question_id']),
                            "question_txt": cur_sample['question'],
                            "answer_type": cur_sample['answer_type'],
                            "answer_id": answer_id, # TODO 待定看要不要合到模型中，感觉最好还是和a2v一块
                            "answer_txt": answer_txt
                    })
        else:
            raise NotImplementedError(self.dataset_name)

        return data_list

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index. 从data_list中获取数据并增加一些额外信息"""
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality


        return data_info


@COLLATE_FUNCTIONS.register_module()
def videoqa_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: tensorized batch with the question and the ans candidates padded to the max length of the batch
    """
    qmax_len = max(len(batch[i]["question_Bert_token"]) for i in range(len(batch)))

    for i in range(len(batch)):
        if len(batch[i]["question_Bert_token"]) < qmax_len:
            batch[i]["question_Bert_token"] = torch.cat(
                [
                    batch[i]["question_Bert_token"],
                    torch.zeros(qmax_len - len(batch[i]["question_Bert_token"]), dtype=torch.long),
                ],
                0,
            )

    return default_collate(batch)

if __name__ == '__main__':

    test_dataset = VideoQADataset(ann_file='yourpath/agqa_v2/data/agqa/agqa_val.jsonl',
        video_feature_file='feature/agqa/frame_feat/clip_patch_feat_all.h5',
        data_prefix=dict(root='yourpath/agqa_v2'),
        pipeline=[],
        test_mode=True)
    from mmengine.runner import Runner
    test_dataloader = Runner.build_dataloader(dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='videoqa_collate_fn'),
    dataset=dict(
        type='VideoQADataset',
        ann_file='data/agqa/agqa_val.jsonl',
        video_feature_file='feature/agqa/frame_feat/clip_patch_feat_all.h5',
        data_prefix=dict(root='yourpath/agqa_v2'),
        pipeline=[],
        test_mode=True)))