import av
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from petrel_client.client import Client
from transformers import (
    AutoImageProcessor,
    SiglipImageProcessor,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)
import  torch.nn.functional as F
np.random.seed(0)
import io
import os
import sys

import decord
import pandas as pd
from decord import VideoReader
from timm.utils import accuracy
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, SiglipVisionModel

import io
import logging
import os

import cv2
import numpy as np
import torch
from PIL import Image
from vid_prompt_gen import read_text


def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def read_video_decord(vr, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        :param vr:
    '''
    vr.seek(0)
    buffer = vr.get_batch(indices).asnumpy()
    return buffer


def sample_frame_indices(clip_len, frame_sample_rate, seg_len, num_segment, chunk_nb):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''

    # end_idx = seg_len - 1
    temporal_step = seg_len // num_segment
    start_idx = int(chunk_nb * temporal_step)
    end_idx = start_idx + temporal_step
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, 0, seg_len - 1).astype(np.int64)
    return indices

class VideoClsDataset(Dataset):
    def __init__(self, anno_path, prefix, processor, num_segment) -> None:
        super().__init__()
        cleaned = pd.read_csv(anno_path, header=None, delimiter=" ")
        self.samples = list(cleaned.values[:, 0])
        self.labels = list(cleaned.values[:, 1])
        self.num_segment = num_segment

        self.label_array = list()
        self.dataset_samples = list()
        for i in range(self.num_segment):
            for sample, label in zip(self.samples, self.labels):
                self.label_array.append((i, label))
                self.dataset_samples.append(sample)

        self.prefix = prefix
        self.processor = processor
        self.client_ckpt = Client(conf_path="~/petreloss.conf")

    def __len__(self):
        return len(self.label_array)

    def __getitem__(self, index):
        video_file = os.path.join(self.prefix, self.dataset_samples[index])
        # print(video_file)
        chunk_nb, label = self.label_array[index]

        if "s3://" in video_file:
            video_bytes = self.client_ckpt.get(video_file)
            vr = VideoReader(io.BytesIO(video_bytes), num_threads=0)
            indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=len(vr),
                                           num_segment=self.num_segment, chunk_nb=chunk_nb)
            frames = read_video_decord(vr, indices)[:, :, :, ::-1]
        else:
            video = cv2.VideoCapture(video_file)
            frames = [x for x in _frame_from_video(video)]
        assert (len(frames) > 0)
        while len(frames) < 16:
            frames.append(frames[-1])
        process_video = self.processor(frames)

        inputs = dict()
        inputs.update(pixel_values=process_video)
        inputs.update(labels=label)
        return inputs


class ImageClsDataset(Dataset):
    def __init__(self, anno_path, prefix, processor, num_segment) -> None:
        super().__init__()
        cleaned = pd.read_csv(anno_path, header=None, delimiter=' ')
        self.samples = list(cleaned.values[:, 0])
        self.labels = list(cleaned.values[:, 1])
        self.num_segment = num_segment

        self.label_array = list()
        self.dataset_samples = list()
        for i in range(self.num_segment):
            for sample, label in zip(self.samples, self.labels):
                self.label_array.append((i, label))
                self.dataset_samples.append(sample)

        self.prefix = prefix
        self.processor = processor
        self.client_ckpt = Client(conf_path='~/petreloss.conf')

    def __len__(self):
        return len(self.label_array)

    def __getitem__(self, index):
        video_file = os.path.join(self.prefix, self.dataset_samples[index])
        chunk_nb, label = self.label_array[index]

        if "s3://" in video_file:
            video_bytes = self.client_ckpt.get(video_file)
            vr = VideoReader(io.BytesIO(video_bytes), num_threads=0)
        else:
            vr = VideoReader(video_file, num_threads=0)

        indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=len(vr),
                                       num_segment=self.num_segment, chunk_nb=chunk_nb)

        video = read_video_decord(vr, indices)

        process_video = []
        for v in video:
            process_video.append(self.processor(Image.fromarray(v)))
        while len(process_video) < 16:
            process_video.append(process_video[-1])

        process_video = torch.stack(process_video, dim=0)
        inputs = dict()
        inputs.update(pixel_values=process_video)
        inputs.update(labels=label)
        return inputs