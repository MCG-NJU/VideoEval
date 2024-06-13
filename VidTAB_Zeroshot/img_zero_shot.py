import av
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from transformers import AutoProcessor, AutoModel
from transformers import AutoProcessor, SiglipVisionModel
from tqdm import tqdm
from timm.utils import accuracy
import io
import sys
from decord import VideoReader
import decord

from img_prompt_gen import read_text
from datasets import ImageClsDataset
from models import create_models
import argparse

@torch.no_grad()
def extract_feature(data_loader, model, tokenizer, prompts, device):
    
    input_ids = tokenizer(prompts).to(device)
    text_features = model.encode_text(input_ids)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    model.eval()
    labels_list = []
    preds_list = []
    for batch in tqdm(data_loader):
        for k in batch.keys():
            batch[k] = batch[k].to(device, non_blocking=True)
        B, T, C, H, W = batch['pixel_values'].shape
        batch['pixel_values'] = batch['pixel_values'].flatten(0, 1)
        target = batch.pop('labels')
        image_features = model.encode_image(batch['pixel_values'])
        image_features = image_features.reshape(B, T, image_features.shape[-1]).mean(1)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_ids = (100.0 * image_features @ text_features.T).argmax(dim=-1)
        labels_list.append(target.cpu())
        preds_list.append(text_ids.cpu())

    labels_list = torch.concat(labels_list, 0)
    preds_list = torch.concat(preds_list, 0)
    print(labels_list.shape, preds_list.shape)
    print("accuracy:", (labels_list == preds_list).float().mean())
    return labels_list, preds_list




parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--anno_path', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--prefix', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dict = create_models(args.model_name, args.model_path)
dataset = ImageClsDataset(args.anno_path, args.prefix, model_dict["preprocess"], num_segment=1)
dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size, shuffle=False)
prompts = read_text(args.data_path)
extract_feature(dataloader, model_dict["model"], model_dict["tokenizer"], prompts, device)


    #datasets = ['animal_kingdom', 'breakfast', 'MOB',
    #            'SurgicalActions160', 'CAER', 'DOVER',
    #            'facefake', 'ARID']
    #datasets = ['MOB']
    
    #for DATASET in datasets:



        #DATA_PATH, PREFIX = return_datapath_prefix(DATASET)
        
        
        
