import argparse
import av
import numpy as np
import torch
import  torch.nn.functional as F
np.random.seed(0)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import torch
from models import create_models
from datasets import VideoClsDataset
from vid_prompt_gen import read_text


@torch.no_grad()
def extract_feature(data_loader, model, tokenizer, prompts, device):
    text_features = tokenizer(text=prompts).to(device)
    model.eval()
    labels_list = []
    preds_list = []
    for batch in tqdm(data_loader):
        for k in batch.keys():
            batch[k] = batch[k].to(device, non_blocking=True)
        batch["pixel_values"] = batch["pixel_values"].squeeze(1)
        target = batch.pop("labels")
        image_features = model.get_vid_feat(batch["pixel_values"])
        image_features /= image_features.norm(dim=-1, keepdim=True)
        _, text_ids = model.predict_label(image_features, text_features, top=1)
        labels_list.append(target.cpu())
        preds_list.append(text_ids.squeeze(1).cpu())

    labels_list = torch.concat(labels_list, 0)
    preds_list = torch.concat(preds_list, 0)
    print(labels_list.shape, preds_list.shape)
    print(
        "accuracy:", (labels_list == preds_list).float().mean()
    )
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
dataset = VideoClsDataset(args.anno_path, args.prefix, model_dict["preprocess"], num_segment=1)
dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size, shuffle=False)
prompts = read_text(args.data_path)
extract_feature(dataloader, model_dict["model"], model_dict["tokenizer"], prompts, device)
