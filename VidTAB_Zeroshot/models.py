import clip
from eva_clip import create_model_and_transforms, get_tokenizer
import torch
from functools import partial
# CLIP-L /mnt/petrelfs/lixinhao/lxh_exp/pretrained_models/CLIP/ViT-L-14.pt
# EVA01-CLIP-g-14-plus "/mnt/petrelfs/lixinhao/lxh_exp/pretrained_models/EVA/EVA01_CLIP_g_14_plus_psz14_s11B.pt"
# 'VICLIP-L "/mnt/petrelfs/lixinhao/lxh_exp/pretrained_models/lyz/InternVid200M_videoclip.pth"
# Internvideo2 "/mnt/petrelfs/share_data/lixinhao/avp_1b_f4_coco_smit_e4.pt"
def create_models(model_name, model_path, device='cuda'):
    if model_name == 'CLIP-L':
        model, preprocess = clip.load(model_path, device=device)
        tokenizer = clip.tokenize
        
    elif model_name == 'EVA01-CLIP-g-14-plus':
        model, _, preprocess = create_model_and_transforms(model_name, model_path, force_custom_clip=True)
        tokenizer = get_tokenizer(model_name)
        model = model.to(device)

        return dict(model=model, 
                    preprocess=preprocess, 
                    tokenizer=tokenizer)
        
    elif model_name == 'VICLIP-L':
        from viclip import frames2tensor, get_viclip, get_text_feat_dict

        
        model_cfgs = {
                "size": "l",
                "pretrained": model_path,
        }
        m = get_viclip(model_cfgs["size"], model_cfgs["pretrained"])
        model, text_preprocess = m["viclip"], m["tokenizer"]
        model = model.to(device)
        preprocess = frames2tensor
        
        @torch.no_grad()
        def get_text_feat(model, text, tokenizer):
            text_feat_d = {}
            text_feat_d = get_text_feat_dict(text, model, tokenizer, text_feat_d)
            text_feats = [text_feat_d[t] for t in text]
            text_feats_tensor = torch.cat(text_feats, 0)
            text_feats_tensor /= text_feats_tensor.norm(dim=-1, keepdim=True)
            return text_feats_tensor
        tokenizer = partial(get_text_feat, model=model, tokenizer = text_preprocess)

    elif model_name == 'Internvideo2':
        
        from internvideo2.demo.config import Config, eval_dict_leaf
        from internvideo2.demo.utils import (
            frames2tensor,
            get_text_feat_dict,
            setup_internvideo2,        
        )

    
        config = Config.from_file("internvideo2/internvideo2_stage2_config.py")
        config = eval_dict_leaf(config)
        config["pretrained_path"] = model_path

        model, _ = setup_internvideo2(config)
        model.to(device)

        fn = config.get("num_frames", 8)
        size_t = config.get("size_t", 224)
        preprocess = partial(frames2tensor, fnum=fn, target_size=(size_t, size_t))
        
        
        @torch.no_grad()
        def get_text_feat(model, text):
            text_feat_d = {}
            text_feat_d = get_text_feat_dict(text, model, text_feat_d)
            text_feats = [text_feat_d[t] for t in text]
            text_feats_tensor = torch.cat(text_feats, 0)
            text_feats_tensor /= text_feats_tensor.norm(dim=-1, keepdim=True)
            return text_feats_tensor
        
        tokenizer = partial(get_text_feat, model=model)


    else:
        raise NotImplementedError
    
    return dict(model=model, 
                preprocess=preprocess,
                tokenizer=tokenizer)

    