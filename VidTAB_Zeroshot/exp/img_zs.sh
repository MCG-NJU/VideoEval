python img_zero_shot.py --model_name CLIP-L \
                 --model_path /mnt/petrelfs/lixinhao/lxh_exp/pretrained_models/CLIP/ViT-L-14.pt \
                 --anno_path /mnt/petrelfs/lixinhao/lxh_exp/data/video_eval/MOB/test.txt \
                 --data_path /mnt/petrelfs/lixinhao/lxh_exp/data/video_eval/MOB \
                 --prefix /mnt/petrelfs/lixinhao/lxh_exp/data/video_eval/MOB/video \
                 --batch_size 32
