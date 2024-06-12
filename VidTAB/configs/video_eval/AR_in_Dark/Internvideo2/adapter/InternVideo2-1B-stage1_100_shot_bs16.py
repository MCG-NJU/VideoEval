_base_ = [
    '../../../default_runtime_on_local_low.py'
]

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='InternVideo2',
        pretrained="/mnt/petrelfs/share_data/likunchang/model/um_teacher/umt2/vit_g14_1.1M_CLIP+MAE_300e_pt_k710_ft.pth",
        adaptation_type='adapter',
        in_chans=3, img_size=224, patch_size=14,
        embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
         qkv_bias=False,
        drop_path_rate=0.25,
        init_values=0.00001,
        qk_normalization=True,
        use_flash_attn=False,
        use_fused_rmsnorm=False,
        use_fused_mlp=False,
        fused_mlp_heuristic=1,
        layerscale_no_force_fp32=False,
        num_frames=8,
        tubelet_size=1,
        sep_pos_embed=False,
        sep_image_video_pos_embed=False),
    cls_head=dict(
        type='I3DHead',
        in_channels=1408,
        num_classes=11,
        spatial_type=None,
        dropout_ratio=0.5,
        # label_smooth_eps=0.1, 
        average_clips='prob'),
        data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[122.769, 116.74, 104.04],
        std=[68.493, 66.63, 70.321], # 注意clip和imagenet的不一样
        format_shape='NCTHW'))


# dataset settings
dataset_type = 'VideoDataset'
data_root = 'yourpath/video_eval/AR_in_Dark/clips_v1.5/'
data_root_val = data_root
ann_file_train = 'yourpath/video_eval/AR_in_Dark/train_100.txt'
ann_file_val = 'yourpath/video_eval/AR_in_Dark/test.txt'
ann_file_test = 'yourpath/video_eval/AR_in_Dark/test.txt'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=8, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=8, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=25, val_begin=1, dynamic_intervals=[(1, 5), (20, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


accumulative_counts = 8
# optimizer
optim_wrapper = dict(
    type='GradMonitorAmpOptimWrapper',
    accumulative_counts=accumulative_counts,
    optimizer=dict(
        type='AdamW', lr=5e-5, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='GradMonitorSwinOptimWrapperConstructor',
    paramwise_cfg=dict(class_embedding=dict(decay_mult=0.),
                        positional_embedding=dict(decay_mult=0.),
                        temporal_positional_embedding=dict(decay_mult=0.),
                        # backbone=dict(lr_mult=0.1) # NOTE
                                    ))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=25,
        eta_min=1e-6,
        by_epoch=True,
        begin=0,
        end=25)
]

find_unused_parameters = True
auto_scale_lr = dict(enable=False)

