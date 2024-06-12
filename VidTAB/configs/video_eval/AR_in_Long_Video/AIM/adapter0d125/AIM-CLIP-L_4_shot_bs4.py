_base_ = [
    '../../../default_runtime_on_local_low.py'
]

num_frames = 8

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViT_AIM_CLIP',
        pretrained="yourpath/CLIP",
        input_resolution=224,
        patch_size=14,
        num_frames=8,
        width=1024,
        layers=24,
        heads=16,
        drop_path_rate=0.2,
        adapter_scale=1,
        mlp_ratio=0.125,
        num_tadapter=2
    ),
    cls_head=dict(
        type='I3DHead',
        in_channels=1024,
        num_classes=10,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'),
        data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[122.769, 116.74, 104.04],
        std=[68.493, 66.63, 70.321], # 注意clip和imagenet的不一样
        format_shape='NCTHW'))


# dataset settings
dataset_type = 'VideoDataset'
data_root = 'pssd:s3://AR_in_Long_Video/AR_in_Long_VideoII_15fps_qvga_sync'
data_root_val = data_root
ann_file_train = 'yourpath/video_eval/AR_in_Long_Video/train_4.txt'
ann_file_val = 'yourpath/video_eval/AR_in_Long_Video/val_test.txt'
ann_file_test = 'yourpath/video_eval/AR_in_Long_Video/val_test.txt'
file_client_args = dict(io_backend='petrel')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, test_mode=True, num_clips=4),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=4,
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
    type='EpochBasedTrainLoop', max_epochs=25, val_begin=1, dynamic_intervals=[(1, 5)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

accumulative_counts = 1
# optimizer
optim_wrapper = dict(
    type='GradMonitorAmpOptimWrapper',
    accumulative_counts=accumulative_counts,
    optimizer=dict(
        type='AdamW', lr=3e-4, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='GradMonitorSwinOptimWrapperConstructor',
    # accumulative_counts=2,
    paramwise_cfg=dict(class_embedding=dict(decay_mult=0.),
                        positional_embedding=dict(decay_mult=0.),
                        temporal_embedding=dict(decay_mult=0.),
                        absolute_pos_embed=dict(decay_mult=0.), # 这玩意不一定有，写着反正没损失
                        ln_1=dict(decay_mult=0.),
                        ln_2=dict(decay_mult=0.),
                        ln_pre=dict(decay_mult=0.),
                        ln_post=dict(decay_mult=0.),# TODO 理论上不需要
                        backbone=dict(lr_mult=0.1)
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
auto_scale_lr = dict(enable=False, base_batch_size=64)

