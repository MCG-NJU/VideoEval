_base_ = [
    '../../_base_/default_runtime_on_ceph_hdd.py'
]

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViT_lyz',
        pretrained='yourpath/lyz/ckpt_best_model.pth',
        input_resolution=224, kernel_size=1,
        patch_size=14, width=1024, layers=24, heads=16, output_dim=None,
        num_frames=8, drop_path=0.1, checkpoint_num=0, # center=True,
        dropout=0., freeze_backbone=True),
    cls_head=dict(
        type='I3DHead',
        in_channels=1024,
        num_classes=400,
        spatial_type=None,
        dropout_ratio=0.5,
        use_bn=True,
        # label_smooth_eps=0.1, 
        average_clips='prob'),
        data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[122.769, 116.74, 104.04],
        std=[68.493, 66.63, 70.321], 
        format_shape='NCTHW'))


# dataset settings
dataset_type = 'VideoDataset'
data_root = 's3://k400/'
data_root_val = data_root
ann_file_train = 'yourpath/kinetics_400/train.csv'
ann_file_val = 'yourpath/kinetics_400/val.csv'
ann_file_test = 'yourpath/kinetics_400/val.csv'

file_client_args = dict(io_backend='petrel')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='Flip', flip_ratio=0.5),
    # dict(
    #     type='PytorchVideoWrapper',
    #     op='RandAugment',
    #     magnitude=7,
    #     num_layers=4),
    # dict(type='RandomErasing', erase_prob=0.25),
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
    dict(type='UniformSample', clip_len=8, test_mode=True, num_clips=4),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16*4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
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

# accumulative_counts = 2
# optimizer
optim_wrapper = dict(
    type='GradMonitorAmpOptimWrapper',
    # accumulative_counts=accumulative_counts,
    optimizer=dict(type='LARS', lr=0.1, weight_decay=0, momentum=0.9),
    constructor='GradMonitorSwinOptimWrapperConstructor',
    paramwise_cfg=dict(class_embedding=dict(decay_mult=0.),
                        positional_embedding=dict(decay_mult=0.),
                        temporal_positional_embedding=dict(decay_mult=0.),
                        # backbone=dict(lr_mult=0.1) # NOTE
                                    ))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.05,
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


