_base_ = [
    'faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py'
]


dataset_type = 'XMLDataset'
piropo_root = 'data/PIROPO/'
mw18_root = 'data/MW_18Mar/'
cvrg_root = 'data/CVRG/HumanCarOmniDataset/HumanSet/omni/'
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
#classes = ['person']
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=[
            piropo_root + 'Room_A/omni_1A/omni1A_training/images.txt',
            piropo_root + 'Room_A/omni_2A/omni2A_training/images.txt',
            piropo_root + 'Room_A/omni_3A/omni3A_training/images.txt',
            piropo_root + 'Room_B/omni_1B/omni1B_training/images.txt',
            mw18_root + 'Train/MW-18Mar-2/images.txt',
            mw18_root + 'Train/MW-18Mar-3/images.txt',
            mw18_root + 'Train/MW-18Mar-7/images.txt',
            mw18_root + 'Train/MW-18Mar-8/images.txt',
            mw18_root + 'Train/MW-18Mar-10/images.txt',
            mw18_root + 'Train/MW-18Mar-12/images.txt',
            mw18_root + 'Train/MW-18Mar-13/images.txt',
            mw18_root + 'Train/MW-18Mar-14/images.txt',
            mw18_root + 'Train/MW-18Mar-17/images.txt',
            mw18_root + 'Train/MW-18Mar-18/images.txt',
            mw18_root + 'Train/MW-18Mar-19/images.txt',
            mw18_root + 'Train/MW-18Mar-21/images.txt',
            mw18_root + 'Train/MW-18Mar-22/images.txt',
            mw18_root + 'Train/MW-18Mar-23/images.txt',
            mw18_root + 'Train/MW-18Mar-24/images.txt',
            mw18_root + 'Train/MW-18Mar-25/images.txt',
            mw18_root + 'Train/MW-18Mar-26/images.txt',
            mw18_root + 'Train/MW-18Mar-27/images.txt',
            mw18_root + 'Train/MW-18Mar-31/images.txt',
            cvrg_root + 'images.txt',
        ],
        img_prefix=[
            piropo_root + 'Room_A/omni_1A/omni1A_training/',
            piropo_root + 'Room_A/omni_2A/omni2A_training/',
            piropo_root + 'Room_A/omni_3A/omni3A_training/',
            piropo_root + 'Room_B/omni_1B/omni1B_training/',
            mw18_root + 'Train/MW-18Mar-2/',
            mw18_root + 'Train/MW-18Mar-3/',
            mw18_root + 'Train/MW-18Mar-7/',
            mw18_root + 'Train/MW-18Mar-8/',
            mw18_root + 'Train/MW-18Mar-10/',
            mw18_root + 'Train/MW-18Mar-12',
            mw18_root + 'Train/MW-18Mar-13/',
            mw18_root + 'Train/MW-18Mar-14/',
            mw18_root + 'Train/MW-18Mar-17/',
            mw18_root + 'Train/MW-18Mar-18/',
            mw18_root + 'Train/MW-18Mar-19/',
            mw18_root + 'Train/MW-18Mar-21/',
            mw18_root + 'Train/MW-18Mar-22/',
            mw18_root + 'Train/MW-18Mar-23/',
            mw18_root + 'Train/MW-18Mar-24/',
            mw18_root + 'Train/MW-18Mar-25/',
            mw18_root + 'Train/MW-18Mar-26/',
            mw18_root + 'Train/MW-18Mar-27/',
            mw18_root + 'Train/MW-18Mar-31/',
            cvrg_root,
        ],
        pipeline=train_pipeline),
    val=dict(
        type='CocoDataset',
        ann_file='data/MW18Mar_PIROPO_test.json',
        img_prefix='data/',
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
         ann_file='data/MW18Mar_PIROPO_test.json',
        img_prefix='data/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
load_from = './checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_bbox_mAP-0.398_20200504_163323-30042637_only-person.pth'
work_dir = './work_dirs/blub'
