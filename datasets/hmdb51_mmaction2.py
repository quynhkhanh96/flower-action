from mmaction.datasets import build_dataset

def hmdb51_partition(n_clients):
    pass    

def get_hmdb51_client_dataset(client_id, fold, hmdb51_root):

    dataset_type = 'RawframeDataset'
    #'data/hmdb51/rawframes'
    data_root = hmdb51_root + f'/client_{client_id}/rawframes' 
    # ann_file_train = f'data/hmdb51/hmdb51_train_split_{fold}_rawframes.txt'
    ann_file_train = hmdb51_root + f'/client_{client_id}/hmdb51_train_split_{fold}_rawframes.txt'

    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], 
                        std=[58.395, 57.12, 57.375], 
                        to_bgr=False)

    train_pipeline = [
        dict(type='SampleFrames', clip_len=8, frame_interval=4, num_clips=1),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='RandomResizedCrop'),
        dict(type='Resize', scale=(224, 224), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label'])
    ]

    data_train_cfg = dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline)

    train_dataset = build_dataset(data_train_cfg)



    