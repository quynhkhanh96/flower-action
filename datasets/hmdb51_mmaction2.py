import os 
import pickle 
import shutil 
import sys 
from mmaction.datasets import build_dataset

def hmdb51_partition(n_clients, preprocessed_dir, fold):
    """
    Args:
        preprocessed_dir (str): path to directory containing HMDB51's 
                                `rawframes`, `videos`, split files (.txt)
    """
    os.makedirs(preprocessed_dir + f'/partition', exist_ok=True)
    all_actions = {action: [] 
                    for action in os.listdir(preprocessed_dir + '/rawframes')}
    with open(preprocessed_dir + f'/hmdb51_train_split_{fold}_rawframes.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        action, _ = line.split('/')
        all_actions[action].append(line)
    for action in all_actions:
        all_actions[action] = [all_actions[action][i::n_clients] for i in range(n_clients)]

    for client_id in range(n_clients):
        client_files = []
        for action in all_actions:
            client_files.extend(all_actions[action][client_id])
        metadata_path = '{}/partition/hmdb51_train_split_{}_client_{}_rawframes.txt'.format(
            preprocessed_dir, fold, client_id
        )
        with open(metadata_path, 'a') as f:
            for client_file in client_files:
                f.write(client_file + '\n')

    shutil.copy(preprocessed_dir + f'/hmdb51_val_split_{fold}_rawframes.txt',
                preprocessed_dir + f'/partition/hmdb51_val_split_{fold}_rawframes.txt')
    
    with open(preprocessed_dir + '/partition/clients.pkl', 'wb') as handle:
        pickle.dump(all_actions, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_hmdb51_client_dataset(client_id, fold, hmdb51_root):

    dataset_type = 'RawframeDataset'
    data_root = hmdb51_root + f'/rawframes' 
    ann_file_train = hmdb51_root + f'/partition/hmdb51_train_split_{fold}_client_{client_id}_rawframes.txt'
    ann_file_val = hmdb51_root + f'/partition/hmdb51_val_split_{fold}_rawframes.txt'

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
    val_pipeline = [
        dict(
            type='SampleFrames',
            clip_len=8,
            frame_interval=4,
            num_clips=1,
            test_mode=True),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='CenterCrop', crop_size=224),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]

    data_train_cfg = dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline
    )
    data_val_cfg = dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root,
        pipeline=val_pipeline
    )

    train_dataset = build_dataset(data_train_cfg)
    val_dataset = build_dataset(data_val_cfg)

    return train_dataset, val_dataset

if __name__ == '__main__':

    preprocessed_dir = sys.argv[1]
    n_clients = int(sys.argv[2])
    try:
        fold = int(sys.argv[3])
    except:
        fold = 1
    hmdb51_partition(n_clients=n_clients, 
                    preprocessed_dir=preprocessed_dir, fold=fold)


    