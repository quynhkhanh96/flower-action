import os 
import random 
from torchvision.datasets import HMDB51
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import augmentations.video as T
import yaml 
import sys

# HMDB dataset
def _data_transforms_hmdb51():
    train_transform = transforms.Compose([  
        T.ToFloatTensorInZeroOne(),
        T.Resize((200, 200)),
        T.RandomHorizontalFlip(),
        T.RandomCrop((172, 172))
    ])

    valid_transform = transforms.Compose([                           
        T.ToFloatTensorInZeroOne(),
        T.Resize((200, 200)),
        T.CenterCrop((172, 172))
    ])
    return train_transform, valid_transform

# HMDB partition
def hmdb51_partition(n_clients, ann_path, fed_ann_path):
    # fed_test_train_splits: `n_clients` directories (eg. client_0): each has the same filenames as `test_train_splits`
    os.makedirs(fed_ann_path, exist_ok=True)
    for i in range(n_clients):
        os.makedirs(fed_ann_path + f'/client_{i}', exist_ok=True)
    
    for ann_file in os.listdir(ann_path):
        train_fnames = []
        other_fnames = []
        with open(ann_path + '/' + ann_file) as f:
            lines = f.readlines()
        for line in lines:
            video_fname, tag_str = line.split()
            if int(tag_str) == 1:
                train_fnames.append(video_fname)
            else:
                other_fnames.append([video_fname, tag_str])
        random.shuffle(train_fnames)
        # paritioning 
        train_fnames = [train_fnames[i::n_clients] for i in range(n_clients)]
        for i in range(n_clients):
            with open(fed_ann_path + f'/client_{i}/{ann_file}', 'w') as f:
                for fname in train_fnames[i]:
                    f.write(f'{fname} 1\n')
                for fname, tag_str in other_fnames:
                    f.write(f'{fname} {tag_str}\n')

# HMBD get client loader and test loader 
def get_hmdb51_client_loader(client_id, fold, 
                            num_frames, frame_rate, clip_steps,
                            local_bz, test_bz,
                            video_data_dir, working_dir):

    transform_train, transform_test = _data_transforms_hmdb51()

    hmdb51_train = HMDB51(video_data_dir, working_dir + f'/fed_test_train_splits/client_{client_id}', 
                            num_frames, frame_rate=frame_rate,
                            step_between_clips=clip_steps, 
                            fold=fold, train=True,
                            transform=transform_train, num_workers=2)

    hmdb51_test = HMDB51(video_data_dir, working_dir + f'/fed_test_train_splits/client_{client_id}', 
                            num_frames, frame_rate=frame_rate,
                            step_between_clips=clip_steps, 
                            fold=fold, train=False,
                            transform=transform_test, num_workers=2)

    train_loader = DataLoader(hmdb51_train, batch_size=local_bz, shuffle=True)
    test_loader  = DataLoader(hmdb51_test, batch_size=test_bz, shuffle=False)

    hmdb51_classes = hmdb51_train.classes
    num_classes = len(hmdb51_classes)

    return train_loader, test_loader, num_classes

if __name__ == '__main__':

    working_dir = sys.argv[1]
    cfg_path = sys.argv[2]
    test_train_splits_path = sys.argv[3]

    with open(cfg_path, 'r') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)

    hmdb51_partition(n_clients=cfgs['num_C'], 
                    ann_path=test_train_splits_path, 
                    fed_ann_path=working_dir + '/fed_test_train_splits')
