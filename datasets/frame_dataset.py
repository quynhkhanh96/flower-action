import os
from collections import defaultdict 
import cv2
import random 
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import train_test_split

class FrameDataset(Dataset):
    def __init__(self, frame_dir,
                    annotation_file_path,
                    n_frames,
                    mode='train',
                    to_rgb=True,
                    transform=None,
                    use_albumentations=False):
        self.frame_dir = frame_dir
        self.annotation_file_path = annotation_file_path
        self.n_frames = n_frames

        self.to_rgb = to_rgb
        self.transform = transform
        self.use_albumentations = use_albumentations

        self.mode = mode 

        with open(self.annotation_file_path) as f:
            lines = [l.strip() for l in f.readlines()]
            self.clips = [l.strip().split(' ')[0] for l in lines]
            self.labels = [int(l.strip().split(' ')[1]) for l in lines]

    def __len__(self):
        return len(self.clips)

    def sample_frames(self, video_file):
        '''
            Inside each /<video_id> directory, there are images named `frame_<id>.jpg`,
            for eg. `frame_000.jpg`. This function samples a subsequence of length 
            `self.n_frames` of them.
        '''
        all_frames = np.array(sorted(os.listdir(video_file), 
                            key=lambda x: int(x.split('_')[1].split('.')[0])))

        start_frame, end_frame = 0, len(all_frames) - 1
        if self.mode == 'train':
            segments = np.linspace(start_frame, end_frame, self.n_frames + 1)
            segment_length = (end_frame - start_frame) / self.n_frames
            sampled_frame_ids = segments[:-1] + np.random.rand(self.n_frames) * segment_length
        else:
            sampled_frame_ids = np.linspace(start_frame, end_frame, self.n_frames)

        frames = all_frames[sampled_frame_ids.round().astype(np.int64)]
        return frames
    
    def __getitem__(self, idx):
        video_id = self.clips[idx]
        video_file = os.path.join(self.frame_dir, video_id)

        ''' Sample video frames '''
        frame_names = self.sample_frames(video_file)
        frames = []
        for frame_name in frame_names:
            frame = cv2.imread(video_file + '/' + frame_name)
            frames.append(frame)

        ''' Transform and augment RGB images '''
        if self.to_rgb:
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        if self.transform is not None:
            frames = [self.transform(frame) if not self.use_albumentations
                      else self.transform(image=frame)['image'] for frame in frames]                
        data = torch.from_numpy(np.stack(frames).transpose((1, 0, 2, 3)))        
        # data shape (c, s, w, h) where s is seq_len, c is number of channels
        return data, self.labels[idx], video_file 

def data_partition(n_clients, data_dir,
                    train_annotation_path,
                    mode='iid'):
    '''
    Args:
        n_clients (int): number of clients
        data_dir (str): path to data (including `rgb_frames`, `train.txt` and `val.txt` directories)
        train_annotation_path (str): path to train annotation file, in which each line is 
                                <video_id> <label>
        mode (str): data partition mode
    '''
    print(f'Partitioning data among {n_clients} clients ...')
    with open(train_annotation_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        train_video_ids = [l.split(' ')[0] for l in lines]
        train_labels = [int(l.split(' ')[1]) for l in lines]

    train_video_ids = np.array(train_video_ids)
    train_labels = np.array(train_labels)
    if mode == 'iid':
        # split by labels to `n_clients` clients
        res = defaultdict(list)
        n_classes = len(set(train_labels))
        for cls in range(n_classes):
            ids = np.where(train_labels == cls)[0]
            random.shuffle(ids)
            ids = [ids[i::n_clients] for i in range(n_clients)]
            for client_id in range(n_clients):
                res[client_id] += list(ids[client_id])

        for i in range(n_clients):
            with open(data_dir + f'/client_{i}_train.txt', 'a') as f:
                for id in res[i]:
                    f.write('{} {}\n'.format(
                        train_video_ids[id], train_labels[id]))
    else:
        # raise ValueError(f'Partition mode {mode} not implemented.')
        train_sorted_index = np.argsort(train_labels)
        train_video_ids = train_video_ids[train_sorted_index]
        train_labels = train_labels[train_sorted_index]

        shard_size = len(train_labels) // n_clients
        shard_start_index = [i for i in range(0, len(train_labels), shard_size)]
        random.shuffle(shard_start_index)

        num_shards = len(shard_start_index) // n_clients
        for client_id in range(n_clients):
            _index = num_shards * client_id
            local_video_ids = np.concatenate([
                train_video_ids[shard_start_index[_index +
                                            i]:shard_start_index[_index + i] +
                          shard_size] for i in range(num_shards)
            ], axis=0)

            local_labels = np.concatenate([
                train_labels[shard_start_index[_index +
                                              i]:shard_start_index[_index +
                                                                   i] +
                            shard_size] for i in range(num_shards)
            ], axis=0)

            with open(data_dir + f'/client_{client_id}_train.txt', 'a') as f:
                for i in range(len(local_video_ids)):
                    f.write('{} {}\n'.format(
                        local_video_ids[i], local_labels[i]
                    ))

def get_client_loaders(client_id, data_dir, cfgs):
    '''
    Args:
        client_id (int): id of the client 
        data_dir (str): path that contains <client_id> directory, the results from 
                        data partition step 
        cfgs: configuration object
    Returns:
        Tuple[torch.utils.data.dataset]: train and val dataset
    '''
    scaler = T.Resize(((cfgs.height, cfgs.width)))
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
		                    std=[0.229, 0.224, 0.225])    
    transform= T.Compose([
		T.ToPILImage(),
		scaler,
		T.ToTensor(),
		normalize      
		])  
    
    train_set = FrameDataset(
        frame_dir=data_dir + '/rgb_frames',
        annotation_file_path=data_dir + f'/client_{client_id}_train.txt',
        n_frames=cfgs.seq_len,
        mode='train',
        transform=transform,
        use_albumentations=False,
    )
    train_loader = DataLoader(train_set, batch_size=cfgs.train_bz,
                                num_workers=cfgs.num_workers, 
                                shuffle=True
    )

    val_set = FrameDataset(
        frame_dir=data_dir + '/rgb_frames',
        annotation_file_path=data_dir + f'/val.txt',
        n_frames=cfgs.seq_len,
        mode='test',
        transform=transform,
        use_albumentations=False,
    )
    val_loader = DataLoader(val_set, batch_size=cfgs.test_bz,
                                num_workers=cfgs.num_workers, 
                                shuffle=False
    )

    return train_loader, val_loader

def get_client_local_loaders(client_id, data_dir, work_dir, cfgs):
    '''
    Args:
        client_id (int): id of the client 
        data_dir (str): path that contains <client_id> directory, the results from 
                        data partition step 
        work_dir (str): path to saving logs, checkpoints, ... 
        cfgs: configuration object
    Returns:
        Tuple[torch.utils.data.dataset]: client's local train and val dataset
    '''
    scaler = T.Resize(((cfgs.height, cfgs.width)))
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
		                    std=[0.229, 0.224, 0.225])    
    transform= T.Compose([
        T.ToPILImage(),
        scaler,
        T.ToTensor(),
        normalize      
        ])  
    
    # split client's video ids 8:2 to create local train & val set
    with open(data_dir + f'/client_{client_id}_train.txt') as f:
        lines = [l.strip() for l in f.readlines()]
        video_ids = [l.strip().split(' ')[0] for l in lines]
        labels = [int(l.strip().split(' ')[1]) for l in lines]

    train_inds, val_inds = train_test_split(list(range(len(video_ids))), 
                                            test_size=0.33,
                                            random_state=int(cfgs.seed))
    local_train_path = work_dir + f'/client_{client_id}_local_train.txt'
    local_val_path = work_dir + f'/client_{client_id}_local_val.txt'
    if not os.path.exists(local_train_path):
        with open(local_train_path, 'a') as f:
            for idx in train_inds:
                f.write('{} {}\n'.format(video_ids[idx], labels[idx]))
    if not os.path.exists(local_val_path):
        with open(local_val_path, 'a') as f:
            for idx in val_inds:
                f.write('{} {}\n'.format(video_ids[idx], labels[idx]))
    
    train_set = FrameDataset(
        frame_dir=data_dir + '/rgb_frames',
        annotation_file_path=local_train_path,
        n_frames=cfgs.seq_len,
        mode='train',
        transform=transform,
        use_albumentations=False,
    )
    train_loader = DataLoader(train_set, batch_size=cfgs.train_bz,
                                num_workers=cfgs.num_workers, 
                                shuffle=True
    )

    val_set = FrameDataset(
        frame_dir=data_dir + '/rgb_frames',
        annotation_file_path=local_val_path,
        n_frames=cfgs.seq_len,
        mode='test',
        transform=transform,
        use_albumentations=False,
    )
    val_loader = DataLoader(val_set, batch_size=cfgs.train_bz,
                                num_workers=cfgs.num_workers, 
                                shuffle=False
    )
    return train_loader, val_loader
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Data paritioning")
    parser.add_argument(
        "--n_clients", type=int, required=True, help="Number of clients"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="path to the original data directory",
    )
    parser.add_argument(
        "--train_ann",
        type=str,
        help="path to train annotation files",
    )
    parser.add_argument(
        "--val_ann",
        type=str,
        help="path to val annotation files",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="iid",
        help="Data partition mode"
    )
    args = parser.parse_args()

    data_partition(n_clients=int(args.n_clients),
                    data_dir=args.data_dir,
                    train_annotation_path=args.train_ann,
                    mode=args.mode)