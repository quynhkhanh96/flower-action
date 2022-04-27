import os
import sys 
import cv2
import random
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from .utils.video_sampler import *

__all__ = ['VideoDataset']

class VideoDataset(Dataset):
    def __init__(self,
                 video_dir,
                 annotation_file_path,
                 sampler=SystematicSampler(n_frames=16),
                 to_rgb=True,
                 transform=None,
                 use_albumentations=False,
                 ):
        self.video_dir = video_dir
        self.annotation_file_path = annotation_file_path
        self.sampler = sampler
        self.to_rgb = to_rgb
        self.transform = transform
        self.use_albumentations = use_albumentations

        self.clips = []
        self.labels = []

        with open(self.annotation_file_path) as f:
            subject_dirs = [_.strip() for _ in f.readlines()]
            for subject_dir in subject_dirs:
                subject_dir_path = os.path.join(self.video_dir, subject_dir)
                for timestamp_dir in sorted(os.listdir(subject_dir_path)):
                    timestamp_dir_path = os.path.join(subject_dir_path, timestamp_dir)
                    for video_file in filter(lambda _: _.endswith('.mp4'),
                                             sorted(os.listdir(timestamp_dir_path))):
                        label = int(os.path.splitext(video_file)[0]) - 1
                        video_file = os.path.join(subject_dir, timestamp_dir, video_file)
                        self.clips.append((video_file, subject_dir))
                        self.labels.append(label)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, item):
        video_file, subject = self.clips[item]
        video_file = os.path.join(self.video_dir, video_file)
        frames = self.sampler(video_file, sample_id=item)        
        if self.to_rgb:
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        if self.transform is not None:
            frames = [self.transform(frame) if not self.use_albumentations
                      else self.transform(image=frame)['image'] for frame in frames]                
        data = torch.from_numpy(np.stack(frames).transpose((1, 0, 2, 3)))        
        # data shape (c, s, w, h) s for seq_len, c for channel
        return data, self.labels[item], video_file   

def data_partition(n_clients, 
                    video_dir,
                    train_annotation_path, 
                    val_annotation_path,
                    working_dir):
    """
    Args:
        n_clients: number of clients
        data_dir: path to video directory
        train_annotation_path: path to train annotation file, 
                                in which each line is person id 
        val_annotation_path: path to valid annotation file
        working_dir: path to directory which has the following structure:
                        working_dir
                        ├── client_0
                        │   ├── train.txt
                        |   ├── val.txt
                        │   └── videos (contains .mp4 files)
                        └── client_1
                            ├── train.txt
                            ├── val.txt
                            └── videos
    """
    print(f'Partitioning data among {n_clients} clients ...')
    os.makedirs(working_dir, exist_ok=True)
    for i in range(n_clients):
        os.makedirs(working_dir + f'/client_{i}', exist_ok=True)
        os.makedirs(working_dir + f'/client_{i}/videos', exist_ok=True)
        
    with open(train_annotation_path, 'r') as f:
        person_ids = f.readlines()
    person_ids = [x.strip() for x in person_ids]
    random.shuffle(person_ids)

    person_lists = [person_ids[i::n_clients] for i in range(n_clients)]
    for i in range(n_clients):
        person_list = person_lists[i]
        with open(working_dir + f'/client_{i}/train.txt', 'a') as f:
            for person_id in person_list:
                f.write(person_id + '\n')
        os.system(f'cp {val_annotation_path} {working_dir}/client_{i}')
        val_ann_fname = os.path.basename(val_annotation_path)
        if val_ann_fname != 'val.txt':
            os.system(f'mv {working_dir}/client_{i}/{val_ann_fname} {working_dir}/client_{i}/val.txt')

        for person_id in person_list:
            os.system(f'cp -r {video_dir}/{person_id} {working_dir}/client_{i}/videos')

def get_client_loaders(client_id, data_dir, cfgs):
    """
    Args:
        client_id (int): id of the client 
        data_dir (str): path that contains <client_id> directory, the results from 
                        data partition step 
        cfgs: 
    Returns:
        Tuple[torch.utils.data.dataset]: train and val dataset
    """

    scaler = T.Resize(((cfgs.height, cfgs.width)))
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
		                    std=[0.229, 0.224, 0.225])    
    transform= T.Compose([
		T.ToPILImage(),
		scaler,
		T.ToTensor(),
		normalize      
		])  

    train_set = VideoDataset(
        video_dir=data_dir + f'/client_{client_id}/videos',
        annotation_file_path=data_dir + f'/client_{client_id}/train.txt',
        sampler=RandomTemporalSegmentSampler(n_frames=cfgs.seq_len),
        transform=transform,
        use_albumentations=False,
    )
    train_loader = DataLoader(train_set, batch_size=cfgs.train_bz,
                                num_workers=cfgs.num_workers, 
                                shuffle=True
    )

    val_set = VideoDataset(
        video_dir=data_dir + f'/client_{client_id}/videos',
        annotation_file_path=data_dir + f'/client_{client_id}/val.txt',
        sampler=SystematicSampler(n_frames=cfgs.seq_len),
        transform=transform,
        use_albumentations=False,
    )
    val_loader = DataLoader(val_set, batch_size=cfgs.test_bz,
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
        "--video_dir",
        type=str,
        help="path to the original video directory",
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
        "--working_dir",
        type=str,
        help="path to save partition results",
    )
    args = parser.parse_args()

    data_partition(n_clients=int(args.n_clients),
                    video_dir=args.video_dir,
                    train_annotation_path=args.train_ann,
                    val_annotation_path=args.val_ann,
                    working_dir=args.working_dir)
