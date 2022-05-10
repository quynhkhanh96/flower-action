import os
import cv2
import random
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

class FrameDataset(Dataset):
    def __init__(self, frame_dir,
                    annotation_file_path,
                    n_frames,
                    mode='train',
                    to_rgb=False,
                    transform=None,
                    use_albumentations=False):
        self.frame_dir = frame_dir
        self.annotation_file_path = annotation_file_path
        self.n_frames = n_frames

        self.to_rgb = to_rgb
        self.transform = transform
        self.use_albumentations = use_albumentations

        self.mode = mode 

        self.clips = []
        self.labels = []

        with open(self.annotation_file_path) as f:
            subject_dirs = [_.strip() for _ in f.readlines()]

            for subject_dir in subject_dirs:
                subject_dir_path = os.path.join(self.frame_dir, subject_dir)

                for timestamp_dir in sorted(os.listdir(subject_dir_path)):
                    timestamp_dir_path = os.path.join(subject_dir_path, timestamp_dir)

                    for video_file in sorted(os.listdir(timestamp_dir_path)):
                        label = int(os.path.splitext(video_file)[0]) - 1
                        video_file = os.path.join(subject_dir, timestamp_dir, video_file)
                        self.clips.append((video_file, subject_dir))
                        self.labels.append(label)

    def __len__(self):
        return len(self.clips)

    def sample_frames(self, video_file):
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

    def __getitem__(self, item):
        video_file, _ = self.clips[item]
        video_file = os.path.join(self.frame_dir, video_file)

        """ Sample video frames """
        frames = self.sample_frames(video_file)

        """ Transform and augment RGB images"""
        if self.to_rgb:
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        if self.transform is not None:
            frames = [self.transform(frame) if not self.use_albumentations
                      else self.transform(image=frame)['image'] for frame in frames]                
        data = torch.from_numpy(np.stack(frames).transpose((1, 0, 2, 3)))        
        # data shape (c, s, w, h) where s is seq_len, c is number of channels
        return data, self.labels[item], video_file  

def data_partition(n_clients, 
                    data_dir,
                    train_annotation_path, 
                    val_annotation_path):
    """
    Args:
        n_clients: number of clients
        data_dir: path to data (including `videos`, `train.txt`, `val.txt`) directory
        train_annotation_path: path to train annotation file, 
                                in which each line is person id 
        val_annotation_path: path to valid annotation file

    """
    print(f'Partitioning data among {n_clients} clients ...')
        
    with open(train_annotation_path, 'r') as f:
        person_ids = f.readlines()
    person_ids = [x.strip() for x in person_ids]
    random.shuffle(person_ids)
    
    with open(val_annotation_path, 'r') as f:
        val_person_ids = f.readlines()
    val_person_ids = [x.strip() for x in val_person_ids]

    person_lists = [person_ids[i::n_clients] for i in range(n_clients)]
    for i in range(n_clients):
        person_list = person_lists[i]
        with open(data_dir + f'/client_{i}_train.txt', 'a') as f:
            for person_id in person_list:
                f.write(person_id + '\n')

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
    args = parser.parse_args()

    data_partition(n_clients=int(args.n_clients),
                    data_dir=args.data_dir,
                    train_annotation_path=args.train_ann,
                    val_annotation_path=args.val_ann)




