import os
import sys
import time
import logging
import collections
import csv
import pickle 

import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import yaml

# ************************** Dataset and augmentations **************************
class Landmarks(Dataset):

    def __init__(self, data_dir, allfiles, dataidxs=None, train=True, 
                transform=None, target_transform=None,
                download=False):
        """
        allfiles is [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ...  
                     {'user_id': xxx, 'image_id': xxx, 'class': xxx} ... ]
        """
        self.allfiles = allfiles
        if dataidxs == None:
            self.local_files = self.allfiles
        else:
            self.local_files = self.allfiles[dataidxs[0]: dataidxs[1]]
        self.data_dir = data_dir
        self.dataidxs = dataidxs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.local_files)

    def __getitem__(self, idx):
        img_name = self.local_files[idx]['image_id']
        label = int(self.local_files[idx]['class'])

        img_name = os.path.join(self.data_dir, str(img_name) + ".jpg")

        # convert jpg to PIL (jpg -> Tensor -> PIL)
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

class GLD23kDataset(Dataset):
    def __init__(self, data_dir, image_ids,
                    transform=None, target_transform=None):
        """
            image_ids = [(image_id, label), ...]
        """
        self.data_dir = data_dir
        self.local_files = [image_id for (image_id, _) in image_ids]
        self.labels = [int(label) for (_, label) in image_ids]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.local_files)

    def __getitem__(self, idx):
        img_name = self.local_files[idx]
        label = self.labels[idx]

        img_path = os.path.join(self.data_dir, str(img_name) + ".jpg")

        # convert jpg to PIL (jpg -> Tensor -> PIL)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_landmarks():
    # IMAGENET_MEAN = [0.5071, 0.4865, 0.4409]
    # IMAGENET_STD = [0.2673, 0.2564, 0.2762]

    IMAGENET_MEAN = [0.5, 0.5, 0.5]
    IMAGENET_STD = [0.5, 0.5, 0.5]

    image_size = 224
    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, valid_transform

# ************************** Data partition **************************
def _read_csv(path: str):
  """Reads a csv file, and returns the content inside a list of dictionaries.
  Args:
    path: The path to the csv file.
  Returns:
    A list of dictionaries. Each row in the csv file will be a list entry. The
    dictionary is keyed by the column names.
  """
  with open(path, 'r') as f:
    return list(csv.DictReader(f))

# TODO: rewrite this function: get_mapping_per_user(fn, n_clients),
# for now it does not perform data partitioning given user defined number of clients
def get_mapping_per_user(fn):
    """
    mapping_per_user is {'user_id': [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ... {}], 
                         'user_id': [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ... {}],
    } or               
                        [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ...  
                         {'user_id': xxx, 'image_id': xxx, 'class': xxx} ... ]
    }
    """
    mapping_table = _read_csv(fn)
    expected_cols = ['user_id', 'image_id', 'class']
    if not all(col in mapping_table[0].keys() for col in expected_cols):
        raise ValueError(
            'The mapping file must contain user_id, image_id and class columns. '
            'The existing columns are %s' % ','.join(mapping_table[0].keys()))

    data_local_num_dict = dict()

    mapping_per_user = collections.defaultdict(list)
    data_files = []
    net_dataidx_map = {}
    sum_temp = 0

    for row in mapping_table:
        user_id = row['user_id']
        mapping_per_user[user_id].append(row)
    for user_id, data in mapping_per_user.items():
        num_local = len(mapping_per_user[user_id])
        net_dataidx_map[int(user_id)]= (sum_temp, sum_temp+num_local)
        data_local_num_dict[int(user_id)] = num_local
        sum_temp += num_local
        data_files += mapping_per_user[user_id]
    assert sum_temp == len(data_files)

    return data_files, data_local_num_dict, net_dataidx_map

def get_dataloader_Landmarks(datadir, train_files, test_files, 
                            train_bs, test_bs, dataidxs=None):

    transform_train, transform_test = _data_transforms_landmarks()

    train_ds = Landmarks(datadir, train_files, dataidxs=dataidxs, 
                        train=True, transform=transform_train, download=True)
    test_ds = Landmarks(datadir, test_files, dataidxs=None, 
                        train=False, transform=transform_test, download=True)

    train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl

def get_dataloader(dataset, datadir, 
                    train_files, test_files, 
                    train_bs, test_bs, 
                    dataidxs=None):
    return get_dataloader_Landmarks(datadir, train_files, test_files, 
                                    train_bs, test_bs, dataidxs)

def get_dataloaders(data_dir, train_ids, test_ids, local_bz, test_bz):
    transform_train, transform_test = _data_transforms_landmarks()

    train_ds = GLD23kDataset(data_dir, train_ids, transform_train)
    test_ds = GLD23kDataset(data_dir, test_ids, transform_test)

    train_dl = DataLoader(dataset=train_ds, batch_size=local_bz, 
                            shuffle=True, drop_last=False)
    test_dl = DataLoader(dataset=test_ds, batch_size=test_bz, 
                            shuffle=False, drop_last=False)

    return train_dl, test_dl

def load_partition_data_landmarks(dataset, data_dir, fed_train_map_file, fed_test_map_file, 
                            partition_method=None, partition_alpha=None, client_number=233, batch_size=10):

    train_files, data_local_num_dict, net_dataidx_map = get_mapping_per_user(fed_train_map_file)
    test_files = _read_csv(fed_test_map_file)

    class_num = len(np.unique([item['class'] for item in train_files]))
    train_data_num = len(train_files)

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, 
                                                        train_files, test_files, 
                                                        batch_size, batch_size)
    test_data_num = len(test_files)

    # get local dataset
    train_data_local_dict = dict()
    test_data_local_dict = dict()


    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = dataidxs[1] - dataidxs[0]
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, 
                                                            train_files, test_files, 
                                                            batch_size, batch_size,
                                                            dataidxs)
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    return train_data_local_dict, test_data_local_dict, class_num

# def landmark_partition(n_clients, fed_train_map_file, fed_test_map_file, working_dir):
#     train_files, _, dataidx_map = get_mapping_per_user(fed_train_map_file)
#     test_files = _read_csv(fed_test_map_file)

#     # Ugly hackround that works for 2 clients 
#     net_dataidx_map = {}
#     net_dataidx_map[0] = (dataidx_map[0][0], dataidx_map[len(dataidx_map) // 2][1])
#     net_dataidx_map[1] = (dataidx_map[len(dataidx_map) // 2][1], dataidx_map[len(dataidx_map) - 1][1])
#     # save metadata
#     filehandler = open(working_dir + '/train_files.pkl', 'wb')
#     pickle.dump(train_files, filehandler)
#     filehandler.close()

#     filehandler = open(working_dir + '/test_files.pkl', 'wb')
#     pickle.dump(test_files, filehandler)
#     filehandler.close()

#     filehandler = open(working_dir + '/idx_map.pkl', 'wb')
#     pickle.dump(net_dataidx_map, filehandler)
#     filehandler.close()

def landmark_partition(n_clients, fed_train_map_file, fed_test_map_file, working_dir):
    """
        net_dataidx_map = {
            client_id: list of image_id
        }
    """
    df_train = pd.read_csv(fed_train_map_file)
    df_test = pd.read_csv(fed_test_map_file)

    sample_count = collections.Counter(df_train['class'].tolist())
    num_samples_per_client = {k: v // n_clients for k, v in sample_count.items()}

    class_to_image_ids = df_train.groupby('class')['image_id'].agg(list)
    net_dataidx_map = {client_id: [] for client_id in range(n_clients)}
    for i, client_id in enumerate(range(n_clients)):
        for c, class_image_ids in class_to_image_ids.items():
            n_samples = num_samples_per_client[c]
            class_samples = [(smpl, c) for smpl in class_image_ids[i * n_samples: (i + 1) * n_samples]]
            net_dataidx_map[client_id].extend(class_samples)

    # train_files = df_train['image_id'].tolist()
    test_files = []
    for _, row in df_test.iterrows():
        test_files.append((row['image_id'], row['class']))

    # filehandler = open(working_dir + '/train_files.pkl', 'wb')
    # pickle.dump(train_files, filehandler)
    # filehandler.close()

    filehandler = open(working_dir + '/test_files.pkl', 'wb')
    pickle.dump(test_files, filehandler)
    filehandler.close()

    filehandler = open(working_dir + '/idx_map.pkl', 'wb')
    pickle.dump(net_dataidx_map, filehandler)
    filehandler.close()

# def get_landmark_client_loader(client_id, local_bz, test_bz, data_dir, working_dir):
#     with open(working_dir + '/train_files.pkl', 'rb') as fp:
#         train_files = pickle.load(fp)

#     with open(working_dir + '/test_files.pkl', 'rb') as fp:
#         test_files = pickle.load(fp)

#     with open(working_dir + '/idx_map.pkl', 'rb') as fp:
#         net_dataidx_map = pickle.load(fp)

#     dataidxs = net_dataidx_map[client_id]
#     train_loader, test_loader = get_dataloader(None, data_dir, 
#                                                 train_files, test_files, 
#                                                 local_bz, test_bz,
#                                                 dataidxs)

#     class_num = len(np.unique([item['class'] for item in train_files]))

#     return train_loader, test_loader, class_num

def get_landmark_client_loader(client_id, local_bz, test_bz, data_dir, working_dir):
    # with open(working_dir + '/train_files.pkl', 'rb') as fp:
    #     train_files = pickle.load(fp)

    with open(working_dir + '/test_files.pkl', 'rb') as fp:
        test_files = pickle.load(fp)

    with open(working_dir + '/idx_map.pkl', 'rb') as fp:
        net_dataidx_map = pickle.load(fp)

    client_image_ids = net_dataidx_map[client_id]
    train_loader, test_loader = get_dataloaders(data_dir,
                                                client_image_ids, test_files,
                                                local_bz, test_bz)

    num_classes = len(np.unique([label for (_, label) in client_image_ids]))
    return train_loader, test_loader, num_classes 

if __name__ == '__main__':

    working_dir = sys.argv[1]
    cfg_path = sys.argv[2]
    fed_train_map_file = sys.argv[3]
    fed_test_map_file = sys.argv[4]

    with open(cfg_path, 'r') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)

    landmark_partition(n_clients=cfgs['num_C'], 
                        fed_train_map_file=fed_train_map_file, 
                        fed_test_map_file=fed_test_map_file, 
                        working_dir=working_dir)

    # data_dir = '/home/dothi/Desktop/gld/images'
    # fed_g23k_train_map_file = '/home/dothi/Desktop/gld/data_user_dict/gld23k_user_dict_train.csv'
    # fed_g23k_test_map_file = '/home/dothi/Desktop/gld/data_user_dict/gld23k_user_dict_test.csv'

    # client_number = 233
    # fed_train_map_file = fed_g23k_train_map_file
    # fed_test_map_file = fed_g23k_test_map_file

    # train_data_local_dict, test_data_local_dict, class_num = \
    #     load_partition_data_landmarks(None, data_dir, fed_train_map_file, fed_test_map_file, 
    #                         partition_method=None, partition_alpha=None, 
    #                         client_number=client_number, batch_size=10)

    # for client_idx in range(client_number):
    #     for i, (data, label) in enumerate(train_data_local_dict[client_idx]):
    #         print(data.shape)
    #         print(label)
    #         if i > 3:
    #             break 
