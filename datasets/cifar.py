import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import sys 
import yaml 

from partitioners import split_iid

DATA_PATH = '../data'
# Define a PyTorch Dataset for CIFAR 
# that expects a list of (image, label) pairs
# TODO: write a common Dataset for all classification problem 
# in common.py. The dataset class below should be subclass of 
# that one.
class CIFAR10Dataset(Dataset):
    """
    A custom Dataset class for images
    Args:
        inputs: numpy array [n_data x shape]
        labels: numpy array [n_data (x 1)]
    """

    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return self.inputs.shape[0]

def get_default_data_transforms():
    transforms_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transforms_eval = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    return transforms_train, transforms_eval

def cifar_partition(n_clients, classes_per_client, working_dir):
    # Get images and labels
    data_train = torchvision.datasets.CIFAR10(DATA_PATH, train=True, download=True)
    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)

    data_train = [(i, label) for i, label in enumerate(y_train)]
    clients_split = split_iid(data_train, n_clients, classes_per_client)
    np.save(working_dir + '/splits.npy', clients_split)

def get_cifar_client_loader(client_id, local_bz, test_bz, working_dir):
    data_train = torchvision.datasets.CIFAR10(DATA_PATH, train=True, download=False)
    data_test = torchvision.datasets.CIFAR10(DATA_PATH, train=False, download=True)

    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)

    transforms_train, transforms_eval = get_default_data_transforms()

    clients_split = np.load(working_dir + '/splits.npy')
    sample_inds, client_labels = clients_split[int(client_id)]
    client_images = x_train[sample_inds]
    client_dst = CIFAR10Dataset(client_images, client_labels, transforms_train)
    client_dl = DataLoader(client_dst, batch_size=local_bz,
                    shuffle=True, num_workers=2
    )

    test_dl = DataLoader(CIFAR10Dataset(x_test, y_test, transforms_eval),
                    batch_size=test_bz, shuffle=False
    )

    return client_dl, test_dl

def get_cifar_test_loader(test_bz):
    data_test = torchvision.datasets.CIFAR10(DATA_PATH, train=False, download=True)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets) 

    _, transforms_eval = get_default_data_transforms()   
    test_dl = DataLoader(CIFAR10Dataset(x_test, y_test, transforms_eval),
                    batch_size=test_bz, shuffle=False
    )

    return test_dl 

# Get datasets for each client 
def get_cifar_loaders(n_clients, classes_per_client, local_bz, test_bz):
    """
    Args: 
        n_clients (int): number of clients
        classes_per_clients (int): number of classes in each clients
    Returns:
        List[Dataset]: list of clients' datasets
        Dataset: test dataset
    """
    # Get images and labels
    data_train = torchvision.datasets.CIFAR10(DATA_PATH, train=True, download=True)
    data_test = torchvision.datasets.CIFAR10(DATA_PATH, train=False, download=True)

    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)

    data_train = [(i, label) for i, label in enumerate(y_train)]
    clients_split = split_iid(data_train, n_clients, classes_per_client)

    client_loaders = []
    transforms_train, transforms_eval = get_default_data_transforms()
    for split in clients_split:
        sample_inds, client_labels = split
        client_images = x_train[sample_inds]

        client_dst = CIFAR10Dataset(client_images, client_labels, transforms_train)
        client_dl = DataLoader(client_dst, batch_size=local_bz,
                        shuffle=True, num_workers=2
        )
        client_loaders.append(client_dl)
    
    test_loader = DataLoader(CIFAR10Dataset(x_test, y_test, transforms_eval),
                    batch_size=test_bz, shuffle=False
    )

    return client_loaders, test_loader

if __name__ == '__main__':

    assert len(sys.argv) == 3, print('working directory and configuration file path are expected.')
    working_dir = sys.argv[1]
    cfg_path = sys.argv[2]

    with open(cfg_path, 'r') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)

    cifar_partition(n_clients=cfgs['num_C'], classes_per_client=cfgs['Nc'], 
                    working_dir=working_dir)
