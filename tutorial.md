# **Installation**
Clone this repo to all the devices where you will run your experiments and then run:
```shell
cd flower-action
bash install.sh
```
You also need to download the checkpoints from [here]() and put the directory `pre-train` inside this directory. 
# **Prepare the data**
First you have your dataset on one machine, before running your federated learning experiments, you need to preprocess your data (i.e convert videos to RGB frames and resize them if neccessary, restructure the directories if they are not in required format) and parition it among the clients. 

## **Convert dataset to required format**
At the beginning your dataset is assumed to be in one of the following formats:
- When person ids are provided:
    ```
    <your_dataset>
    ├── videos
    │   ├── person_000
    │   │   ├── video_000.mp4
    │   │   ├── video_001.mp4
    │   │   └── ...
    │   ├── person_001
    │   │   ├── video_000.mp4
    │   │   ├── video_001.mp4
    │   │   └── ...
    │   └── ...
    ├── train.txt
    └── val.txt
    ```
- When there is no person id:
    ```
    <your_dataset>
    ├── videos
    │   ├── video_000.mp4
    │   ├── video_001.mp4
    │   └── ...
    ├── train.txt
    └── val.txt
    ```
We need to convert it to this format, i.e `.mp4` videos are converted to `.jpg` RGB frames:
```
<your_dataset>
├── videos
│   ├── person_000
│   │   ├── video_000
│   │   │   ├── frame_0000.jpg
│   │   │   ├── frame_0001.jpg
│   │   │   └── ...
│   │   ├── video_001
│   │   │   ├── frame_0000.jpg
│   │   │   ├── frame_0001.jpg
│   │   │   └── ...
│   │   └── ...
│   ├── person_001
│   │   ├── video_000
│   │   │   ├── frame_0000.jpg
│   │   │   ├── frame_0001.jpg
│   │   │   └── ...
│   │   ├── video_001
│   │   │   ├── frame_0000.jpg
│   │   │   ├── frame_0001.jpg
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── train.txt
└── val.txt
```
Run this to obtain that:
```shell
cd tools  
python build_rawframes.py <path/to/your/dataset> <path/to/preprocessed/dataset> 172
```
If there is no person id in your dataset, set `no_person_ids` to `True`, the script will create a "pseudo" person id for each video, i.e one video belongs to one person. 

## **Partition the dataset**
After having the data in the required format, we are ready to partition them among the clients.
```shell 
python -m datasets.frame_dataset --n_clients=4 \
        --data_dir <path/to/preprocessed/dataset> \
        --train_ann=<path/to/preprocessed/dataset>/train.txt \
        --val_ann=<path/to/preprocessed/dataset>/val.txt
```
This script will create lists of person ids (saved as `.txt` files, for eg. `<your_dataset>/client_0.txt`) for each clients, so from them we know each video belongs to which client. Now copy the whole preprocessed dataset directory to all of your clients and server (i.e the machines where you want to run your experiments on).

# Start the experiment
Before running the experiments, one extra step you need to do is to prepare your config file, this is where you set all the configuration about how your data will be processed, the model architecture, training setting (for eg. learning rate, optimizer, ...) and federated learning config (number of rounds, number of clients, participation ratio, ...).
An example from `configs/afosr_mobilenet3d_v2.yaml`:
```yaml
# local data loaders
height: 112
width: 112
seq_len: 16
train_bz: 16
test_bz: 8
num_workers: 4

# network architecture
arch: 'mobilenet3d_v2'
width_mult: 0.45 
num_classes: 12
pretrained_model: 'pre-trained/kinetics_mobilenetv2_0.45x_RGB_16_best.pth'

# local training
local_e: 3
epochs: 30
loss: 'CrossEntropyLoss'
optimizer: 'SGD'
lr: 0.0003
print_freq: 10

# federated learning setting 
FL: 'FedAvg'
frac: 1 # participation ratio
num_C: 4 # client number
min_sample_size: 2
min_num_clients: 2 
seed: 1234 
device: 'cuda'
```

## Start server

## Start clients

## Measure communication cost