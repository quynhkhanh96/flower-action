# **Installation**
Clone this repo to all the devices where you will run your experiments and then run:
```shell
cd flower-action
bash install.sh
```
You also need to download the checkpoints from [here](https://drive.google.com/file/d/14lRhTH_eNZmFLeMx0KYlXsnzVtT0gxS-/view?usp=sharing) and put the directory `pre-train` inside this directory. 

# **Set up machines**
This step is optional, serves only for when you don't have the direct access to the machines where clients and server will be running on.
First, on every machine (both server and clients), install and start `openssh` (this step is done only one time):
```shell
sudo systemctl enable ssh
sudo systemctl start ssh
```
Now on server, run this to forward the port:
```shell
ssh -R <port1>:127.0.0.1:<port2> <username_on_client>@<client_ip_address> 
```
Here `<port2>` is port for communication on server, `<port1>` on client (make sure that these ports are not occupied on both machines). For example: 
```shell
ssh -R 8089:127.0.0.1:8085 khanhdtq@14.232.166.97 -p8001
```
After entering the password, type `tail` to keep the session. Do the same for the other clients.


# **Prepare the data**
First you have your dataset on one machine, before running your federated learning experiments, you need to preprocess your data (i.e convert videos to RGB frames and resize them if neccessary, restructure the directories if they are not in required format) and parition it among the clients. 

## **Convert dataset to required format**
Your dataset needs to be the following format, i.e `.mp4` videos are converted to `.jpg` RGB frames:
```
<your_dataset>
├── rgb_frames
│   ├── <video_id1>
│   │   ├── frame_0000.jpg
│   │   ├── frame_0001.jpg
│   │   └── ...
│   ├── <video_id2>
│   │   ├── frame_0000.jpg
│   │   ├── frame_0001.jpg
│   │   └── ...
│   └── ...
├── train.txt
└── val.txt
```
Each line in `train.txt` or `val.txt` is `<video_id> <label>`, where label (one of `0`, `1`, ..., `N_CLASSES-1`) is the corresponding ground truth of video `<video_id>`. 

## **Partition the dataset**
After having the data in the required format, we are ready to partition them among the clients.
```shell 
python -m datasets.frame_dataset --n_clients=4 \
        --data_dir <path/to/preprocessed/dataset> \
        --train_ann <path/to/preprocessed/dataset>/train.txt \
        --val_ann <path/to/preprocessed/dataset>/val.txt
```
This script will create lists of person ids (saved as `.txt` files, for eg. `<your_dataset>/client_0.txt`) for each clients, so from them we know each video belongs to which client. Now copy the whole preprocessed dataset directory to all of your clients and server (i.e the machines where you want to run your experiments on).

# **Start the experiment**
Before running the experiments, one extra step you need to do is to prepare your config file, this is where you set all the configuration about how your data will be processed, the model architecture, training setting (for eg. learning rate, optimizer, ...) and federated learning config (number of rounds, number of clients, participation ratio, ...).
An example from `configs/afosr_mobilenet3d_v2.yaml`:
```yaml
# local data loaders
height: 112 # height of frame image
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
num_C: 4 # number of clients
min_sample_size: 2
min_num_clients: 2 
seed: 1234 
device: 'cuda'
```

## **Start server**
On the machine that is to run the server, set the paths and some variables:
```shell
SERVER_ADDRESS="127.0.0.1:<port2>" # For eg. "127.0.0.1:8085"
CFG_PATH=<path/to/config/file>
DATA_DIR=<path/to/preprocessed/dataset>
TRAIN_ANNOTATION_PATH=<path/to/preprocessed/dataset>/train.txt
VAL_ANNOTATION_PATH=<path/to/preprocessed/dataset>/val.txt
```
In particular:
- `<port2>` in `SERVER_ADDRESS` needs to be the same as what is set in set up machines step. 
- `CFG_PATH` is the path to your .yaml config file, for example: "configs/afosr_movinetA0.yaml"
- `DATA_DIR` is where the preprocessed dataset is on the server that you just made and copy to there.

Now start the server:
```shell
CUDA_VISIBLE_DEVICES=0 python -m video_server 
                --server_address=$SERVER_ADDRESS \
                --cfg_path=$CFG_PATH \
                --data_dir=$DATA_DIR \
                --work_dir=$SERVER_DIR
```
Note that server can run in a machine with no gpus, but it will be slow due to the evaluation step. 
## **Start clients**
On each client, also set the paths first:
```shell
SERVER_ADDRESS="127.0.0.1:<port1>" # For eg. "127.0.0.1:8089"
CFG_PATH=<path/to/config/file>
DATA_DIR=<path/to/preprocessed/dataset>
```
`DATA_DIR` is also where the preprocessed data is on that client. `<port1>` in `SERVER_ADDRESS` needs to be the same as what is set in set up machines step.

Then start the client:
```shell
CUDA_VISIBLE_DEVICES=0 python -m video_client \
                --server_address=$SERVER_ADDRESS \
                --cid=0 --cfg_path=$CFG_PATH \
                --data_dir=$DATA_DIR 
```
Here `cid` is the client id. On other client, for example:
```shell
CUDA_VISIBLE_DEVICES=1 python -m video_client \
                --server_address=$SERVER_ADDRESS \
                --cid=1 --cfg_path=$CFG_PATH \
                --data_dir=$DATA_DIR 
```
Now the model training should start.
## **Measure communication cost**
The step is optional, and if you wish to measure the communication cost caused by the experiment, you need to set this up before starting the server and clients.

On the machine that runs server, open a terminal then run:
```shell
cd communication_cost/
sudo python capture_packets.py <path/to/logs_dir>
```
This script will capture every packet that runs to and from `<port2>` of server and get the size of the packet and log it to a file that is named under format `logs_<HhMmSs>.txt` (where `<HhMmSs>` is the time when the experiment starts, for example `logs_15h31m57s.txt`) under directory `logs_dir`.
After that start the experiment as described in the above steps, when all is done, stop the packet capture script (by typing `Ctrl` + `C`) and then run the following to compute the total communication cost (in megabytes):
```shell
python summarize_comcost.py <path/to/log_file>
```
