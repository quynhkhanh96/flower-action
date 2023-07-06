# 1. Introduction

# 2. Data preparation

# 3. Setup configuration

# 4. Run experiments
## 4.1. Framework-based experiments
### Data partition
Run this to split the data among the clients
```shell
python -m datasets.frame_dataset --n_clients=2 --data_dir=$DATA_DIR --train_ann=$DATA_DIR/egogesture_train_rawframes.txt --mmaction_base=True --mode="iid"
```
After this you shall see three new files `client_0_train.txt` and `client_1_train.txt` inside `DATA_DIR`.
### FedAvg
- Start server
```shell
CUDA_VISIBLE_DEVICES=1 python -m video_server --server_address=$SERVER_ADDRESS --cfg_path="examples/egogesture/configs/egogesture_fedavg_p05.yaml" --data_dir=$DATA_DIR --work_dir="$DATA_DIR/fedavg_exps"
```
- Start clients
```shell
CUDA_VISIBLE_DEVICES=1 python -m video_client --server_address=$SERVER_ADDRESS --cid=0 --cfg_path="examples/egogesture/configs/egogesture_fedavg_p05.yaml" --data_dir=$DATA_DIR 

CUDA_VISIBLE_DEVICES=2 python -m video_client --server_address=$SERVER_ADDRESS --cid=1 --cfg_path="examples/egogesture/configs/egogesture_fedavg_p05.yaml" --data_dir=$DATA_DIR 
```
### FedBN

### FedPNS

### STC
- Start server
```shell
CUDA_VISIBLE_DEVICES=0 python -m video_server --server_address=$SERVER_ADDRESS --cfg_path="examples/egogesture/configs/egogesture_stc_p05.yaml" --data_dir=$DATA_DIR --work_dir="$DATA_DIR/stc_exps" --p_up=0.04
```
- Start clients
```shell
CUDA_VISIBLE_DEVICES=1 python -m video_client --server_address=$SERVER_ADDRESS --cid=0 --cfg_path="examples/egogesture/configs/egogesture_stc_p05.yaml" --data_dir=$DATA_DIR --p_up=0.04

CUDA_VISIBLE_DEVICES=2 python -m video_client --server_address=$SERVER_ADDRESS --cid=1 --cfg_path="examples/egogesture/configs/egogesture_stc_p05.yaml" --data_dir=$DATA_DIR --p_up=0.04
```
## 4.2. Simulation
### Data partition
```shell
python -m datasets.frame_dataset --n_clients=20 --data_dir=$DATA_DIR --train_ann=$DATA_DIR/egogesture_train_rawframes.txt --mmaction_base=True --mode="iid"
```
### FedAvg
```shell
cd simulation/
python -m fedavg_sim --work_dir="$DATA_DIR/fedavg_sim" --data_dir=$DATA_DIR --server_device="cuda:0" --cfg_path="../examples/egogesture/configs/egogesture_sim_C20_frac025.yaml"
```
### FedBN

### FedPNS

### QSparse Local SGD
```shell
cd simulation/
python -m qsparse_sgd_sim --work_dir="$DATA_DIR/qsparse_sgd_sim" --data_dir=$DATA_DIR --server_device="cuda:0" --cfg_path="../examples/egogesture/configs/egogesture_sim_C8_frac025.yaml"
```

# 5. Results