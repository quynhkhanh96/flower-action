# 1. Introduction

# 2. Data preparation

# 3. Setup configuration

# 4. Run experiments
## 4.1. Framework-based experiments
### Data partition
Run this to split the data among the clients
```shell
python -m datasets.frame_dataset --n_clients=3 --data_dir=$DATA_DIR --train_ann=$DATA_DIR/egogesture_train_rawframes.txt --mmaction_base=True --mode="iid"
```
After this you shall see three new files `client_0_train.txt`, `client_1_train.txt` and `client_2_train.txt` inside `DATA_DIR`.
### FedAvg
- Start server
```shell

```
- Start clients
```shell

```
### FedBN

### FedPNS

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

# 5. Results