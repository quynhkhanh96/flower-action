# 1. Introduction

# 2. Data preparation
Follow the steps described [here](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/hmdb51) to obtain the data in the structure below:
```
├── hmdb51
│   ├── hmdb51_{train,val}_split_{1,2,3}_rawframes.txt
│   ├── hmdb51_{train,val}_split_{1,2,3}_videos.txt
│   ├── rawframes
│   │   ├── brush_hair
│   │   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0
│   │   │   │   ├── img_00001.jpg
│   │   │   │   ├── img_00002.jpg
│   │   │   │   ├── ...
│   │   ├── ...
│   │   ├── wave
│   │   │   ├── 20060723sfjffbartsinger_wave_f_cm_np1_ba_med_0
│   │   │   ├── ...
│   │   │   ├── winKen_wave_u_cm_np1_ri_bad_1
```
# 3. Run experiments
Before run the scripts, set the paths first by put these links in a file called `set_hmdb51_paths.sh` 
```shell
SERVER_ADDRESS="127.0.0.1:8085"
DATA_DIR=<path/to/your/preprocessed/hmdb51/dataset>
```
then run `source set_hmdb51_paths.sh`
## 3.1. Data partition
Run this to split the data among the clients
```shell
python -m datasets.frame_dataset --n_clients=3 --data_dir=$DATA_DIR --train_ann=$DATA_DIR/hmdb51_train_split_1_rawframes.txt --mmaction_base=True --mode="iid"
```
After this you shall see three new files `client_0_train.txt`, `client_1_train.txt` and `client_2_train.txt` inside `DATA_DIR`.
## 3.2. Start server
```shell
CUDA_VISIBLE_DEVICES=1 python -m video_server --server_address=$SERVER_ADDRESS --cfg_path="examples/hmdb51/configs/hmdb51_fedavg_p07.yaml" --data_dir=$DATA_DIR --work_dir="$DATA_DIR/fedavg_exps"
```
## 3.3. Start clients
```shell
CUDA_VISIBLE_DEVICES=1 python -m video_client --server_address=$SERVER_ADDRESS --cid=0 --cfg_path="examples/hmdb51/configs/hmdb51_fedavg_p07.yaml" --data_dir=$DATA_DIR 

CUDA_VISIBLE_DEVICES=2 python -m video_client --server_address=$SERVER_ADDRESS --cid=1 --cfg_path="examples/hmdb51/configs/hmdb51_fedavg_p07.yaml" --data_dir=$DATA_DIR 

CUDA_VISIBLE_DEVICES=2 python -m video_client --server_address=$SERVER_ADDRESS --cid=2 --cfg_path="examples/hmdb51/configs/hmdb51_fedavg_p07.yaml" --data_dir=$DATA_DIR 
```