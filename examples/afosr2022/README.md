# 1. Introduction
- What is this dataset?
- Directory structure:
```
    afosr2022
    ├── data
    │   ├── <person_id1>
    │   │   ├── <date_1>
    │   │   │   ├── 01.csv
    │   │   │   ├── 01.mp4
    │   │   │   ├── 02.csv
    │   │   │   ├── 02.mp4
    │   │   │   ├── ...
    │   │   │   ├── 12.csv
    │   │   │   └── 12.mp4
    │   │   └── <date_2>
    │   └── <person_id2>
    ├── train.txt
    └── val.txt
```
The dataset's directory structure is described in the above tree. Hereby `train.txt` and `val.txt` are lists of person ids used for train and validation respectively (i.e each line in these files is a person id), for now we have to re-create them so they will be in the format `<video_id> <label>` instead. Inside each `<date_i>` directory are 12 videos (`.mp4` files, each corresponding to one of 12 classes) that were recorded on that date.    
# 2. Data preparation
The dataset will be converted to the required format described in the tutorial by running this script (make sure you are inside the `examples/afosr2022` directory):
```shell
python build_rawframes.py --src_dir=<path/to/your_dataset> \
                        --dst_dir=<path/to/your_preprocessed_dataset> \
                        --new_height=<frame_height> \
                        --new_width=<frame_width> 
```
In particular, 
- `src_dir`: path to AFOSR2022 dataset, inside this path are `data` directory, `train.txt` and `val.txt`. 
- `dst_dir`: path to the preprocessed dataset, where `rgb_frames` directory and the re-created train.txt and val.txt will be.
- `new_height` and `new_width` are the height and width of video frame you want to resize to, for example height = width = 224. 
# 3. Setup configuration

# 4. Run experiments
## Data partition
```shell
python -m datasets.frame_dataset --n_clients=3 --data_dir=$DATA_DIR --train_ann=$DATA_DIR/train.txt --mode="iid"
```
## FedBN
### Start server
```shell
CUDA_VISIBLE_DEVICES=1 python -m video_server --server_address=$SERVER_ADDRESS --cfg_path="examples/afosr2022/configs/afosr_fedbn_resnet183d.yaml" --data_dir=$DATA_DIR --work_dir="$DATA_DIR/fedbn_exps"
```

### Start clients
```shell
CUDA_VISIBLE_DEVICES=1 python -m video_client --server_address=$SERVER_ADDRESS --cid=0 --cfg_path="examples/afosr2022/configs/afosr_fedbn_resnet183d.yaml" --data_dir=$DATA_DIR --work_dir="$DATA_DIR/fedbn_exps"

CUDA_VISIBLE_DEVICES=2 python -m video_client --server_address=$SERVER_ADDRESS --cid=1 --cfg_path="examples/afosr2022/configs/afosr_fedbn_resnet183d.yaml" --data_dir=$DATA_DIR --work_dir="$DATA_DIR/fedbn_exps"

CUDA_VISIBLE_DEVICES=2 python -m video_client --server_address=$SERVER_ADDRESS --cid=2 --cfg_path="examples/afosr2022/configs/afosr_fedbn_resnet183d.yaml" --data_dir=$DATA_DIR --work_dir="$DATA_DIR/fedbn_exps"
```