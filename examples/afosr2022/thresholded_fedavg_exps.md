# Set the paths
```shell
SERVER_ADDRESS="127.0.0.1:8085"
CFG_PATH="../afosr_resnet183d.yaml"
DATA_DIR="/ext_data2/comvis/khanhdtq/afosr2022"
TRAIN_ANNOTATION_PATH="$DATA_DIR/train.txt"
VAL_ANNOTATION_PATH="$DATA_DIR/val.txt"
```
# Start server
```shell
screen -S fed_server
CUDA_VISIBLE_DEVICES=1 python -m video_server --server_address=$SERVER_ADDRESS --cfg_path=$CFG_PATH --data_dir=$DATA_DIR --work_dir="$DATA_DIR/fed_exps"
```
# Start clients
```shell
screen -S client0
CUDA_VISIBLE_DEVICES=1 python -m thresholded_video_client --server_address=$SERVER_ADDRESS --cid=0 --cfg_path=$CFG_PATH --data_dir=$DATA_DIR --work_dir="$DATA_DIR/fed_exps"
```