set -e
cfg_path=$1
data_dir=$2
working_dir=$3

SERVER_ADDRESS="127.0.0.1:8085"

CUDA_VISIBLE_DEVICES=2 python -m classification_server \
    --cfg_path=$cfg_path \
    --server_address=$SERVER_ADDRESS \
    --data_dir=$data_dir \
    --working_dir=$working_dir
