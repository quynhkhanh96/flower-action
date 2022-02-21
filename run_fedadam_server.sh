set -e
cfg_path=$1

SERVER_ADDRESS="127.0.0.1:8085"

python -m classification_fedadam_server \
    --cfg_path=$cfg_path \
    --server_address=$SERVER_ADDRESS 