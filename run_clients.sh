set -e
cfg_path=$1
WORKING_DIR=$2
fed_train_map_file=$3 #TODO: set default for this argument 
fed_test_map_file=$4
data_dir=$5
# Run script to partition data among clients first, the split result will be in `WORKING_DIR` 
echo "Partitioning data among clients..."
# python -m datasets.cifar $WORKING_DIR $cfg_path 
# python ..datasets/mnist.py $WORKING_DIR $cfg_path
python -m datasets/google_landmark_2020 $WORKING_DIR $cfg_path \
  $fed_train_map_file $fed_test_map_file

# SERVER_ADDRESS="[::]:8080"
SERVER_ADDRESS="127.0.0.1:8085"
NUM_CLIENTS=2 #5 

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    CUDA_VISIBLE_DEVICES=1,2 python -m classification_client \
      --cid=$i \
      --cfg_path=$cfg_path \
      --working_dir=$WORKING_DIR \
      --server_address=$SERVER_ADDRESS \
      --data_dir=$data_dir & 
done
echo "Started $NUM_CLIENTS clients."
