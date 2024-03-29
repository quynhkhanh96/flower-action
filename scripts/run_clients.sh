set -e
cfg_path=$1
WORKING_DIR=$2
# Run script to partition data among clients first, the split result will be in `WORKING_DIR` 
echo "Partitioning data among clients..."
python ..datasets/cifar.py $WORKING_DIR $cfg_path 
# python ..datasets/mnist.py $WORKING_DIR $cfg_path

# SERVER_ADDRESS="[::]:8080"
SERVER_ADDRESS="127.0.0.1:8085"
NUM_CLIENTS=5 

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    python -m classification_client \
      --cid=$i \
      --cfg_path=$cfg_path \
      --working_path=$WORKING_DIR \
      --server_address=$SERVER_ADDRESS &
done
echo "Started $NUM_CLIENTS clients."
