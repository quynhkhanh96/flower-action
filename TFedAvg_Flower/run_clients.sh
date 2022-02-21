set -e

# SERVER_ADDRESS="[::]:8080"
SERVER_ADDRESS="127.0.0.1:8085"
NUM_CLIENTS=5 

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    python -m client \
      --cid=$i \
      --server_address=$SERVER_ADDRESS &
done
echo "Started $NUM_CLIENTS clients."