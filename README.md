# Classification problems

Run the following:

**MNIST**

    sudo /home/khanh/anaconda3/envs/demo_sr_env/bin/python capture_packets.py
    ssh -R 8085:127.0.0.1:8085 khanhdtq@172.16.77.146
    ./run_server.sh configs/mnist.yaml
    ./run_clients.sh configs/mnist.yaml ../working

**GLD23k**

    sudo /home/khanh/anaconda3/envs/demo_sr_env/bin/python capture_packets.py
    ssh -R 8085:127.0.0.1:8085 khanhdtq@172.16.77.146

    ./run_client.sh configs/gld23k.yaml ../working /ext_data2/comvis/khanhdtq/gld/data_user_dict/gld23k_user_dict_train.csv \
        /ext_data2/comvis/khanhdtq/gld/data_user_dict/gld23k_user_dict_test.csv /ext_data2/comvis/khanhdtq/gld/images

    ./run_server.sh configs/gld23k.yaml /home/dothi/Desktop/gld/images ../working

or in a more manually manner:

    CUDA_VISIBLE_DEVICES=0 python -m classification_client --cid=0 --cfg_path="configs/gld23k.yaml" --working_dir="../working" --server_address="127.0.0.1:8085" --data_dir="/ext_data2/comvis/khanhdtq/gld/images"

    CUDA_VISIBLE_DEVICES=1 python -m classification_client --cid=1 --cfg_path="configs/gld23k.yaml" --working_dir="../working" --server_address="127.0.0.1:8085" --data_dir="/ext_data2/comvis/khanhdtq/gld/images"

**HMDB51**

Download and extract HMDB51 dataset:

    ./hmdb_prepare_data.sh

Set the paths:

    WORKING_DIR="../working"
    CFG_PATH="configs/hmdb51.yaml"
    DATA_DIR=""
    TEST_TRAIN_SPILTS_PATH=""

First split the dataset among the clients by run:
    
    python -m dataset.hmdb51 $WORKING_DIR $CFG_PATH $test_train_splits_path

Start the server:

    SERVER_ADDRESS="127.0.0.1:8085"
    python -m classification_server --cfg_path=$CFG_PATH --server_address=$SERVER_ADDRESS --data_dir=$data_dir --working_dir=$WORKING_DIR

Start the clients:

    SERVER_ADDRESS="127.0.0.1:8085"

    CUDA_VISIBLE_DEVICES=0 python -m classification_client --cid=0 --cfg_path=$CFG_PATH --working_dir=$WORKING_DIR --server_address=$SERVER_ADDRESS --data_dir=$DATA_DIR

    CUDA_VISIBLE_DEVICES=1 python -m classification_client --cid=1 --cfg_path=$CFG_PATH --working_dir=$WORKING_DIR --server_address=$SERVER_ADDRESS --data_dir=$DATA_DIR
    

