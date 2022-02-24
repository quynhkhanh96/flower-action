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
        /ext_data2/comvis/khanhdtq/gld/data_user_dict/gld23k_user_dict_test.csv /home/dothi/Desktop/gld/images

    ./run_server.sh configs/gld23k.yaml /home/dothi/Desktop/gld/images ../working

    

