# Classification problems

Run the following:

    sudo /home/khanh/anaconda3/envs/demo_sr_env/bin/python capture_packets.py
    ssh -R 8085:127.0.0.1:8085 khanhdtq@172.16.77.146
    ./run_fedadam_server.sh configs/mnist.yaml
    ./run_fedadam_clients.sh configs/mnist.yaml ../working
