from ..federated_learning import FedAvgClient
from ..evaluation.classification import test_classifer
import yaml 
import flwr
import torch 
import argparse
from ..utils.parsing import Dict2Class
from ..datasets import *

DEFAULT_SERVER_ADDRESS = "[::]:8080"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Configuration file path",
    )
    parser.add_argument(
        "--working_path",
        type=str,
        help="Where the split is saved",
    )

    client_args = parser.parse_args()
    client_id = int(client_args.cid)

    # Configurations 
    with open(client_args.cfg_path, 'r') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfgs = Dict2Class(cfgs)

    # Logger
    flwr.common.logger.configure(f"client_{client_id}", 
                                    host=client_args.log_host)

    # data loaders 
    train_loader, test_loader = get_cifar_client_loader(client_id, 
                                                        local_bz=cfgs.batch_size, 
                                                        test_bz=cfgs.batch_size, 
                                                        working_dir=client_args.working_dir)    

    # Model
    if cfgs.model == 'CNN':
        from ..models import CNN as Fed_Model
    elif cfgs.model == 'MLP': 
        from ..models import MLP as Fed_Model
    elif cfgs.model == 'ResNet': 
        from ..models import ResNet as Fed_Model

    net = Fed_Model()
    net.to(cfgs.device)

    # Start client
    fedavg_client = FedAvgClient(client_id=client_id,
                            df_train=train_loader,
                            dl_test=test_loader,
                            net=net, 
                            loss_func=torch.nn.CrossEntropyLoss(),
                            eval_fn=test_classifer,
                            args=cfgs
                        )
    flwr.client.start_client(client_args.server_address, fedavg_client)