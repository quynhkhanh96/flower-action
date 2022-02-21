import argparse
import yaml 
import flwr 
import functools
from utils.parsing import Dict2Class
from evaluation.classification import test_classifer
from datasets import *
from federated_learning.server.fedadam_server import FedAdamStrategy
from flwr.common import weights_to_parameters
import torch 
import numpy as np 

DEFAULT_SERVER_ADDRESS = "[::]:8080"

def get_eval_fn(cfgs):
    if cfgs.dataset == 'cifar10':
        test_loader = get_cifar_test_loader(test_bz=cfgs.batch_size)
    elif cfgs.dataset == 'mnist':
        test_loader = get_mnist_test_loader(test_bz=cfgs.batch_size)
    else:
        raise ValueError(f'No implementation for {cfgs.dataset} dataset')
    
    return functools.partial(test_classifer, test_loader=test_loader)

def fit_config(rnd, cfgs):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(cfgs.local_e),
        "batch_size": str(cfgs.batch_size)
    }
    return config

if __name__ == '__main__':
    """Start server and train for some rounds."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
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
    server_args = parser.parse_args()

    # configuration
    with open(server_args.cfg_path, 'r') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfgs = Dict2Class(cfgs)

    # get intial parameters 
    if cfgs.model == 'CNN':
        from models import CNN as Fed_Model
    elif cfgs.model == 'MLP': 
        from models import MLP as Fed_Model
    elif cfgs.model == 'ResNet': 
        from models import ResNet as Fed_Model

    net = Fed_Model()
    initial_weights = net.get_weights()
    for i, w in enumerate(initial_weights):
        try:
            _ = len(w)
        except:
            initial_weights[i] = np.array([0])
    initial_parameters = weights_to_parameters(initial_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create strategy
    strategy = FedAdamStrategy(
        model_name=cfgs.model,
        device=device,
        fraction_fit=cfgs.frac,
        min_fit_clients=cfgs.min_sample_size,
        min_available_clients=cfgs.min_num_clients,
        eval_fn=get_eval_fn(cfgs),
        on_fit_config_fn=functools.partial(fit_config, cfgs=cfgs),
        initial_parameters=initial_parameters
    )

    # Configure logger and start server
    flwr.common.logger.configure("server", host=server_args.log_host)
    flwr.server.start_server(
        server_args.server_address,
        config={"num_rounds": cfgs.epochs},
        strategy=strategy,
    )
