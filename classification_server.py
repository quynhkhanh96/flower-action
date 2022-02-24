import argparse
import yaml 
import flwr 
import functools
from utils.parsing import Dict2Class
from evaluation.classification import test_classifer
from datasets import *
from federated_learning.server.fedavg_server import FedAvgStrategy

DEFAULT_SERVER_ADDRESS = "[::]:8080"

def get_eval_fn(cfgs, server_args):
    if cfgs.dataset == 'cifar10':
        test_loader = get_cifar_test_loader(test_bz=cfgs.batch_size)
        num_classes = 10
    elif cfgs.dataset == 'mnist':
        test_loader = get_mnist_test_loader(test_bz=cfgs.batch_size)
        num_classes = 10 
    elif cfgs.dataset == 'gld23k':
        if server_args.data_dir == '':
            raise ValueError('`data_dir` (path to image directory) for gld23k is missing.')
        _, test_loader, num_classes = get_landmark_client_loader(
            0, local_bz=cfgs.batch_size, test_bz=cfgs.batch_size,
            data_dir=server_args.data_dir, 
            working_dir=server_args.working_dir
        )
    else:
        raise ValueError(f'No implementation for {cfgs.dataset} dataset')
    
    return functools.partial(test_classifer, test_loader=test_loader), num_classes

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
    parser.add_argument(
        "--data_dir",
        type=str,
        default='', # only GLD23k dataset needs it 
        help="image directory",
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        help="Where the split is saved",
    )
    server_args = parser.parse_args()

    # configuration
    with open(server_args.cfg_path, 'r') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfgs = Dict2Class(cfgs)

    eval_fn, num_classes = get_eval_fn(cfgs, server_args)
    # Create strategy
    strategy = FedAvgStrategy(
        model_name=cfgs.model,
        num_classes=num_classes,
        device=cfgs.device,
        fraction_fit=cfgs.frac,
        min_fit_clients=cfgs.min_sample_size,
        min_available_clients=cfgs.min_num_clients,
        eval_fn=eval_fn,
        on_fit_config_fn=functools.partial(fit_config, cfgs=cfgs),
    )

    # Configure logger and start server
    flwr.common.logger.configure("server", host=server_args.log_host)
    flwr.server.start_server(
        server_args.server_address,
        config={"num_rounds": 100},
        strategy=strategy,
    )
