import os 
import argparse
import flwr 
import torch
import functools
from datasets.frame_dataset import get_client_loaders
from evaluation.video_recognition import evaluate_topk_accuracy
import yaml 
from utils.parsing import Dict2Class
from utils import seed_torch

DEFAULT_SERVER_ADDRESS = "[::]:8080"

def fit_config(rnd, cfgs):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(cfgs.local_e),
        "batch_size": str(cfgs.train_bz)
    }
    return config

if __name__ == '__main__':
    """Start server and train for some rounds."""
    parser = argparse.ArgumentParser(description="FlowerAction")
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
        help="image, metadata directory",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        help="where checkpoints are saved, progress is logged, etc",
    )
    parser.add_argument(
        "--p_up",
        default=-1.,
        type=float,
        help="Upstream compression factor for STC and DGC",
    )
    parser.add_argument(
        "--fp_layers",
        default='',
        help="Layers that are not to quantize, this argument is a string of layer names separated by `,`",
    )
    server_args = parser.parse_args()
    os.makedirs(server_args.work_dir, exist_ok=True)

    # Configuration
    with open(server_args.cfg_path, 'r') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfgs = Dict2Class(cfgs)
    if server_args.p_up != -1:
        cfgs.p_up = server_args.p_up

    seed_torch(int(cfgs.seed))

    # eval_fn
    eval_fn = evaluate_topk_accuracy

    # test loader
    if hasattr(cfgs, 'base') and cfgs.base == 'mmaction2':
        from datasets.frame_dataset import get_client_mmaction_loaders
        _, test_loader = get_client_mmaction_loaders(
            0, server_args.data_dir, cfgs
        )
    else:
        from datasets.frame_dataset import get_client_loaders
        _, test_loader = get_client_loaders(
            0, server_args.data_dir, cfgs
        )

    # create strategy
    if cfgs.FL in ['FedAvg', 'FedBN']:
        from federated_learning.server.fedavg_video_server import FedAvgVideoStrategy
        strategy = FedAvgVideoStrategy(
            cfgs=cfgs,
            dl_test=test_loader,
            ckpt_dir=server_args.work_dir,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            fraction_fit=cfgs.frac,
            min_fit_clients=cfgs.min_sample_size,
            min_available_clients=cfgs.min_num_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=functools.partial(fit_config, cfgs=cfgs),
        )
    elif cfgs.FL in ['STC']:
        from federated_learning.server.stc_video_server import STCVideoStrategy
        strategy = STCVideoStrategy(
            cfgs=cfgs,
            dl_test=test_loader,
            ckpt_dir=server_args.work_dir,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            fraction_fit=cfgs.frac,
            min_fit_clients=cfgs.min_sample_size,
            min_available_clients=cfgs.min_num_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=functools.partial(fit_config, cfgs=cfgs),
        )
    elif cfgs.FL in ['QSGD']:
        from federated_learning.server.qsgd_video_server import QSGDVideoServer
        strategy = QSGDVideoServer(
            random=cfgs.random, n_bit=cfgs.n_bit, lower_bit=cfgs.lower_bit,
            q_down=cfgs.q_down, no_cuda=False, 
            fp_layers=server_args.fp_layers, cfgs=cfgs, 
            dl_test=test_loader, ckpt_dir=server_args.work_dir,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            fraction_fit=cfgs.frac,
            min_fit_clients=cfgs.min_sample_size,
            min_available_clients=cfgs.min_num_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=functools.partial(fit_config, cfgs=cfgs),            
        )
    else:
        raise ValueError(f'No implementation for {cfgs.FL}.')

    # Configure logger and start server
    flwr.common.logger.configure("server", host=server_args.log_host)
    flwr.server.start_server(
        server_args.server_address,
        config={"num_rounds": cfgs.epochs},
        strategy=strategy,
    )
