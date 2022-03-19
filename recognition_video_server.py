import argparse
import flwr 
import torch
import functools
from evaluation.video_recognition import evaluate_video_recognizer
from federated_learning.server.fedavg_video_server import FedAvgVideoStrategy
from datasets import * 
from mmcv import Config

DEFAULT_SERVER_ADDRESS = "[::]:8080"

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
        default='', 
        help="image, metadata directory",
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        help="where checkpoints are saved, progress is logged, etc",
    )
    parser.add_argument(
        "--fold", type=int, default=1, help="split id"
    )
    server_args = parser.parse_args()

    # Configuration
    cfg = Config.fromfile(server_args.cfg_path)
    cfg.omnisource = False # dirty fix 
    cfg.work_dir = server_args.work_dir 

    # test dataset, evaluation function
    if cfg.dataset_name == 'hmdb51':
        _, test_dataset = get_hmdb51_client_dataset(0, server_args.fold, 
                                                    server_args.data_dir)
    else:
        raise ValueError(f'No data loaders implemented for {cfg.dataset_name} dataset.') 
    eval_fn = evaluate_video_recognizer

    # create strategy
    strategy = FedAvgVideoStrategy(
        cfg=cfg,
        test_dataset=test_dataset,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        fraction_fit=cfg.frac,
        min_fit_clients=cfg.min_sample_size,
        min_available_clients=cfg.min_num_clients,
        eval_fn=eval_fn,
        on_fit_config_fn=functools.partial(fit_config, cfgs=cfg),
    )

    # Configure logger and start server
    flwr.common.logger.configure("server", host=server_args.log_host)
    flwr.server.start_server(
        server_args.server_address,
        config={"num_rounds": cfg.rounds},
        strategy=strategy,
    )
