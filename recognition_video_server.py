import argparse
import flwr 
import torch
import functools
from evaluation.video_recognition import evaluate_video_recognizer
from federated_learning.server.fedavg_video_server import FedAvgVideoStrategy
from datasets import * 
from mmcv import Config
import yaml 
from utils.parsing import Dict2Class

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
        "--fed_cfg_path",
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
        "--work_dir",
        type=str,
        help="where checkpoints are saved, progress is logged, etc",
    )
    parser.add_argument(
        "--fold", type=int, default=1, help="split id"
    )
    server_args = parser.parse_args()
    os.makedirs(server_args.work_dir, exist_ok=True)

    # Configuration
    cfg = Config.fromfile(server_args.cfg_path)
    cfg.omnisource = False # dirty fix 
    # cfg.work_dir = server_args.work_dir 
    with open(server_args.fed_cfg_path, 'r') as yamlfile:
        fed_cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    fed_cfgs = Dict2Class(fed_cfgs)

    # test dataset, evaluation function
    # if fed_cfgs.dataset_name == 'hmdb51':
    #     _, test_dataset = get_hmdb51_client_dataset(0, server_args.fold, 
    #                                                 server_args.data_dir)
    # else:
    #     raise ValueError(f'No data loaders implemented for {cfg.dataset_name} dataset.') 

    _, test_dataset = get_client_dataset(0, server_args.fold, server_args.data_dir, cfg)

    eval_fn = evaluate_video_recognizer

    # create strategy
    strategy = FedAvgVideoStrategy(
        cfg=cfg,
        test_dataset=test_dataset,
        ckpt_dir=server_args.work_dir,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        fraction_fit=fed_cfgs.frac,
        min_fit_clients=fed_cfgs.min_sample_size,
        min_available_clients=fed_cfgs.min_num_clients,
        eval_fn=eval_fn,
        on_fit_config_fn=functools.partial(fit_config, cfgs=fed_cfgs),
    )

    # Configure logger and start server
    flwr.common.logger.configure("server", host=server_args.log_host)
    flwr.server.start_server(
        server_args.server_address,
        config={"num_rounds": fed_cfgs.epochs},
        strategy=strategy,
    )
