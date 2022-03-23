from federated_learning import FedAvgVideoClient
from federated_learning.client.update.video_base import MMAction2LocalUpdate 
from evaluation.video_recognition import evaluate_video_recognizer
import flwr
import torch 
import argparse
from datasets import *
from mmcv import Config
from mmaction.models import build_model
import yaml 
from utils.parsing import Dict2Class

DEFAULT_SERVER_ADDRESS = "[::]:8080"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="FlowerAction")
    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--cid", type=int, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--cfg_path",
        default='configs/hmdb51_rgb_k400_pretrained.py',
        type=str,
        help="Configuration file path",
    )
    parser.add_argument(
        "--fed_cfg_path",
        type=str,
        help="Configuration file path",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        help="where checkpoints are saved, progress is logged, etc",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default='', # should be ${MMACTION}/data/hmdb51
        help="image, metadata directory",
    )
    parser.add_argument(
        "--fold", type=int, default=1, help="split id"
    )

    client_args = parser.parse_args()
    client_id = client_args.cid

    # Configurations 
    cfg = Config.fromfile(client_args.cfg_path)
    cfg.omnisource = False # dirty fix 
    cfg.work_dir = client_args.work_dir + f'/client_{client_id}'
    cfg.gpu_ids = range(1) #TODO

    with open(client_args.fed_cfg_path, 'r') as yamlfile:
        fed_cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    fed_cfgs = Dict2Class(fed_cfgs)

    # Logger
    flwr.common.logger.configure(f"client_{client_id}", 
                                    host=client_args.log_host)

    # # datasets
    # if fed_cfgs.dataset_name == 'hmdb51':
    #     train_dataset, test_dataset = get_hmdb51_client_dataset(client_id, 
    #                                     client_args.fold, client_args.data_dir)
    # else:
    #     raise ValueError(f'No data loaders implemented for {cfg.dataset_name} dataset.') 

    # datasets
    train_dataset, test_dataset = get_client_dataset(client_id, client_args.fold,
                                                        client_args.data_dir, cfg)

    # Model 
    model = build_model( 
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    # local trainer and evaluation function
    local_update = MMAction2LocalUpdate(train_dataset, cfg)
    eval_fn = evaluate_video_recognizer

    # Start client 
    fedavg_client = FedAvgVideoClient(client_id=client_id,
                        ds_train=train_dataset, ds_test=test_dataset,
                        model=model, local_update=local_update,
                        eval_fn=eval_fn, cfg=cfg
                    )
    flwr.client.start_client(client_args.server_address, fedavg_client)