# from federated_learning import FedAvgVideoClient
from federated_learning.client.fedavg_video_client import ThresholdedFedAvgVideoClient
from federated_learning.client.update.video_base import VideoLocalUpdate 
from evaluation.video_recognition import evaluate_topk_accuracy
import flwr
import argparse
# from datasets.video_dataset import get_client_loaders
from datasets.frame_dataset import get_client_local_loaders
from models.build import build_model, build_loss, build_optimizer
import yaml 
from utils.parsing import Dict2Class
# import wandb 
import os 
DEFAULT_SERVER_ADDRESS = "[::]:8080"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="FlowerAction Client")
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
        "--work_dir",
        type=str, required=True,
        help="Working directory, used for saving logs, checkpoints etc.",
    )
    parser.add_argument(
        "--cfg_path",
        default='configs/afosr_movinetA0.yaml',
        type=str,
        help="Configuration file path",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="image, metadata directory",
    )
    client_args = parser.parse_args()
    client_id = client_args.cid

    # configurations 
    with open(client_args.cfg_path, 'r') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfgs = Dict2Class(cfgs)

    # datasets
    train_loader, val_loader = get_client_local_loaders(client_id, 
                                        client_args.data_dir,
                                        client_args.work_dir,
                                        cfgs)

    # model
    model = build_model(cfgs, mode='train')

    # loss 
    criterion = build_loss(cfgs)

    # local trainer
    local_update = VideoLocalUpdate(train_loader=train_loader,
                                    loss_fn=criterion, cfgs=cfgs)

    # evaluate function
    eval_fn = evaluate_topk_accuracy

    # start client
    if cfgs.FL == 'FedAvg':
        fl_client = ThresholdedFedAvgVideoClient(
                work_dir=client_args.work_dir,
                client_id=client_id,
                dl_train=train_loader, dl_test=val_loader,
                model=model, loss_fn=criterion, 
                local_update=local_update, 
                eval_fn=eval_fn, cfgs=cfgs
        )
    else:
        raise ValueError(f'No implementation for {cfgs.FL}.')

    flwr.client.start_client(client_args.server_address, fl_client)
