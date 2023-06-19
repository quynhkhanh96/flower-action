# from federated_learning import FedAvgVideoClient
from federated_learning.client.fedavg_video_client import FedAvgVideoClient
from federated_learning.client.fedbn_video_client import FedBNVideoClient
from federated_learning.client.stc_video_client import STCVideoClient
from federated_learning.client.qsgd_video_client imprt QSGDVideoClient
import flwr
import argparse
from models.build import build_loss
from evaluation.video_recognition import evaluate_topk_accuracy
import yaml 
from utils.parsing import Dict2Class
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
        "--log_host",
        type=str,
        help="Logserver address (no default)",
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
    parser.add_argument(
        "--p_up",
        default=-1.,
        type=float,
        help="Upstream compression factor for STC and DGC",
    )
    client_args = parser.parse_args()
    client_id = client_args.cid

    # configurations 
    with open(client_args.cfg_path, 'r') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfgs = Dict2Class(cfgs)
    if client_args.p_up != -1:
        cfgs.p_up = client_args.p_up

    # loss 
    criterion = build_loss(cfgs)
    # evaluate function
    eval_fn = evaluate_topk_accuracy

    if hasattr(cfgs, 'base') and cfgs.base == 'mmaction2':
        # datasets
        from datasets.frame_dataset import get_client_mmaction_loaders
        train_loader, test_loader = get_client_mmaction_loaders(
            client_id, client_args.data_dir, cfgs
        )

        # model
        from models.base import build_mmaction_model
        model = build_mmaction_model(cfgs, mode='train')

        # local trainer
        from federated_learning.client.update.video_base import MMActionLocalUpdate 
        local_update = MMActionLocalUpdate(train_loader=train_loader,
                                        loss_fn=criterion, cfgs=cfgs)

    else:
        # datasets
        from datasets.frame_dataset import get_client_loaders
        train_loader, test_loader = get_client_loaders(
            client_id, client_args.data_dir, cfgs
        )

        # model
        from models.build import build_model
        model = build_model(cfgs, mode='train')

        # local trainer
        from federated_learning.client.update.video_base import VideoLocalUpdate
        local_update = VideoLocalUpdate(train_loader=train_loader,
                                        loss_fn=criterion, cfgs=cfgs)
    
    # start client
    if cfgs.FL == 'FedAvg':
        fl_client = FedAvgVideoClient(client_id=client_id,
                dl_train=train_loader, dl_test=test_loader,
                model=model, loss_fn=criterion, 
                local_update=local_update, 
                eval_fn=eval_fn, cfgs=cfgs
        )
    elif cfgs.FL == 'FedBN':
        fl_client = FedBNVideoClient(client_id=client_id,
                dl_train=train_loader, dl_test=test_loader,
                model=model, loss_fn=criterion, 
                local_update=local_update, 
                eval_fn=eval_fn, cfgs=cfgs
        )
    elif cfgs.FL == 'STC':
        fl_client = STCVideoClient(client_id=client_id,
                dl_train=train_loader, dl_test=test_loader,
                model=model, loss_fn=criterion, 
                local_update=local_update, 
                eval_fn=eval_fn, cfgs=cfgs
        )
    elif cfgs.FL == 'QSGD':
        fl_client = QSGDVideoClient(
            random=cfgs.random, n_bit=cfgs.n_bit, lower_bit=cfgs.lower_bit,
            q_down=cfgs.q_down, no_cuda=cfgs.no_cuda, client_id=client_id,
            dl_train=train_loader, dl_test=test_loader,
            model=model, loss_fn=criterion, 
            local_update=local_update, 
            eval_fn=eval_fn, cfgs=cfgs            
        )
    else:
        raise ValueError(f'No implementation for {cfgs.FL}.')

    flwr.client.start_client(client_args.server_address, fl_client)
