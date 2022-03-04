from federated_learning import FedAvgClient
from evaluation.classification import *
import yaml 
import flwr
import torch 
import argparse
from utils.parsing import Dict2Class
from datasets import *

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
        "--working_dir",
        type=str,
        help="Where the split is saved",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default='', # only GLD23k, HMDB51 dataset needs it 
        help="image directory",
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
    # TODO: generalization for all classification dataset (eg. MNIST)
    if cfgs.dataset == 'cifar10':
        train_loader, test_loader = get_cifar_client_loader(client_id, 
                                                            local_bz=cfgs.batch_size, 
                                                            test_bz=cfgs.batch_size, 
                                                            working_dir=client_args.working_dir)   
        num_classes = 10 
    elif cfgs.dataset == 'mnist':
        train_loader, test_loader = get_mnist_client_loader(client_id, 
                                                            local_bz=cfgs.batch_size, 
                                                            test_bz=cfgs.batch_size, 
                                                            working_dir=client_args.working_dir) 
        num_classes = 10 
    elif cfgs.dataset == 'gld23k':
        if client_args.data_dir == '':
            raise ValueError('`data_dir` (path to image directory) for gld23k is missing.')
        train_loader, test_loader, num_classes = get_landmark_client_loader(client_id,
                                                            local_bz=cfgs.batch_size, 
                                                            test_bz=cfgs.batch_size, 
                                                            data_dir=client_args.data_dir, 
                                                            working_dir=client_args.working_dir)   
    
    elif cfgs.dataset == 'hmdb51':
        if client_args.data_dir == '':
            raise ValueError('`data_dir` (path to video directory) for hmdb51 is missing.')
        train_loader, test_loader, num_classes = get_hmdb51_client_loader(client_id,
                fold=cfgs.fold, num_frames=cfgs.num_frames, clip_steps=cfgs.clip_steps,
                local_bz=cfgs.batch_size, test_bz=cfgs.batch_size,
                video_data_dir=client_args.video_data_dir, 
                working_dir=client_args.working_dir 
        )
    else:
        raise ValueError(f'No data loaders implemented for {cfgs.dataset} dataset.')    

    # Model
    if cfgs.model == 'CNN':
        # from models import CNN as Fed_Model
        from models.classification import CNN as Fed_Model
        net = Fed_Model(num_classes=num_classes)
    elif cfgs.model == 'MLP': 
        from models.classification import MLP as Fed_Model
        net = Fed_Model(num_classes=num_classes)
    elif cfgs.model == 'ResNet': 
        from models.classification import ResNet as Fed_Model
        net = Fed_Model(num_classes=num_classes)
    elif cfgs.model == 'MobileNetV3':
        from models.classification import MobileNetV3 as Fed_Model
        net = Fed_Model(num_classes=num_classes)
    elif cfgs.model == 'MoViNet':
        from models.video.movinets.models import MoViNet as Fed_Model
        from models.video.movinets.config import _C
        net = Fed_Model(_C.MODEL.MoViNetA0, causal=True, pretrained=True , num_classes=num_classes)
    else:
        raise ValueError(f'No models implemented for {cfgs.model} model.')

    # net = Fed_Model(num_classes=num_classes)
    net.to(cfgs.device)

    # local trainer and evaluate function
    if 'image' in cfgs.task:
        from federated_learning.client.update.base import BaseLocalUpdate
        local_update = BaseLocalUpdate(dl_train=train_loader,
                                        loss_func=torch.nn.CrossEntropyLoss(),
                                        args=cfgs)
        eval_fn = test_classifer
    elif 'video' in cfgs.task:
        from federated_learning.client.update.video_base import VideoLocalUpdate
        local_update = VideoLocalUpdate(dl_train=train_loader,
                                        loss_func=torch.nn.CrossEntropyLoss(),
                                        args=cfgs)
        eval_fn = test_video_classifer

    # Start client
    fedavg_client = FedAvgClient(client_id=client_id,
                            dl_train=train_loader,
                            dl_test=test_loader,
                            net=net, 
                            local_update=local_update,
                            eval_fn=eval_fn,
                            args=cfgs
                        )
    flwr.client.start_client(client_args.server_address, fedavg_client)
