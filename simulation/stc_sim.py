import sys, os
sys.path.insert(0, os.path.abspath('..'))
import copy
import numpy as np 
import argparse
import yaml
from utils.parsing import Dict2Class
from utils import seed_torch
from models.build import build_loss
from evaluation.video_recognition import evaluate_topk_accuracy

from stc_client import STCClient
from stc_server import STCServer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulation FedAvg")
    parser.add_argument(
        "--work_dir",
        type=str, required=True,
        help="Working directory, used for saving logs, checkpoints etc.",
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
        "--server_device",
        default='cuda:1',
        type=str,
        help="device for running server",
    )
    parser.add_argument(
        "--aggregation",
        default='mean',
        type=str,
        help="aggregation method, one of mean|weighted_mean|majority",
    )
    parser.add_argument(
        "--compression",
        default='stc_up',
        type=str,
        help="compression method, one of stc_up|signsgd|dgc_up",
    )
    parser.add_argument(
        "--p_up",
        default=0.04,
        type=float,
        help="Upstream compression factor for STC and DGC",
    )
    args = parser.parse_args()

    # configurations 
    with open(args.cfg_path, 'r') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfgs = Dict2Class(cfgs)

    # compression hyperparameters
    compression = [args.compression, {'p_up': args.p_up}]
    # fix randomness
    seed_torch(int(cfgs.seed))

    # model
    if hasattr(cfgs, 'base') and cfgs.base == 'mmaction2':
        from models.base import build_mmaction_model
        global_model = build_mmaction_model(cfgs, mode='train')
    else:
        from models.build import build_model
        global_model = build_model(cfgs, mode='train')

    # loss function
    criterion = build_loss(cfgs)

    # evaluation function
    eval_fn = evaluate_topk_accuracy

    # server initialization
    fl_server = STCServer(
        compression=compression,
        data_dir=args.data_dir, work_dir=args.work_dir,
        eval_fn=eval_fn, model=global_model,
        cfgs=cfgs, device=args.server_device
    )

    # client initialization
    fl_client = STCClient(
        compression=compression,
        data_dir=args.data_dir, work_dir=args.work_dir,
        model=copy.deepcopy(global_model), loss_fn=criterion,
        eval_fn=eval_fn, cfgs=cfgs
    )

    for rnd in range(int(cfgs.epochs)):

        # randomly sample clients
        selected_clients = fl_server.sample_clients()
        print('[INFO]Round {}: clients {} are selected'.format(
            rnd, ', '.join([str(client_id) for client_id in selected_clients])
        ))

        # clients perform local training and send back weight updates
        res = []
        for client_id in selected_clients:
            weight_updates, num_examples = fl_client.train(rnd, client_id,
                                            fl_server.model)
            res.append([weight_updates, num_examples])
        
        # server aggregates the updates to create new global model
        fl_server.aggregate_weight_updates(res, args.aggregation)
        fl_server.evaluate(rnd)