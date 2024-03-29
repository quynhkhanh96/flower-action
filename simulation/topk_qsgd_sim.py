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

from qsgd_client import TopKQSGDClient
from qsgd_server import TopKQSGDServer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulation QSGD FedAvg")
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
        "--random",
        action="store_false",
        help="whether the quantization is stochastic",
    )
    parser.add_argument(
        "--k",
        default=0.05,
        type=float,
        help="Top k largest gradients in terms of magnitude",
    )
    parser.add_argument(
        "--n_bit",
        default=8,
        type=int,
        help="Number of bits for quantization",
    )
    parser.add_argument(
        "--lower_bit",
        default=-1,
        type=int,
        help="Used for differential quantization, -1 means same level of quant. for all layer",
    )
    parser.add_argument(
        "--q_down",
        action="store_true",
        help="whether compression is applied downlink",
    )
    parser.add_argument(
        "--fp_layers",
        default='',
        help="Layers that are not to quantize, this argument is a string of layer names separated by `,`",
    )
    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    # configurations 
    with open(args.cfg_path, 'r') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfgs = Dict2Class(cfgs)

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
    fl_server = TopKQSGDServer(
        random=args.random, n_bit=args.n_bit, 
        lower_bit=args.lower_bit, no_cuda=False, 
        fp_layers=args.fp_layers,
        data_dir=args.data_dir, work_dir=args.work_dir,
        eval_fn=eval_fn, model=global_model,
        cfgs=cfgs, device=args.server_device
    )

    # client initialization
    fl_client = TopKQSGDClient(k=args.k,
        random=args.random, n_bit=args.n_bit, 
        lower_bit=args.lower_bit, no_cuda=False,
        q_down=args.q_down, fp_layers=args.fp_layers,
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
        if args.q_down:
            cur_global = fl_server.compress_weight_down()
        else:
            cur_global = fl_server.model
        for client_id in selected_clients:
            client_updates, num_examples = fl_client.train(rnd, client_id,
                                            cur_global)
            res.append([client_updates, num_examples])
        
        # server aggregates the updates to create new global model
        fl_server.aggregate_weight_updates(res, args.aggregation)
        fl_server.evaluate(rnd)