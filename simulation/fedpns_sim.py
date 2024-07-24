import sys, os
sys.path.insert(0, os.path.abspath('..'))
import copy
import numpy as np 
import argparse
import yaml
import random
from utils.parsing import Dict2Class
from utils import seed_torch
from models.build import build_loss
from evaluation.video_recognition import evaluate_topk_accuracy

from base_client import Client
from fedpns_server import FedPNSServer
from fedpns_utils import probabilistic_selection

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
    fl_server = FedPNSServer(v=0.7, data_dir=args.data_dir, 
                        work_dir=args.work_dir, eval_fn=eval_fn, 
                        model=global_model, cfgs=cfgs, 
                        device=args.server_device)

    # client initialization
    fl_client = Client(data_dir=args.data_dir, work_dir=args.work_dir,
                        model=copy.deepcopy(global_model), loss_fn=criterion,
                        eval_fn=eval_fn, cfgs=cfgs)

    node_prob, test_count = {}, {}
    for i in range(cfgs.num_C):
        node_prob[i] = 1 / cfgs.num_C
        tupe = []
        for j in range(3):
            tupe.append(0)
            test_count[i] = tupe

    which_node, remove_who = {}, {}
    for rnd in range(1, int(cfgs.epochs)+1):
        loss_local = []
        w_locals, gradient, num_examples = {}, {}, {}
        if rnd == 1:
            idxs_users = fl_server.sample_clients()

        # log the selected clients
        msg = 'Round {}: clients {} are selected'.format(
            rnd, ', '.join([str(client_id) for client_id in idxs_users])
        )
        print(msg)
        with open(args.work_dir + f'/node_C{cfgs.num_C}_frac{cfgs.frac}_logs.txt', 'a') as f:
            f.write(msg + '\n')
        
        for i in range(len(idxs_users)):
            test_count[idxs_users[i]][0] += 1

        # Each client performs local training
        for idx in idxs_users:
            client_weight, data_size = fl_client.train(rnd, idx, 
                                            copy.deepcopy(global_model))
            w_locals[idx] = copy.deepcopy(client_weight)
            num_examples[idx] = data_size
        # Server aggregate the weights
        w_glob, idxs_before, idxs_left, labeled, test_count = fl_server.aggregate(
                                w_locals, idxs_users, num_examples, test_count)
        # Update new global weight
        global_model = fl_server.load_weights(w_glob)
        # Server evaluates new global weight on test data
        fl_server.evaluate(rnd)
        # Server perform probabilistic selection
        idxs_users, node_prob, test_count = probabilistic_selection(node_prob, 
                test_count, idxs_before, idxs_left, labeled, fl_server.num_selected)
        # Learning rate update??