import copy
import numpy as np 
import argparse
import yaml
from utils.parsing import Dict2Class
from utils import seed_torch
from models.build import build_model, build_loss
from evaluation.video_recognition import evaluate_topk_accuracy

from .fedbn_client import FedBnClient
from .base_server import Server

def get_zero_weights(model):
    weights = []
    for _, val in model.state_dict().items():
        try:
            weights.append(np.full_like(val, 0.))
        except:
            weights.append(np.full_like(val.cpu().numpy(), 0.))
    return weights

def collect_weights(weights, client_weight, num_examples):
    for i in range(len(weights)):
        weights[i] = np.add(weights[i], client_weight[i] * num_examples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulation FedBN")
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

    # configurations 
    with open(args.cfg_path, 'r') as yamlfile:
        cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfgs = Dict2Class(cfgs)

    # fix randomness
    seed_torch(int(cfgs.seed))

    # model
    global_model = build_model(cfgs, mode='train')

    # loss function
    criterion = build_loss(cfgs)

    # evaluation function
    eval_fn = evaluate_topk_accuracy

    # server initialization
    fl_server = Server(data_dir=args.data_dir, work_dir=args.work_dir,
                        eval_fn=eval_fn, model=global_model, 
                        cfgs=cfgs, device=args.server_device)

    # client initialization
    fl_client = FedBnClient(data_dir=args.data_dir, work_dir=args.work_dir,
                        model=copy.deepcopy(global_model), loss_fn=criterion,
                        eval_fn=eval_fn, cfgs=cfgs)

    for rnd in range(int(cfgs.epochs)):
        weights = get_zero_weights(global_model)
        
        # randomly sample clients
        selected_clients = fl_server.sample_clients()
        print('[INFO]Round {}: clients {} are selected'.format(
            rnd, ', '.join([str(client_id) for client_id in selected_clients])
        ))

        # local training and collect trained weights from clients
        num_examples_total = 0
        for client_id in selected_clients:
            client_weight, num_examples = fl_client.train(rnd, client_id, 
                                                    copy.deepcopy(global_model))
            collect_weights(weights, client_weight, num_examples)
            num_examples_total += num_examples
        
        # aggregate weights
        weights = [weight / num_examples_total for weight in weights]
        global_model = fl_server.load_weights(weights)

        # evaluate new model on server's test data
        fl_server.evaluate(rnd)