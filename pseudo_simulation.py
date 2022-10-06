import os 
import copy
from collections import OrderedDict
import numpy as np 
import argparse
import yaml
import torch
from utils.parsing import Dict2Class
from utils import seed_torch
from models.build import build_model, build_loss
from datasets.frame_dataset import get_client_loaders, get_client_local_loaders
from federated_learning.client.update.video_base import VideoLocalUpdate
from evaluation.video_recognition import evaluate_topk_accuracy

def get_zero_weights(model):
    weights = []
    for _, val in model.state_dict().items():
        weights.append(np.full_like(val, 0.))

    return weights

def collect_weights(weights, client_weight, num_examples):
    for i in range(len(weights)):
        weights[i] = np.add(weights[i], client_weight[i] * num_examples)

class Client:
    def __init__(self, data_dir, work_dir, model,
                        loss_fn, eval_fn, cfgs):
        self.data_dir = data_dir 
        self.work_dir = work_dir
        self.model = model
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.cfgs = cfgs 

    def get_data_loaders(self, client_id):
        train_loader, val_loader = get_client_local_loaders(client_id,
            self.data_dir, self.work_dir, self.cfgs
        )
        return train_loader, val_loader
    
    @staticmethod
    def postprocess_weights(weights):
        for i, w in enumerate(weights):
            try:
                _ = len(w)
            except:
                weights[i] = np.array([0])

        return weights
    
    def load_weights(self, weights):
        weights = self.postprocess_weights(weights)
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        self.model.load_state_dict(state_dict)

    def get_weights(self):
        weights = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return self.postprocess_weights(weights)

    def train(self, rnd, client_id, weights):
        # load the new global weights
        self.load_weights(weights)

        train_loader, val_loader = self.get_data_loaders(client_id)
        local_trainer = VideoLocalUpdate(train_loader=train_loader,
                                        loss_fn=self.loss_fn, 
                                        cfgs=self.cfgs)
        # train loop
        local_trainer.train(self.model, client_id)
        # validation loop
        topk_accuracy = self.eval_fn(self.model, val_loader, self.cfgs.device)
        with open(self.work_dir + f'/client_{client_id}_accs.txt', 'a') as f:
            f.write('{} {:.3f}\n'.format(rnd, topk_accuracy['top1']))

        return self.get_weights(), len(train_loader.dataset)

class Server:
    def __init__(self, data_dir, work_dir, model, eval_fn, cfgs, device):
        self.data_dir = data_dir
        self.work_dir = work_dir
        self.model = model 
        self.eval_fn = eval_fn
        self.cfgs = cfgs 
        self.device = device
        self.test_loader = self.get_test_loader()

    def get_test_loader(self):
        _, test_loader = get_client_loaders(0, self.data_dir, self.cfgs)
        return test_loader

    def sample_clients(self):
        num_clients = int(self.cfgs.num_C)
        num_selected = max(int(num_clients * self.cfgs.frac), 
                            int(self.cfgs.min_num_clients))
        selected_client_ids = np.random.choice(range(num_clients), 
                                num_selected, replace=False)
        return selected_client_ids
    
    def load_weights(self, weights):
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        self.model.load_state_dict(state_dict)
        return self.model

    def evaluate(self, rnd):
        topk_accuracy = self.eval_fn(self.model, self.test_loader, self.device)
        print('\t Round {}: server accuracy top1 {:.3f}, top5 {:.3f}'.format(
            rnd, topk_accuracy['top1'], topk_accuracy['top5']
        ))
        with open(self.work_dir + '/server_accs.txt', 'a') as f:
            f.write('{} {:.3f}\n'.format(rnd, topk_accuracy['top1']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulation FedAvg")
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
    fl_client = Client(data_dir=args.data_dir, work_dir=args.work_dir,
                        model=global_model.copy(), loss_fn=criterion,
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