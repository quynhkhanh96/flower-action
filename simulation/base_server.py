import sys, os
sys.path.insert(0, os.path.abspath('..'))
from collections import OrderedDict
import numpy as np 
import torch
from datasets.frame_dataset import get_client_loaders

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
        print('Round {}: server accuracy top1 {:.3f}, top5 {:.3f}'.format(
            rnd, topk_accuracy['top1'], topk_accuracy['top5']
        ))
        with open(self.work_dir + '/server_accs.txt', 'a') as f:
            f.write('{} {:.3f}\n'.format(rnd, topk_accuracy['top1']))