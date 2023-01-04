import sys, os
sys.path.insert(0, os.path.abspath('..'))
import torch
import copy
import operator
import torch.nn.functional as F
from torch.autograd import Variable
from base_server import Server

import stc_utils

class STCServer(Server):
    def __init__(self, **kwargs):
        super(STCServer, self).__init__(**kwargs)
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.dW_compressed = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                                                for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(self.cfgs.device) 
                                for name, value in self.W.items()}

        self.A = {name: torch.zeros(value.shape).to(self.cfgs.device) 
                                for name, value in self.W.items()}
        self.n_params = sum([T.numel() for T in self.W.values()])
        
        # Compression hyperparameters
        self.hp_comp = stc_utils.get_hp_compression(self.cfgs.compression)

    def load_weights(self):
        for name, value in self.dW.items():
            self.model[name] += value.clone()

    def aggregate_weight_updates(self, clients, aggregation='mean'):
        if aggregation == 'mean':
            stc_utils.average(
                target=self.dW,
                sources=[client[0] for client in clients]
            )
        
        elif aggregation == 'weighted_mean':
            stc_utils.weighted_average(
                target=self.dW,
                sources=[client[0] for client in clients],
                weights=[client[1] for client in clients]
            )
        
        elif aggregation == 'majority':
            stc_utils.majority_vote(
                target=self.dW,
                sources=[client[0] for client in clients]
            )
        
        self.load_weights()