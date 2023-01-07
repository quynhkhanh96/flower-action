import sys, os
sys.path.insert(0, os.path.abspath('..'))
import torch
import copy
import operator
from collections import OrderedDict
import torch.nn.functional as F
from torch.autograd import Variable
from base_server import Server

import stc_utils

class STCServer(Server):
    def __init__(self, compression, **kwargs):
        super(STCServer, self).__init__(**kwargs)

        self.dW = {name: torch.zeros(value.shape).to(self.cfgs.device)
                    for name, value in self.model.named_parameters()}        
        # Compression hyperparameters
        self.hp_comp = stc_utils.get_hp_compression(compression)

    def load_weights(self):
        state_dict = self.model.state_dict()
        for name, value in self.dW.items():
            state_dict[name] += value.clone()
        self.model.load_state_dict(state_dict, strict=False)

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