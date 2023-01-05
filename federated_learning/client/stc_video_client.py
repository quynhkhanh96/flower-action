from flwr.common.typing import FitRes, Parameters
import flwr 
from flwr.common import FitIns, FitRes, ParametersRes, EvaluateIns, EvaluateRes, Weights
from flwr.common import parameters_to_weights, weights_to_parameters

import numpy as np 
import torch 
import torch.optim as optim
from collections import OrderedDict

from fedavg_video_client import FedAvgVideoClient
from ops import stc_ops, stc_compress

class STCVideoClient(FedAvgVideoClient):

    def __init__(self, compression, **kwargs):
        super(STCVideoClient, self).__init__(**kwargs)

        self.W_old = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.model.items()}
        self.dW = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.model.items()}
        self.dW_compressed = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.model.items()}
        self.A = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.model.items()}

        self.n_params = sum([T.numel() for T in self.model.values()])

        # optimizer_object = getattr(optim, self.cfgs.optimizer)
        # optimizer_parameters = {k : v for k, v in self.cfgs.__dict__.items() 
        #                 if k in optimizer_object.__init__.__code__.co_varnames}

        # self.optimizer = optimizer_object(self.model.parameters(), **optimizer_parameters)

        # Compression hyperparameters
        self.hp_comp = stc_compress.get_hp_compression(compression)