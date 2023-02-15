import sys
import os 
sys.path.insert(0, os.path.abspath('..'))
from flwr.common.typing import FitRes, Parameters
import flwr 
from flwr.common import FitIns, FitRes, ParametersRes, EvaluateIns, EvaluateRes, Weights
from flwr.common import parameters_to_weights, weights_to_parameters

import numpy as np 
import torch 
import torch.optim as optim
from collections import OrderedDict

from fedavg_video_client import FedAvgVideoClient
from utils import stc_ops, stc_compress

class STCVideoClient(FedAvgVideoClient):

    def __init__(self, **kwargs):
        super(STCVideoClient, self).__init__(**kwargs)

        self.W = {name : value for name, value in self.model.named_parameters()}
        self.W_old = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
        self.dW = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
        self.dW_compressed = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
        self.A = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}

        optimizer_object = getattr(optim, self.cfgs.optimizer)
        optimizer_parameters = {k : v for k, v in self.cfgs.__dict__.items() 
                        if k in optimizer_object.__init__.__code__.co_varnames}

        self.optimizer = optimizer_object(self.model.parameters(), **optimizer_parameters)

        # Compression hyperparameters
        compression = [self.cfgs.compression, {'p_up': self.cfgs.p_up}]
        self.hp_comp = stc_compress.get_hp_compression(compression)

    def compress_weight_update_up(self, compression=None, accumulate=False):
        if accumulate and compression[0] != "none":
            # compression with error accumulation
            stc_ops.add(target=self.A, source=self.dW)
            stc_compress.compress(
                target=self.dW_compressed, source=self.A,
                compress_fun=stc_compress.compression_function(*compression)
            )
            stc_ops.subtract(target=self.A, source=self.dW_compressed)
        else:
            # compression without error accumulation
            stc_compress.compress(
                target=self.dW_compressed, source=self.dW, 
                compress_fun=stc_compress.compression_function(*compression)
            )

    def fit(self, ins: FitIns) -> FitRes:
        # set local model weights with that of the new global model
        weights: Weights = parameters_to_weights(ins.parameters)
        weights = self.postprocess_weights(weights)
        
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        self.model.load_state_dict(state_dict)

        # compute weight updates
        ## W_old = W
        stc_ops.copy(target=self.W_old, source=self.W)
        ## train model locally 
        self.local.train(model=self.model, client_id=self.client_id)
        ## dW = W - W_old
        stc_ops.subtract_(target=self.dW, minuend=self.W, subtrachend=self.W_old)
        # compress weight updates up
        self.compress_weight_update_up(compression=self.hp_comp['compression_up'],
                                accumulate=self.hp_comp['accumulation_up'])
        
        weights_prime = self.postprocess_weights(self.dW_compressed)
        params_prime = weights_to_parameters(weights_prime)
        num_examples_train = len(self.dl_train.dataset)

        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train        
        )