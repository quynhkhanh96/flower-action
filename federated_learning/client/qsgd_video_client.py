import os 
import torch
from collections import OrderedDict
from functools import reduce 

from flwr.common.typing import FitIns, FitRes, Parameters
from flwr.common import (
    parameters_to_weights,
    ndarray_to_bytes, bytes_to_ndarray
)
from fedavg_video_client import FedAvgVideoClient
from ..utils import qsgd

class QSGDVideoClient(FedAvgVideoClient):
    
    def __init__(self, random, n_bit, lower_bit, q_down, no_cuda, fp_layers, **kwargs):
        super().__init__(**kwargs)

        self.quantizer = qsgd.QSGDQuantizer(
            random, n_bit, no_cuda
        )
        self.q_down = q_down
        self.lower_bit = lower_bit if lower_bit != -1 else n_bit
        self.fp_layers = [fp_layer for fp_layer in fp_layers.split(',')
                            if fp_layer != '']
        self.coder = qsgd.QSGDCoder(2 ** n_bit)

        self.W = {name: value for name, value in self.model.state_dict().items()}
        self.W_old = {name: torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}

        for lname, weight in self.W.items():
            try:
                _ = len(weight)
            except:
                self.fp_layers.append(lname)     

    def _keep_layer_full_precision(self, lname):
        for fp_layer in self.fp_layers:
            if fp_layer in lname:
                return True
        return False
    
    def load_weights(self, params: Parameters):
        if self.q_down:
            state_dict = {}
            for i, (lname, lweight) in enumerate(self.W.items()):
                if self._keep_layer_full_precision(lname):
                    state_dict[lname] = torch.Tensor(bytes_to_ndarray(params.tensors[i]))
                else:
                    dec = self.coder.decode(params.tensors[i], 
                            reduce(lambda x, y: x*y, lweight.shape))
                    state_dict[lname] = torch.Tensor(dec).view(lweight.shape)
        else:
            weights = parameters_to_weights(params)
            weights = self.postprocess_weights(weights)
            state_dict = OrderedDict(
                {k: torch.Tensor(v) 
                for k, v in zip(self.model.state_dict().keys(), weights)}
            )
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.cfgs.device)
        self.W = {name: value for name, value in self.model.state_dict().items()}

    def compress_weight_update_up(self):
        params_prime = []
        s = self.quantizer.s
        for lname, lgrad in self.dW.items():
            if self._keep_layer_full_precision(lname):
                params_prime.append(
                    ndarray_to_bytes(lgrad.cpu().numpy())
                )
                continue

            if torch.count_nonzero(lgrad) == 0: 
                params_prime.append(bytes(1 & 0xff))
                continue

            if 'conv' in lname and 'bn' not in lname:
                self.quantizer.s = s
            else:
                self.quantizer.s = 2 ** self.lower_bit
            signature = self.quantizer.quantize(lgrad)

            norm = signature[0].cpu().numpy()[0][0]
            signs = signature[1].view(-1).cpu().numpy()
            epsilon = signature[2].view(-1).cpu().numpy()

            params_prime.append(
                self.coder.encode(norm, signs, epsilon)
            )

        self.quantizer.s = s

        return params_prime             

    def fit(self, ins: FitIns) -> FitRes:
        # set new local weights
        self.load_weights(ins.parameters)

        # compute weight updates
        qsgd.copy(target=self.W_old, source=self.W)
        self.local.train(model=self.model, client_id=self.client_id)
        qsgd.subtract_(target=self.dW, minuend=self.W, subtrachend=self.W_old)

        num_examples_train = len(self.dl_train.dataset)
        # compress and encode gradients
        params_prime = self.compress_weight_update_up()

        return FitRes(
            parameters=Parameters(tensors=params_prime, tensor_type="numpy.ndarray"),
            num_examples=num_examples_train
        )