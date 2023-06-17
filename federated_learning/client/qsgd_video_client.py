import sys, os
sys.path.insert(0, os.path.abspath('..'))

import torch
from collections import OrderedDict

from flwr.common.typing import FitIns, FitRes, Parameters
from flwr.common import parameters_to_weights, weights_to_parameters
from fedavg_video_client import FedAvgVideoClient
from utils import qsgd

class QSGDClient(FedAvgVideoClient):
    
    def __init__(self, random, n_bit, lower_bit, q_down, **kwargs):
        super().__init__(**kwargs)

        self.quantizer = qsgd.QSGDQuantizer(
            random, n_bit, no_cuda
        )
        self.q_down = q_down
        self.lower_bit = lower_bit if lower_bit != -1 else n_bit

        self.coder = qsgd.QSGDCoder(2 ** n_bit)

        self.W = {name: value for name, value in self.model.named_parameters()}
        self.W_old = {name: torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}     

    def load_weights(self, params: Parameters):
        if self.q_down:
            state_dict = {}
            for i, (lname, lweight) in enumerate(self.W.items()):
                dec = self.coder.decode(msg, 
                        reduce(lambda x, y: x*y, lweight.shape))
                state_dict[lname] = torch.tensor(dec).view(lweight.shape)
        else:
            weights = parameters_to_weights(params)
            state_dict = OrderedDict(
                {k: torch.Tensor(v) 
                for k, v in zip(self.model.state_dict().keys(), weights)}
            )
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.cfgs.device)
        self.W = {name: value for name, value in self.model.named_parameters()}

    def compress_weight_update_up(self):
        params_prime = []
        s = self.quantizer.s
        for lname, lgrad in self.dW.items():
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