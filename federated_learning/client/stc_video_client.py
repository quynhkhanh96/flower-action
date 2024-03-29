from flwr.common.typing import FitRes, Parameters
import flwr 
from flwr.common import FitIns, FitRes, Weights
from flwr.common import parameters_to_weights, weights_to_parameters
from flwr.common import ndarray_to_bytes

import numpy as np 
import torch 
import torch.optim as optim
from collections import OrderedDict

from fedavg_video_client import FedAvgVideoClient
import stc_ops
import stc_compress
import stc_encode

class STCVideoClient(FedAvgVideoClient):

    def __init__(self, **kwargs):
        super(STCVideoClient, self).__init__(**kwargs)

        self.W = {name : value.to(self.cfgs.device) for name, value in self.model.named_parameters()}
        self.W_old = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
        self.dW = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
        self.dW_compressed = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
        self.A = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}

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
            {k: torch.Tensor(v) 
            for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.cfgs.device)
        self.W = {name: value for name, value in self.model.named_parameters()}

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
        
        # dW_comp = [val.cpu().numpy() for _, val in self.dW_compressed.items()]
        # weights_prime = self.postprocess_weights(dW_comp)
        # params_prime = weights_to_parameters(weights_prime)
        # num_examples_train = len(self.dl_train.dataset)

        # return FitRes(
        #     parameters=params_prime,
        #     num_examples=num_examples_train        
        # )

        msgs, signs, mus = [], [], []
        for _, dW_layer in self.dW_compressed.items():
            msg_, sign_, mu_ = stc_encode.golomb_position_encode(dW_layer, 
                                self.cfgs.p_up)
            msgs.append(msg_.encode('ascii'))
            signs.append(sign_.encode('ascii'))
            mus.append(mu_.cpu().numpy())
        
        mus = ndarray_to_bytes(np.array(mus))
        num_examples_train = len(self.dl_train.dataset)
        
        params_prime = msgs + signs + [mus]
        return FitRes(
            parameters=Parameters(tensors=params_prime, tensor_type="numpy.ndarray"),
            num_examples=num_examples_train
        )

