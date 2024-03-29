import torch
from base_server import Server
import qsgd_utils
import numpy as np

class QSparseSGDServer(Server):
    def __init__(self, random, n_bit, no_cuda, **kwargs):
        super().__init__(**kwargs)

        self.quantizer = qsgd_utils.QSGDQuantizer(
            random, n_bit, no_cuda
        )
        self.dW = {name: torch.zeros(value.shape).to(self.cfgs.device) 
                for name, value in self.model.named_parameters()}

    def decode_fit_results(self, clients):
        grads = []
        for res, num_samples in clients:
            grad_ = {}
            for lname, signature in res.items():
                grad_[lname] = self.quantizer.dequantize(signature)
            grads.append([grad_, num_samples])

        return grads        

    def aggregate_weight_updates(self, clients, aggregation='mean'):
        grads = self.decode_fit_results(clients)
        
        if aggregation == 'mean':
            qsgd_utils.average(
                target=self.dW,
                sources=[grad_[0] for grad_ in grads]
            )
        
        elif aggregation == 'weighted_mean':
            qsgd_utils.weighted_average(
                target=self.dW,
                sources=[grad_[0] for grad_ in grads],
                weights=torch.stack([torch.Tensor(grad_[1]).to(self.device) for grad_ in grads])
            )

        state_dict = self.model.state_dict()
        with torch.no_grad():
            for name, value in self.dW.items():
                state_dict[name] = state_dict[name] - value.clone().to(self.device)
        self.model.load_state_dict(state_dict, strict=False)