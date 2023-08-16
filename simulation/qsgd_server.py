import torch
from base_server import Server
import qsgd_utils
import numpy as np

class QSGDServer(Server):
    def __init__(self, random, n_bit, lower_bit, no_cuda, fp_layers, **kwargs):
        super(QSGDServer, self).__init__(**kwargs)

        self.quantizer = qsgd_utils.QSGDQuantizer(
            random, n_bit, no_cuda
        )
        self.lower_bit = lower_bit if lower_bit != -1 else n_bit
        self.fp_layers = fp_layers.split(',')
        self.dW = {name: torch.zeros(value.shape).to(self.cfgs.device) 
                for name, value in self.model.state_dict().items()}

    def load_weights(self):
        state_dict = self.model.state_dict()
        with torch.no_grad():
            for name, value in self.dW.items():
                state_dict[name] = state_dict[name] + value.clone().to(self.device)
        self.model.load_state_dict(state_dict, strict=False)

    def compress_weight_down(self):
        res = {}
        for lname, lweight in self.model.named_parameters():
            if lname in self.fp_layers:
                res[lname] = lweight
                continue
            res[lname] = self.quantizer.quantize(lweight)
        
        return res

    def decode_fit_results(self, clients):
        grads = []
        s = self.quantizer.s
        for res, num_samples in clients:
            grad_ = {}
            for lname, signature in res.items():
                if 'conv' in lname and 'bn' not in lname:
                    self.quantizer.s = s
                else:
                    self.quantizer.s = 2 ** self.lower_bit
                grad_[lname] = self.quantizer.dequantize(signature)
            grads.append([grad_, num_samples])
        self.quantizer.s = s

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
                
        self.load_weights()

class TopKQSGDServer(QSGDServer):

    def decode_fit_results(self, clients):
        grads = []
        s = self.quantizer.s
        for res, num_samples in clients:
            grad_ = {}

            for lname in res:

                if 'conv' in lname and 'bn' not in lname:
                    s_top, s_rest = res[lname]

                    self.quantizer.s = s
                    grad_top = self.quantizer.dequantize(s_top)

                    self.quantizer.s = 2 ** self.lower_bit
                    grad_rest = self.quantizer.dequantize(s_rest)

                    with torch.no_grad():
                        grad_[lname] = grad_top + grad_rest

                else:
                    self.quantizer.s = 2 ** self.lower_bit
                    grad_[lname] = self.quantizer.dequantize(res[lname])

            grads.append([grad_, num_samples])

        self.quantizer.s = s

        return grads  

