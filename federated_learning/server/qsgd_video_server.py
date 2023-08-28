import sys, os
sys.path.insert(0, os.path.abspath('..'))

import torch
from functools import reduce
from collections import OrderedDict

from flwr.common.typing import FitIns, FitRes, Parameters
from flwr.common import parameters_to_weights, weights_to_parameters
from fedavg_video_server import FedAvgVideoStrategy
from utils import qsgd

class QSGDServer(FedAvgVideoStrategy):
    def __init__(self, random, n_bit, lower_bit, q_down, no_cuda, **kwargs):
        super().__init__(**kwargs)

        self.quantizer = qsgd.QSGDQuantizer(
            random, n_bit, no_cuda
        )
        self.q_down = q_down
        self.lower_bit = lower_bit if lower_bit != -1 else n_bit

        self.coder = qsgd.QSGDCoder(2 ** n_bit)

        if hasattr(self.cfgs, 'base') and self.cfgs.base == 'mmaction2':
            from models.base import build_mmaction_model
            self.model = build_mmaction_model(self.cfgs, mode='train')

        else:
            from models.build import build_model
            self.model = build_model(self.cfgs, mode='train')
        self.dW = {name: torch.zeros(value.shape).to(self.device)
                    for name, value in self.model.state_dict().items()} 
        
        self.best_top1_acc = -1      

    def compress_weight_down(self):
        params_prime = []
        for lname, lweight in self.model.state_dict().items():
            signature = self.quantizer.quantize(lweight)

            norm = signature[0].cpu().numpy()[0][0]
            signs = signature[1].view(-1).cpu().numpy()
            epsilon = signature[2].view(-1).cpu().numpy()

            params_prime.append(
                self.coder.encode(norm, signs, epsilon)
            )
        
        return Parameters(tensors=params_prime, tensor_type="numpy.ndarray")
          
    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}            

        if not self.accept_failures and failures:
            return None, {}
        
        # encode gradients
        grads_results = []
        s = self.coder.s
        for _, fit_res in results:
            grads = {}
            grad_prime = fit_res.parameters.tensors
            for i, (lname, lgrad) in enumerate(self.dW.items()):
                if 'conv' in lname and 'bn' not in lname:
                    self.coder.s = s
                else:
                    self.coder.s = 2 ** self.lower_bit
                dec = self.coder.decode(grad_prime[i], 
                        reduce(lambda x, y: x*y, lgrad.shape))
                grads[lname] = torch.tensor(dec).view(lgrad.shape).to(self.device)
        grads_results.append((grads, fit_res.num_examples))
        self.coder.s = s

        # aggregate gradients
        if self.aggregation == 'mean':
            qsgd.average(
                target=self.dW,
                sources=[grads for grads, _ in grads_results]
            )
        elif self.aggregation == 'weighted_mean':
            qsgd.weighted_average(
                target=self.dW,
                sources=[grads for grads, _ in grads_results],
                weights=torch.stack([torch.Tensor(num_examples).to(self.device) for _, num_examples in grads_results])
            )
        # create new global weights
        state_dict = self.model.state_dict()
        with torch.no_grad():
            for name, value in self.dW.items():
                state_dict[name] += value.clone().to(self.device)
        self.model.load_state_dict(state_dict, strict=False)

        # compress and encode global model to send downstream
        self.model.eval()
        return self.compress_weight_down(), {}

    def evaluate(self, parameters):
        if self.eval_fn is None:
            return None
        
        eval_res = self.eval_fn(self.model, self.dl_test, self.device)
        if eval_res is None:
            return None
        
        metrics = {'top1_accuracy': eval_res['top1'], 
                    'top5_accuracy': eval_res['top5']}
        
        with open(self.ckpt_dir + '/server_accs.txt', 'a') as f:
            f.write('{:.3f} {:.3f}\n'.format(metrics['top1_accuracy'],
                                            metrics['top5_accuracy']))
        
        if eval_res['top1'] > self.best_top1_acc:
            self.best_top1_acc = eval_res['top1']
            torch.save({'state_dict': self.model.state_dict()}, 
                        self.ckpt_dir + '/best.pth')

        return 0., metrics
