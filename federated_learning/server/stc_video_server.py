import sys
import os 
sys.path.insert(0, os.path.abspath('..'))
import flwr 
from flwr.common import parameters_to_weights, weights_to_parameters
import numpy as np
import torch
from collections import OrderedDict
import time 
from fedavg_video_server import FedAvgVideoStrategy
from utils import stc_ops, stc_compress

class STCVideoStrategy(FedAvgVideoStrategy):
    def __init__(self, **kwargs):
        super(STCVideoStrategy, self).__init__(**kwargs) 

        if hasattr(self.cfgs, 'base') and self.cfgs.base == 'mmaction2':
            from models.base import build_mmaction_model
            self.model = build_mmaction_model(self.cfgs, mode='test')

        else:
            from models.build import build_model
            self.model = build_model(self.cfgs, mode='test')
        self.dW = {name: torch.zeros(value.shape).to(self.device)
                    for name, value in self.model.named_parameters()}        
        # Compression hyperparameters
        compression = [self.cfgs.compression, {'p_up': self.cfgs.p_up}]
        self.hp_comp = stc_compress.get_hp_compression(compression)
        self.aggregation = self.cfgs.aggregation

    def aggregate_fit(self, rnd, results, failures):
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        grads_results = []
        for _, fit_res in results:
            grad_updates = parameters_to_weights(fit_res.paramaters)
            grads_results.append((grad_updates, fit_res.num_examples))
        if self.aggregation == 'mean':
            stc_ops.average(
                target=self.dW,
                source=[grads for grads, _ in grads_results]
            )
        elif self.aggregation == 'weighted_mean':
            stc_ops.weighted_average(
                target=self.dW,
                sources=[grads for grads, _ in grads_results],
                weights=torch.stack([torch.Tensor(num_examples).to(self.device) for _, num_examples in grads_results])
            )
        elif self.aggregation == 'majority':
            stc_ops.majority_vote(
                target=self.dW,
                sources=[grads for grads, _ in grads_results],
                lr=self.cfgs.lr
            )

        state_dict = self.model.state_dict()
        for name, value in self.dW.items():
            state_dict[name] += value.clone().to(self.device)
        weights = [weight for _, weight in state_dict.items()]

        return weights_to_parameters(weights), {}

    def evaluate(self, parameters):
        if self.eval_fn is None:
            return None

        weights = parameters_to_weights(parameters)
        weights = self.postprocess_weights(weights)

        state_dict = OrderedDict(
            {k: torch.Tensor(v) 
            for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        self.model.load_state_dict(state_dict, strict=False)
        
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