import flwr 
from flwr.common import parameters_to_weights
import numpy as np
import torch 
from models.build import build_model
from collections import OrderedDict

class FedAvgVideoStrategy(flwr.server.strategy.FedAvg):
    def __init__(self, cfgs, dl_test, ckpt_dir, device, **kwargs):
        self.cfgs = cfgs 
        self.dl_test = dl_test 
        self.ckpt_dir = ckpt_dir
        self.device = device 
        self.best_top1_acc = -1 
        super(FedAvgVideoStrategy, self).__init__(**kwargs) 

    @staticmethod
    def postprocess_weights(weights):
        for i, w in enumerate(weights):
            try:
                _ = len(w)
            except:
                weights[i] = np.array([0])

        return weights

    def evaluate(self, parameters):
        if self.eval_fn is None:
            return None

        weights = parameters_to_weights(parameters)
        weights = self.postprocess_weights(weights)

        state_dict = OrderedDict(
            {k: torch.Tensor(v) 
            for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        model = build_model(self.cfgs, mode='test')
        model.load_state_dict(state_dict)
        
        eval_res = self.eval_fn(model, self.dl_test, self.device)
        if eval_res is None:
            return None
        
        metrics = {'top1_accuracy': eval_res['top1'], 'top5_accuracy': eval_res['top5']}
        if eval_res['top1'] > self.best_top1_acc:
            self.best_top1_acc = eval_res['top1']
            torch.save({'state_dict': model.state_dict(),}, self.ckpt_dir + '/best.pth')

        return 0., metrics

        