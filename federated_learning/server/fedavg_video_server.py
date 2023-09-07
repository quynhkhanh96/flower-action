import flwr 
from flwr.common import parameters_to_weights, weights_to_parameters
from flwr.server.strategy.aggregate import aggregate
import numpy as np
import torch
from collections import OrderedDict
import time 
# import wandb

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
                weights[i] = np.array([w.item()])

        return weights

    def evaluate(self, parameters):
        if self.eval_fn is None:
            return None

        weights = parameters_to_weights(parameters)
        weights = self.postprocess_weights(weights)

        if hasattr(self.cfgs, 'base') and self.cfgs.base == 'mmaction2':
            from models.base import build_mmaction_model
            model = build_mmaction_model(self.cfgs, mode='test')

        else:
            from models.build import build_model
            model = build_model(self.cfgs, mode='test')
        state_dict = OrderedDict(
            {k: torch.Tensor(v) 
            for k, v in zip(model.state_dict().keys(), weights)}
        )
        model.load_state_dict(state_dict, strict=False)
        
        eval_res = self.eval_fn(model, self.dl_test, self.device)
        if eval_res is None:
            return None
        
        metrics = {'top1_accuracy': eval_res['top1'], 'top5_accuracy': eval_res['top5']}
        with open(self.ckpt_dir + '/server_accs.txt', 'a') as f:
            f.write('{:.3f} {:.3f}\n'.format(metrics['top1_accuracy'], metrics['top5_accuracy']))
        if eval_res['top1'] > self.best_top1_acc:
            self.best_top1_acc = eval_res['top1']
            torch.save({'state_dict': model.state_dict()}, self.ckpt_dir + '/best.pth')

        torch.save({'state_dict': model.state_dict()},
                    self.ckpt_dir + '/{}.pth'.format(
                    time.strftime('%H%M%S', time.localtime())
                    )
        )

        return 0., metrics