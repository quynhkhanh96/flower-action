import sys
import os 
sys.path.insert(0, os.path.abspath('..'))
import flwr 
from flwr.common import parameters_to_weights, weights_to_parameters
from flwr.server.strategy.aggregate import aggregate
import numpy as np
import torch
from collections import OrderedDict
import time 
from fedavg_video_server import FedAvgVideoStrategy
from utils import stc_ops, stc_compress

class STCVideoStrategy(FedAvgVideoStrategy):
    def __init__(self, compression, **kwargs):
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
        self.hp_comp = stc_compress.get_hp_compression(compression)

    

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
        # wandb.log({f"top1_accuracy": eval_res['top1']})
        # wandb.log({f"top5_accuracy": eval_res['top5']})
        with open(self.ckpt_dir + '/server_accs.txt', 'a') as f:
            f.write('{:.3f} {:.3f}\n'.format(metrics['top1_accuracy'],
                                            metrics['top5_accuracy']))
        if eval_res['top1'] > self.best_top1_acc:
            self.best_top1_acc = eval_res['top1']
            torch.save({'state_dict': model.state_dict()}, self.ckpt_dir + '/best.pth')

        torch.save({'state_dict': model.state_dict()},
                    self.ckpt_dir + '/{}.pth'.format(
                    time.strftime('%H%M%S', time.localtime())
                    )
        )

        return 0., metrics