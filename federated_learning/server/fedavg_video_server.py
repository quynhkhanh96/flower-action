import flwr 
from flwr.common import parameters_to_weights
import numpy as np
import torch 
# from mmcv.runner import load_checkpoint
from mmaction.models import build_recognizer
from mmcv.runner import load_state_dict
from collections import OrderedDict
import copy 
from flwr.server.strategy import FedAvg

class FedAvgVideoStrategy(flwr.server.strategy.FedAvg):
    def __init__(self, cfg, test_dataset, device, **kwargs):
        self.cfg = cfg 
        self.test_dataset = test_dataset 
        # self.device = torch.device("cuda" if torch.cuda.is_available() 
        #                             else "cpu")
        self.device = device 
        super(FedAvgVideoStrategy, self).__init__(**kwargs) 

    def evaluate(self, parameters):
        if self.eval_fn is None:
            return None

        weights = parameters_to_weights(parameters)
        for i, w in enumerate(weights):
            try:
                _ = len(w)
            except:
                weights[i] = np.array([0])

        cfg = copy.deepcopy(self.cfg)
        cfg.model.backbone.pretrained = None
        model = build_recognizer(cfg.model, test_cfg=cfg.get('test_cfg'))
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
        )
        load_state_dict(model, state_dict)
        model.cfg = cfg
        # model.to('cuda')
        model.to(self.device)
        model.eval()

        # model = build_model(
        #     self.cfg.model,
        #     train_cfg=self.cfg.get('train_cfg'),
        #     test_cfg=self.cfg.get('test_cfg'))
        # # TODO: add set_weights() method to mmaction2 model 
        # state_dict = OrderedDict(
        #     {k: torch.Tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
        # )
        # # load_checkpoint(model, state_dict) #, map_location='cpu')
        # load_state_dict(model, state_dict)
        
        eval_res = self.eval_fn(model, self.test_dataset, self.device)
        if eval_res is None:
            return None
        top1_acc, top5_acc = eval_res
        metrics = {'top1_accuracy': top1_acc, 'top5_accuracy': top5_acc}
        return metrics

        

        