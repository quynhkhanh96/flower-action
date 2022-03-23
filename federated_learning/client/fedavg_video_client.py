from flwr.common.typing import FitRes, Parameters
import flwr 
from flwr.common import FitIns, FitRes, ParametersRes, EvaluateIns, EvaluateRes, Weights
from flwr.common import parameters_to_weights, weights_to_parameters

import timeit
import numpy as np 
import torch 
from collections import OrderedDict

from mmaction.models import build_recognizer
from mmcv.runner import load_state_dict
import copy 

class FedAvgVideoClient(flwr.client.Client):
    def __init__(self, client_id, ds_train, ds_test,
                        model, local_update,
                        eval_fn, cfg):
        
        self.client_id = client_id
        self.ds_train = ds_train
        self.ds_test = ds_test
        self.model = model 

        self.eval_fn = eval_fn 
        self.local = local_update
        self.cfg = cfg 

    def get_parameters(self) -> ParametersRes:
        # TODO: add get_weights() method to mmaction2 model 
        weights: Weights = [val.cpu().numpy() 
                            for _, val in self.model.state_dict().items()]
        parameters = weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        # set local model weights with that of the new global model
        weights: Weights = parameters_to_weights(ins.parameters)
        for i, w in enumerate(weights):
            try:
                _ = len(w)
            except:
                weights[i] = np.array([0])
        # TODO: add set_weights() method to mmaction2 model 
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        load_state_dict(self.model, state_dict)

        fit_begin = timeit.default_timer()

        # train model locally 
        self.local.train(model=self.model)
        weights_prime: Weights = [val.cpu().numpy() 
                            for _, val in self.model.state_dict().items()]
        for i, w in enumerate(weights_prime):
            try:
                _ = len(w)
            except:
                weights_prime[i] = np.array([0])
        params_prime = weights_to_parameters(weights_prime)
        num_examples_train = len(self.ds_train)
        fit_duration = timeit.default_timer() - fit_begin 
        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            num_examples_ceil=num_examples_train,
            fit_duration=fit_duration,            
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Compute loss of the new global model on this client's data"""
        weights: Weights = parameters_to_weights(ins.parameters)
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
        
        # Evaluate the updated model on the local dataset
        topk_accuracy = self.eval_fn(model, self.ds_test, 'cuda')

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            loss=0., num_examples=len(self.ds_test), 
            metrics=topk_accuracy
        )