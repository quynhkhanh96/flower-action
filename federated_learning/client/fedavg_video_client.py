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
    def __init__(self, client_id, dl_train, dl_test,
                        model, loss_fn, local_update, 
                        eval_fn, cfgs):
        
        self.client_id = client_id
        self.dl_train = dl_train
        self.dl_test = dl_test

        self.model = model 
        self.loss_fn = loss_fn 

        self.eval_fn = eval_fn 
        self.local = local_update
        self.cfgs = cfgs 

    def get_parameters(self) -> ParametersRes:
        weights: Weights = [val.cpu().numpy() 
                for _, val in self.model.state_dict().items()]
        parameters = weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    @staticmethod
    def postprocess_weights(weights):
        for i, w in enumerate(weights):
            try:
                _ = len(w)
            except:
                weights[i] = np.array([0])

        return weights

    def fit(self, ins: FitIns) -> FitRes:
        # set local model weights with that of the new global model
        weights: Weights = parameters_to_weights(ins.parameters)
        weights = self.postprocess_weights(weights)
        
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        self.model.load_state_dict(state_dict)

        fit_begin = timeit.default_timer()
        # train model locally 
        self.local.train(model=self.model)

        weights_prime: Weights = [val.cpu().numpy() 
                            for _, val in self.model.state_dict().items()]
        weights_prime = self.postprocess_weights(weights_prime)

        params_prime = weights_to_parameters(weights_prime)
        num_examples_train = len(self.dl_train.dataset)
        fit_duration = timeit.default_timer() - fit_begin 
        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            # num_examples_ceil=num_examples_train,
            # fit_duration=fit_duration,            
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Compute loss of the new global model on this client's data"""
        weights: Weights = parameters_to_weights(ins.parameters)
        weights = self.postprocess_weights(weights)

        state_dict = OrderedDict(
            {k: torch.Tensor(v) 
            for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        self.model.load_state_dict(state_dict)
        # self.model.to(self.cfgs.device)
        # self.model.eval()
        
        # Evaluate the updated model on the local dataset
        topk_accuracy = self.eval_fn(self.model, self.dl_test, 
                                        self.cfgs.device)

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            loss=0., num_examples=len(self.ds_test), 
            metrics=topk_accuracy
        )