from flwr.common.typing import FitRes, Parameters
import flwr 
from flwr.common import FitIns, FitRes, ParametersRes, EvaluateIns, EvaluateRes, Weights
from flwr.common import parameters_to_weights, weights_to_parameters

import numpy as np 
import torch 
from collections import OrderedDict

class FedBNVideoClient(flwr.client.Client):
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

    def get_parameters(self):
        self.model.train()
        weights = [val.cpu().numpy() 
                for name, val in self.model.state_dict().items()
                if 'bn' not in name]
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

    def set_parameters(self, parameters):
        self.model.train()

        weights = parameters_to_weights(parameters)
        weights = self.postprocess_weights(weights)

        state_dict = OrderedDict()
        keys = [k for k in self.model.state_dict().keys()]
        for i in range(len(weights)):
            if 'bn' not in keys[i]:
                 state_dict[keys[i]] = torch.tensor(weights[i])
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, ins):
        # set local model weights with that of the new global model
        self.set_parameters(ins.parameters)

        # train model locally 
        self.local.train(model=self.model, client_id=self.client_id)

        weights_prime: Weights = [val.cpu().numpy() 
                            for _, val in self.model.state_dict().items()]
        weights_prime = self.postprocess_weights(weights_prime)

        params_prime = weights_to_parameters(weights_prime)
        num_examples_train = len(self.dl_train.dataset)
        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train        
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Compute loss of the new global model on this client's data"""
        self.set_parameters(ins.parameters)
        
        # Evaluate the updated model on the local dataset
        topk_accuracy = self.eval_fn(self.model, self.dl_test, 
                                        self.cfgs.device)

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            loss=0., num_examples=len(self.dl_test), 
            metrics=topk_accuracy
        )