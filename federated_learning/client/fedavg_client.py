from flwr.common.typing import FitRes, Parameters
import flwr 
from flwr.common import FitIns, FitRes, ParametersRes, EvaluateIns, EvaluateRes, Weights
from flwr.common import parameters_to_weights, weights_to_parameters

import timeit
import numpy as np 
from update.base import BaseLocalUpdate

class FedAvgClient(flwr.client.Client):
    def __init__(self, client_id, dl_train, dl_test, net, loss_func, eval_fn, args):
        self.client_id = client_id
        self.dl_train = dl_train
        self.dl_test = dl_test
        self.net = net 
        self.eval_fn = eval_fn
        self.args = args 

        self.local = BaseLocalUpdate(dl_train, loss_func, args)

    def get_parameters(self) -> ParametersRes:
        weights: Weights = self.net.get_weights()
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
        self.net.set_weights(weights)
        fit_begin = timeit.default_timer()

        # Train model 
        self.local.train(net=self.net)
        weights_prime: Weights = self.net.get_weights()
        for i, w in enumerate(weights_prime):
            try:
                _ = len(w)
            except:
                weights_prime[i] = np.array([0])
        params_prime = weights_to_parameters(weights_prime)

        num_examples_train = len(self.dl_train.dataset)
        fit_duration = timeit.default_timer() - fit_begin 
        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            num_examples_ceil=num_examples_train,
            fit_duration=fit_duration,            
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Compute loss of the new global model on this client's data"""

        weights = parameters_to_weights(ins.parameters)
        for i, w in enumerate(weights):
            try:
                _ = len(w)
            except:
                weights[i] = np.array([0])
        # Use provided weights to update the local model
        self.net.set_weights(weights)

        # Evaluate the updated model on the local dataset
        loss, accuracy = self.eval_fn(self.net, self.dl_test) 

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            loss=float(loss), num_examples=len(self.dl_test.dataset), accuracy=float(accuracy)
        )


