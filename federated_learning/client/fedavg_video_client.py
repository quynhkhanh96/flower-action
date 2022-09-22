from flwr.common.typing import FitRes, Parameters
import flwr 
from flwr.common import FitIns, FitRes, ParametersRes, EvaluateIns, EvaluateRes, Weights
from flwr.common import parameters_to_weights, weights_to_parameters

import timeit
import numpy as np 
import torch 
from collections import OrderedDict

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
        self.local.train(model=self.model, client_id=self.client_id)

        weights_prime: Weights = [val.cpu().numpy() 
                            for _, val in self.model.state_dict().items()]
        weights_prime = self.postprocess_weights(weights_prime)

        params_prime = weights_to_parameters(weights_prime)
        num_examples_train = len(self.dl_train.dataset)
        fit_duration = timeit.default_timer() - fit_begin 
        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train        
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
        
        # Evaluate the updated model on the local dataset
        topk_accuracy = self.eval_fn(self.model, self.dl_test, 
                                        self.cfgs.device)

        # Return the number of evaluation examples and the evaluation result (loss)
        return EvaluateRes(
            loss=0., num_examples=len(self.ds_test), 
            metrics=topk_accuracy
        )

class ThresholdedFedAvgVideoClient(FedAvgVideoClient):
    def __init__(self, work_dir, **kwargs):
        self.work_dir = work_dir
        self.round = 0
        super(ThresholdedFedAvgVideoClient, self).__init__(**kwargs) 

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
        self.local.train(model=self.model, client_id=self.client_id)

        # Evaluate the updated model on the local dataset
        topk_accuracy = self.eval_fn(self.model, self.dl_test, 
                                        self.cfgs.device)
        
        # # For now save the topk accs by round
        # with open(self.work_dir + f'/client_{self.client_id}_accs.txt', 'a') as f:
        #     f.write('{:.3f} {:.3f}\n'.format(topk_accuracy['top1'],
        #                                     topk_accuracy['top5']))

        # Thresholding (average top1 accuracy of that round)
        if self.client_id == 0:
            prob = 1
        else:
            thresh = 0
            n_clients = int(self.cfgs.num_C)
            for _ in range(n_clients):
                with open(self.work_dir + f'/client_{self.client_id}_accs.txt', 'r') as f:
                    top1_accs = [float(x.strip().split(' ')[0]) for x in f.readlines()]
                thresh += top1_accs[self.round]
            thresh /= n_clients
            prob = topk_accuracy['top1'] >= thresh

        if prob:
            weights_prime: Weights = [val.cpu().numpy() 
                                for _, val in self.model.state_dict().items()]
            weights_prime = self.postprocess_weights(weights_prime)
        else:
            # If the top1 acc does not exceed the average top1 acc of that round
            # the client will not send the weights by only send the empty np.darray
            # of each layer
            weights_prime = [np.array([]) for _ in self.model.state_dict().items()]
            print(f'At round {self.round}, client {self.client_id} does not send weight to server.')
            with open(self.work_dir + f'/client_{self.client_id}_drops.txt', 'a') as f:
                f.write(f'{self.round}\n')
        params_prime = weights_to_parameters(weights_prime)

        num_examples_train = len(self.dl_train.dataset)
        self.round += 1
        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train       
        )
