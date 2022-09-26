import flwr 
from flwr.common import parameters_to_weights, weights_to_parameters
from flwr.server.strategy.aggregate import aggregate
import numpy as np
import torch 
from models.build import build_model
from collections import OrderedDict
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
                weights[i] = np.array([0])

        return weights

    def evaluate(self, parameters):
        if self.eval_fn is None:
            return None

        weights = parameters_to_weights(parameters)
        weights = self.postprocess_weights(weights)

        model = build_model(self.cfgs, mode='test')
        state_dict = OrderedDict(
            {k: torch.Tensor(v) 
            for k, v in zip(model.state_dict().keys(), weights)}
        )
        model.load_state_dict(state_dict)
        
        eval_res = self.eval_fn(model, self.dl_test, self.device)
        if eval_res is None:
            return None
        
        metrics = {'top1_accuracy': eval_res['top1'], 'top5_accuracy': eval_res['top5']}
        # wandb.log({f"top1_accuracy": eval_res['top1']})
        # wandb.log({f"top5_accuracy": eval_res['top5']})
        
        if eval_res['top1'] > self.best_top1_acc:
            self.best_top1_acc = eval_res['top1']
            torch.save({'state_dict': model.state_dict()}, self.ckpt_dir + '/best.pth')

        return 0., metrics

class ThresholdedFedAvgVideoStrategy(FedAvgVideoStrategy):
    def aggregate_fit(
        self, rnd,
        results, failures,
    ):

        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weights_results = []
        for client, fit_res in results:
            weights = parameters_to_weights(fit_res.parameters)
            if all([len(weight) == 0 for weight in weights]):
                continue
            weights_results.append((weights, fit_res.num_examples))

        return weights_to_parameters(aggregate(weights_results)), {}