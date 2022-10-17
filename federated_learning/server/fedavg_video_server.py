import flwr 
from flwr.common import parameters_to_weights, weights_to_parameters
from flwr.server.strategy.aggregate import aggregate
import numpy as np
import torch 
from models.build import build_model
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

class SortedFedAvgVideoStrategy(FedAvgVideoStrategy):
    def __init__(self, num_selected, **kwargs):
        self.num_selected = num_selected # number of clients' weights selected
        super(SortedFedAvgVideoStrategy, self).__init__(**kwargs) 
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

        # Sort clients by their local accuracy
        accs = np.array([fit_res.metrics['top1'] for _, fit_res in results])
        inds = np.argsort(accs)[-self.num_selected:]
        # Logs for debugging
        with open(self.ckpt_dir + '/server_selected.txt', 'a') as f:
            accs_str = ' '.join([str(acc) for acc in accs]) 
            inds_str = ' '.join([str(idx) for idx in inds])
            f.write(f'Round {rnd}: Accuracies {accs_str}, Selected {inds_str}\n')

        # Convert results
        weights_results = []
        for idx, (_, fit_res) in enumerate(results):
            if idx in inds:
                weights = parameters_to_weights(fit_res.parameters)
                weights_results.append((weights, fit_res.num_examples))

        return weights_to_parameters(aggregate(weights_results)), {}