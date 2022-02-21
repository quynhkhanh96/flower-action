import flwr 
from flwr.common import parameters_to_weights
import numpy as np

class FedAdamStrategy(flwr.server.strategy.FedAdam):
    def __init__(self, model_name, device, **kwargs):
        self.model_name = model_name
        self.device = device
        super(FedAdamStrategy, self).__init__(**kwargs) 

    def evaluate(self, parameters):
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            return None

        weights = parameters_to_weights(parameters)
        for i, w in enumerate(weights):
            try:
                _ = len(w)
            except:
                weights[i] = np.array([0])

        if self.model_name == 'CNN':
            from models import CNN as Fed_Model
        elif self.model_name == 'MLP':
            from models import MLP as Fed_Model
        elif self.model_name == 'ResNet':
            from models import ResNet as Fed_Model 
        net = Fed_Model()
        net.set_weights(weights)
        net.to(self.device)

        eval_res = self.eval_fn(net)
        if eval_res is None:
            return None
        loss, other = eval_res
        if isinstance(other, float):
            metrics = {"accuracy": other}
        else:
            metrics = other
        return loss, metrics
