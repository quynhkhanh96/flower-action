import flwr 
from flwr.common import parameters_to_weights
import numpy as np

class FedAvgStrategy(flwr.server.strategy.FedAvg):
    def __init__(self, model_name, num_classes, device, **kwargs):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        super(FedAvgStrategy, self).__init__(**kwargs) 

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
            from models.classification import CNN as Fed_Model
            net = Fed_Model(num_classes=self.num_classes)
        elif self.model_name == 'MLP':
            from models.classification import MLP as Fed_Model
            net = Fed_Model(num_classes=self.num_classes)
        elif self.model_name == 'ResNet':
            from models.classification import ResNet as Fed_Model
            net = Fed_Model(num_classes=self.num_classes) 
        elif self.model_name == 'MobileNetV3':
            from models.classification import MobileNetV3 as Fed_Model
            net = Fed_Model(num_classes=self.num_classes)
        elif self.model_name == 'MoViNet':
            from models.video.movinets.models import MoViNet as Fed_Model
            from models.video.movinets.config import _C
            net = Fed_Model(_C.MODEL.MoViNetA0, causal=True, 
                            pretrained=True , num_classes=self.num_classes)
        else:
            raise ValueError(f'No models implemented for {self.model_name} model.')
        # net = Fed_Model(num_classes=self.num_classes)
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
