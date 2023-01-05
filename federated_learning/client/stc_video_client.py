from flwr.common.typing import FitRes, Parameters
import flwr 
from flwr.common import FitIns, FitRes, ParametersRes, EvaluateIns, EvaluateRes, Weights
from flwr.common import parameters_to_weights, weights_to_parameters

import numpy as np 
import torch 
from collections import OrderedDict

from fedavg_video_client import FedAvgVideoClient

class STCVideoClient(FedAvgVideoClient):
    def __init__(self, **kwargs):
        super(STCVideoClient, self).__init__(**kwargs)