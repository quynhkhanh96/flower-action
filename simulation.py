import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

import flwr
from flwr.common.typing import Scalar
import ray
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
from typing import Dict, Callable, Optional, Tuple, List
from flwr.common import parameters_to_weights, weights_to_parameters
from flwr.common import (
    FitIns, FitRes, Parameters, 
    ParametersRes, EvaluateIns, 
    EvaluateRes, Weights)

from federated_learning.client.update.video_base import VideoLocalUpdate
from federated_learning.server.fedavg_video_server import FedAvgVideoStrategy
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from datasets.frame_dataset import FrameDataset, get_client_loaders
from models.build import build_model, build_loss
from evaluation.video_recognition import evaluate_topk_accuracy
import yaml 
from utils.parsing import Dict2Class
from utils import seed_torch
import functools

#!python -m datasets.frame_dataset --n_clients=$POOL_SIZE --data_dir=$DATA_DIR --train_ann=$DATA_DIR/train.txt --mode="iid"
POOL_SIZE = 20 # number of total clients
NUM_ROUNDS = 30
NUM_CLIENT_CPUS = 2
NUM_CLIENT_GPUS = 1
NUM_SELECTED = 5

SERVER_DEVICE = "cuda:0"
CLIENT_DEVICE = "cuda:0"

CFG_PATH = "examples/afosr2022/configs/afosr_fedavg_sorted.yaml"
DATA_DIR = "/ext_data2/comvis/khanhdtq/afosr2022"
WORK_DIR = f"{DATA_DIR}/simulation_exps"

seed_torch(1234)

# Configuration
with open(CFG_PATH, 'r') as yamlfile:
    cfgs = yaml.load(yamlfile, Loader=yaml.FullLoader)
cfgs = Dict2Class(cfgs)

cfgs.num_C = POOL_SIZE
cfgs.min_sample_size = NUM_SELECTED
cfgs.min_num_clients = NUM_SELECTED
cfgs.device = CLIENT_DEVICE

# Evaluation function
eval_fn = evaluate_topk_accuracy

# =======================#
#       Server           #
# =======================#
def fit_config(rnd, cfgs):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(cfgs.local_e),
        "batch_size": str(cfgs.train_bz)
    }
    return config

_, test_loader = get_client_loaders(0, DATA_DIR, cfgs)

strategy = FedAvgVideoStrategy(
    cfgs=cfgs,
    dl_test=test_loader,
    ckpt_dir=WORK_DIR,
    device=SERVER_DEVICE,
    fraction_fit=NUM_SELECTED / POOL_SIZE,
    min_fit_clients=cfgs.min_sample_size,
    min_available_clients=cfgs.min_num_clients,
    eval_fn=eval_fn,
    on_fit_config_fn=functools.partial(fit_config, cfgs=cfgs),
)

# =======================#
#       Clients          #
# =======================#
def get_client_local_loaders(client_id, data_dir, work_dir, cfgs, workers):
    
    scaler = T.Resize(((cfgs.height, cfgs.width)))
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
    transform= T.Compose([T.ToPILImage(), scaler, T.ToTensor(), normalize])  
    
    # split client's video ids 8:2 to create local train & val set
    with open(data_dir + f'/client_{client_id}_train.txt') as f:
        lines = [l.strip() for l in f.readlines()]
        video_ids = [l.strip().split(' ')[0] for l in lines]
        labels = [int(l.strip().split(' ')[1]) for l in lines]

    train_inds, val_inds = train_test_split(list(range(len(video_ids))), 
                                            test_size=0.33,
                                            random_state=int(cfgs.seed))
    local_train_path = work_dir + f'/client_{client_id}_local_train.txt'
    local_val_path = work_dir + f'/client_{client_id}_local_val.txt'
    if not os.path.exists(local_train_path):
        with open(local_train_path, 'a') as f:
            for idx in train_inds:
                f.write('{} {}\n'.format(video_ids[idx], labels[idx]))
    if not os.path.exists(local_val_path):
        with open(local_val_path, 'a') as f:
            for idx in val_inds:
                f.write('{} {}\n'.format(video_ids[idx], labels[idx]))
    
    train_set = FrameDataset(
        frame_dir=data_dir + '/rgb_frames',
        annotation_file_path=local_train_path,
        n_frames=cfgs.seq_len,
        mode='train',
        transform=transform,
        use_albumentations=False,
    )
    trn_kwargs = {"num_workers": workers, "pin_memory": True, 
                  "drop_last": False, "shuffle": True}
    train_loader = DataLoader(train_set, batch_size=cfgs.train_bz, **trn_kwargs)

    val_set = FrameDataset(
        frame_dir=data_dir + '/rgb_frames',
        annotation_file_path=local_val_path,
        n_frames=cfgs.seq_len,
        mode='test',
        transform=transform,
        use_albumentations=False,
    )
    val_kwargs = {"num_workers": workers, "pin_memory": True, 
                  "drop_last": False, "shuffle": False}
    val_loader = DataLoader(val_set, batch_size=cfgs.train_bz, **val_kwargs)
    return train_loader, val_loader

class SimulationClient(flwr.client.Client):
    def __init__(self, data_dir, work_dir, client_id, model, loss_fn, 
                        eval_fn, cfgs):
        self.data_dir = data_dir
        self.work_dir = work_dir
        
        self.client_id = client_id

        self.model = model 
        self.loss_fn = loss_fn 

        self.eval_fn = eval_fn 
        self.cfgs = cfgs 

    def get_parameters(self):
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

    def fit(self, ins):
        # set local model weights with that of the new global model
        weights: Weights = parameters_to_weights(ins.parameters)
        weights = self.postprocess_weights(weights)
        
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        self.model.load_state_dict(state_dict)
        
        # data loaders are only created when client is selected
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        train_loader, val_loader = get_client_local_loaders(self.client_id, 
                                                            self.data_dir, self.work_dir, self.cfgs, 
                                                            num_workers)
        
        # create local trainer
        local_update = VideoLocalUpdate(train_loader=train_loader,
                                        loss_fn=self.loss_fn, cfgs=self.cfgs)
        # train model locally 
        local_update.train(model=self.model, client_id=self.client_id)

        weights_prime: Weights = [val.cpu().numpy() 
                            for _, val in self.model.state_dict().items()]
        weights_prime = self.postprocess_weights(weights_prime)

        # num_examples_train = len(train_loader.dataset)
        # return weights_prime, num_examples_train, {}
        params_prime = weights_to_parameters(weights_prime)
        num_examples_train = len(train_loader.dataset)
        return FitRes(
            parameters=params_prime,
            num_examples=num_examples_train        
        )

    def evaluate(self, ins):
        # Return the number of evaluation examples and the evaluation result (loss)
        # return 0., 1, {'top1': 0., 'top5': 0.}
        return EvaluateRes(
            loss=0., num_examples=1, 
            metrics={'top1': 0., 'top5': 0.}
        )

def client_fn(cid: str):
    client_id = int(cid)
    
    # model
    model = build_model(cfgs, mode='train')
    # loss function
    criterion = build_loss(cfgs)
    
    fl_client = SimulationClient(
            data_dir=DATA_DIR,
            work_dir=WORK_DIR,
            client_id=client_id,
            model=model, loss_fn=criterion,
            eval_fn=eval_fn, cfgs=cfgs
    )
    
    return fl_client

# =======================#
#       Run              #
# =======================#

client_resources = {
    "num_cpus": NUM_CLIENT_CPUS,
    "num_gpus": NUM_CLIENT_GPUS
}  # each client will get allocated 02 CPUs, 01 GPU

# (optional) specify Ray config
ray_init_args = {"include_dashboard": False}

# start simulation
flwr.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=POOL_SIZE,
    client_resources=client_resources,
    num_rounds=NUM_ROUNDS,
    strategy=strategy,
    ray_init_args=ray_init_args,
)