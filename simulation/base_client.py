import sys, os
sys.path.insert(0, os.path.abspath('..'))
from collections import OrderedDict
import numpy as np 
import torch
from datasets.frame_dataset import get_client_loaders
from federated_learning.client.update.video_base import VideoLocalUpdate

class Client:
    def __init__(self, data_dir, work_dir, model,
                        loss_fn, eval_fn, cfgs):
        self.data_dir = data_dir 
        self.work_dir = work_dir
        self.model = model
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.cfgs = cfgs 

    def get_data_loaders(self, client_id):
        train_loader, val_loader = get_client_loaders(client_id,
            self.data_dir, self.work_dir, self.cfgs
        )
        return train_loader, val_loader
    
    @staticmethod
    def postprocess_weights(weights):
        for i, w in enumerate(weights):
            try:
                _ = len(w)
            except:
                weights[i] = np.array([0])

        return weights
    
    def load_weights(self, weights):
        weights = self.postprocess_weights(weights)
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)}
        )
        self.model.load_state_dict(state_dict)

    def get_weights(self):
        weights = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return self.postprocess_weights(weights)

    def train(self, rnd, client_id, global_model):
        # load the new global weights
        self.model = global_model

        train_loader, val_loader = self.get_data_loaders(client_id)
        local_trainer = VideoLocalUpdate(train_loader=train_loader,
                                        loss_fn=self.loss_fn, 
                                        cfgs=self.cfgs)
        # train loop
        local_trainer.train(self.model, client_id)
        
        return self.get_weights(), len(train_loader.dataset)