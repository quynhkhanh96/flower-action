import sys, os
sys.path.insert(0, os.path.abspath('..'))
from collections import OrderedDict
import numpy as np 
import torch

class Client:
    def __init__(self, data_dir, work_dir, model,
                        loss_fn, eval_fn, cfgs):
        self.data_dir = data_dir 
        self.work_dir = work_dir
        self.model = model
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.cfgs = cfgs
        if hasattr(cfgs, 'base') and cfgs.base == 'mmaction2':
            self.mmaction_base = True
        else:
            self.mmaction_base = False  

    def get_data_loaders(self, client_id):
        if self.mmaction_base:
            from datasets.frame_dataset import get_client_mmaction_loaders
            train_loader, val_loader = get_client_mmaction_loaders(
                client_id, self.data_dir, self.cfgs 
            )
        else:
            from datasets.frame_dataset import get_client_loaders
            train_loader, val_loader = get_client_loaders(client_id,
                self.data_dir, self.cfgs
            )
        return train_loader, val_loader
    
    @staticmethod
    def postprocess_weights(weights):
        for i, w in enumerate(weights):
            try:
                _ = len(w)
            except:
                weights[i] = np.array([w.item()])

        return weights
    
    def load_weights(self, global_model):
        self.model = global_model

    def get_weights(self):
        weights = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return self.postprocess_weights(weights)

    def train(self, rnd, client_id, global_model):
        # load the new global weights
        self.load_weights(global_model)

        # create the data loaders of the current client
        train_loader, val_loader = self.get_data_loaders(client_id)
        if self.mmaction_base:
            from federated_learning.client.update.video_base import MMActionLocalUpdate
            local_trainer = MMActionLocalUpdate(train_loader=train_loader,
                                            loss_fn=self.loss_fn, 
                                            cfgs=self.cfgs)
        else:
            from federated_learning.client.update.video_base import VideoLocalUpdate
            local_trainer = VideoLocalUpdate(train_loader=train_loader,
                                            loss_fn=self.loss_fn, 
                                            cfgs=self.cfgs)
        # train loop
        local_trainer.train(self.model, client_id)
        
        return self.get_weights(), len(train_loader.dataset)