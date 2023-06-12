import os
import torch
import torch.optim as optim

from base_client import Client
import qsgd_utils

class QSGDClient(Client):
    def __init__(self, random, n_bit, no_cuda, **kwargs):
        super().__init__(**kwargs)

        self.quantizer = qsgd_utils.QSGDQuantizer(
            random, n_bit, no_cuda
        )

        self.W = {name: value for name, value in self.model.named_parameters()}
        self.W_old = {name: torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
    
    def load_weights(self, global_model):
        self.model.load_state_dict(global_model.state_dict())
        self.model.to(self.cfgs.device)
        self.W = {name: value for name, value in self.model.named_parameters()}

    def compress_weight_update_up(self):
        res = {}
        for lname, lgrad in self.dW.items():
            res[lname] = self.quantizer.quantize(lgrad)
        
        return res

    def train(self, rnd, client_id, global_model):
        # load the new global weights
        self.load_weights(global_model) 

        # create the data loaders of the current client
        train_loader, _ = self.get_data_loaders(client_id)
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
        
        # compute weight updates
        ## W_old = W
        qsgd_utils.copy(target=self.W_old, source=self.W)
        ## train with local data
        local_trainer.train(self.model, client_id)
        ## dW = W - W_old
        qsgd_utils.subtract_(target=self.dW, minuend=self.W, subtrachend=self.W_old)

        # compress weight update up
        res = self.compress_weight_update_up()

        return res, len(train_loader.dataset)
