import os
import torch
import torch.optim as optim
import numpy as np

from base_client import Client
import qsgd_utils

class QSparseSGDClient(Client):
    def __init__(self, random, n_bit, no_cuda, **kwargs):
        super().__init__(**kwargs)

        self.quantizer = qsgd_utils.QSGDQuantizer(
            random, n_bit, no_cuda
        )

        self.W = {name: value for name, value in self.model.named_parameters()}
        self.W_old = {name: torch.zeros(value.shape).to(self.cfgs.device) 
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

        # initialize local trainer
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
        ## compress updates with error compensation
        if os.path.exists(self.work_dir + f'/A_{client_id}.pth'):
            error_accum = torch.load(self.work_dir + f'/A_{client_id}.pth')
        else:
            error_accum = {name: torch.zeros(value.shape).to(self.cfgs.device) 
                            for name, value in self.W.items()}
        grads, q_grads = {}, {}
        with torch.no_grad():
            for lname in self.W:
                lgrad = error_accum[lname] + self.W_old[lname] - self.W[lname]
                grads[lname] = lgrad
                q_grads[lname] = self.quantizer.quantize(lgrad)        

        # update error-feedback
        with torch.no_grad():
            for lname in error_accum:
                error_accum[lname] = grads[lname] - self.quantizer.dequantize(q_grads[lname])
        torch.save(error_accum, self.work_dir + f'/A_{client_id}.pth')
        
        return q_grads, len(train_loader.dataset)