import os
import torch
import torchvision
import numpy as np
import itertools as it
from collections import OrderedDict
import torch.optim as optim
import time

from base_client import Client
import stc_utils

class STCClient(Client):
    def __init__(self, compression, **kwargs):
        super().__init__(**kwargs)
        
        self.W = {name : value for name, value in self.model.named_parameters()}
        self.W_old = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
        self.dW = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
        self.dW_compressed = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
        # self.A = {name : torch.zeros(value.shape).to(self.cfgs.device) 
        #                 for name, value in self.W.items()}

        self.n_params = sum([T.numel() for T in self.W.values()])
        self.bits_sent = []

        optimizer_object = getattr(optim, self.cfgs.optimizer)
        optimizer_parameters = {k : v for k, v in self.cfgs.__dict__.items() 
                        if k in optimizer_object.__init__.__code__.co_varnames}

        self.optimizer = optimizer_object(self.model.parameters(), **optimizer_parameters)

        # # Learning Rate Schedule
        # self.scheduler = getattr(optim.lr_scheduler, self.hp['lr_decay'][0])(self.optimizer, **self.hp['lr_decay'][1])

        # State
        self.epoch = 0
        self.train_loss = 0.0  

        # Compression hyperparameters
        self.hp_comp = stc_utils.get_hp_compression(compression)

    def load_weights(self, global_model):
        self.model.load_state_dict(global_model.state_dict())
        self.model.to(self.cfgs.device)
        self.W = {name: value for name, value in self.model.named_parameters()}

    def compress_weight_update_up(self, client_id, compression=None, accumulate=False):
        if accumulate and compression[0] != "none":
            if os.path.exists(self.work_dir + f'/A_{client_id}.pth'):
                self.A = torch.load(self.work_dir + f'/A_{client_id}.pth')
            else:
                self.A = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                                for name, value in self.W.items()}
            # compression with error accumulation
            stc_utils.add(target=self.A, source=self.dW)
            stc_utils.compress(
                target=self.dW_compressed, source=self.A,
                compress_fun=stc_utils.compression_function(*compression)
            )
            stc_utils.subtract(target=self.A, source=self.dW_compressed)
            torch.save(self.A, self.work_dir + f'/A_{client_id}.pth')
        else:
            # compression without error accumulation
            stc_utils.compress(
                target=self.dW_compressed, source=self.dW, 
                compress_fun=stc_utils.compression_function(*compression)
            )
        
    def get_weights(self):
        return self.dW_compressed

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

        # compute weight updates
        ## W_old = W
        stc_utils.copy(target=self.W_old, source=self.W)
        ## W = SGD
        local_trainer.train(self.model, client_id)
        ## dW = W - W_old
        stc_utils.subtract_(target=self.dW, minuend=self.W, subtrachend=self.W_old)
        
        # compress weight updates up
        self.compress_weight_update_up(client_id,
                                compression=self.hp_comp['compression_up'],
                                accumulate=self.hp_comp['accumulation_up'])
        
        return self.get_weights(), len(train_loader.dataset)