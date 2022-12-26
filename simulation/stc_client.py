import torch
import torchvision
import numpy as np
import itertools as it
import re
from math import sqrt
import random

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import time

from base_client import Client
import stc_utils

class STCClient:
    def __init__(self, data_dir, work_dir, model,
                        loss_fn, eval_fn, cfgs):
        super().__init__(data_dir, work_dir, model,
                        loss_fn, eval_fn, cfgs)
        
        self.W = {name : value for name, value in self.model.named_parameters()}
        self.W_old = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
        self.dW = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
        self.dW_compressed = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
        self.A = {name : torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}

        self.n_params = sum([T.numel() for T in self.W.values()])
        self.bits_sent = []

        # Optimizer (specified in self.hp, initialized using the suitable parameters from self.hp)
        optimizer_object = getattr(optim, self.cfgs.optimizer)
        optimizer_parameters = {k : v for k, v in self.cfgs.__dict__.items() 
                        if k in optimizer_object.__init__.__code__.co_varnames}

        self.optimizer = optimizer_object(self.model.parameters(), **optimizer_parameters)

        # # Learning Rate Schedule
        # self.scheduler = getattr(optim.lr_scheduler, self.hp['lr_decay'][0])(self.optimizer, **self.hp['lr_decay'][1])

        # State
        self.epoch = 0
        self.train_loss = 0.0  

    def load_weights(self, global_model):
        for name in self.W:
            self.W.data = global_model[name].data.clone()

    def train(self, rnd, client_id, global_model): 
        # load the new global weights
        self.load_weights(global_model) 

