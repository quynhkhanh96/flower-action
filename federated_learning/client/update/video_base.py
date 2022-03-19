import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class VideoLocalUpdate:
    def __init__(self, train_loader, loss_func, args):
        self.args = args
        self.train_loader = train_loader
        self.loss_func = loss_func

    def train(self, net):
        net.to(self.args.device)
        net.train()

        if self.args.optimizer == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=self.args.lr)
        else:
            optimizer = optim.SGD(net.parameters(), lr=self.args.lr) 
        
        for _ in range(self.args.local_e):
            net.clean_activation_buffers()
            optimizer.zero_grad()
            for data, _, target in self.train_loader:
                data, target = data.to(self.args.device), target.to(self.args.device)
                out = net(data)
                loss = self.loss_func(out, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                net.clean_activation_buffers()

from mmaction.apis import train_model
class MMAction2LocalUpdate:
    def __init__(self, train_dataset, mmaction_cfg):
        self.train_dataset = train_dataset
        self.mmaction_cfg = mmaction_cfg 

    def train(self, model):
        train_model(model, [self.train_dataset], self.mmaction_cfg, 
                        distributed=False, validate=False)

