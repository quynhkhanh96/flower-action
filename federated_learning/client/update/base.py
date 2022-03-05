import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

# For now, it's just local training for **classication problem**
# if it's a BaseLocalUpdate, it should implement basic training loop
# including batch generation (for all kind of problem), forward pass 
# and backpropagation
class BaseLocalUpdate(object):
    def __init__(self, train_loader, loss_func, args):
        self.args = args
        self.train_loader = train_loader
        self.loss_func = loss_func # torch.nn.CrossEntropyLoss()

    def train(self, net):
        net.to(self.args.device)
        net.train()

        if self.args.optimizer == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=self.args.lr)
        else:
            optimizer = optim.SGD(net.parameters(), lr=self.args.lr) 

        for iter in range(self.args.local_e):
            for images, labels in self.train_loader:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                
