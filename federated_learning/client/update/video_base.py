import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AverageMeter

# class VideoLocalUpdate:
#     def __init__(self, train_loader, loss_func, args):
#         self.args = args
#         self.train_loader = train_loader
#         self.loss_func = loss_func

#     def train(self, net):
#         net.to(self.args.device)
#         net.train()

#         if self.args.optimizer == 'Adam':
#             optimizer = optim.Adam(net.parameters(), lr=self.args.lr)
#         else:
#             optimizer = optim.SGD(net.parameters(), lr=self.args.lr) 
        
#         for _ in range(self.args.local_e):
#             net.clean_activation_buffers()
#             optimizer.zero_grad()
#             for data, _, target in self.train_loader:
#                 data, target = data.to(self.args.device), target.to(self.args.device)
#                 out = net(data)
#                 loss = self.loss_func(out, target)
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 net.clean_activation_buffers()


class VideoLocalUpdate:
    def __init__(self, train_loader, loss_fn, optimizer, args):
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.args = args 

    def train(self, model):
        model.to(self.args.device)
        model.train()
        losses = AverageMeter("loss")	
        for batch_idx, (imgs, labels, _) in enumerate(self.train_loader):			
            imgs, labels = imgs.to(self.args.device), labels.to(self.args.device)
            imgs, labels = Variable(imgs), Variable(labels)		
            outputs = model(imgs) 		
            loss = self.loss_fn(outputs, labels)	        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()		 		
            losses.update(loss.item(), labels.size(0))
            if (batch_idx + 1) % self.args.print_freq == 0:
                print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, 
                                    len(self.train_loader), losses.val, losses.avg))

from mmaction.apis import train_model
class MMAction2LocalUpdate:
    def __init__(self, train_dataset, mmaction_cfg):
        self.train_dataset = train_dataset
        self.mmaction_cfg = mmaction_cfg 
        self.lr_config_policy = mmaction_cfg.lr_config['policy']

    def train(self, model):
        # print(self.mmaction_cfg.lr_config)
        if 'policy' not in self.mmaction_cfg.lr_config:
            self.mmaction_cfg.lr_config['policy'] = self.lr_config_policy
            
        train_model(model, [self.train_dataset], self.mmaction_cfg, 
                        distributed=False, validate=False)

