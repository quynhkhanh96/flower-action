import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import AverageMeter
from torch.nn.utils import clip_grad_norm_
# import wandb

class VideoLocalUpdate:
    def __init__(self, train_loader, loss_fn, cfgs):
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.cfgs = cfgs 

    def train(self, model, client_id):
        model.to(self.cfgs.device)
        model.train()

        # Rebuild optimizer everytime client receives new model weights from server
        if self.cfgs.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=self.cfgs.lr, momentum=0.9)
        else:
            raise ValueError(f'No implementation for {self.cfgs.optimizer} optimizer.')

        for epoch in range(self.cfgs.local_e):
            print(f'*************** Epoch {epoch} ***************')
            losses = AverageMeter("loss")	
            for batch_idx, (imgs, labels, _) in enumerate(self.train_loader):			
                imgs, labels = imgs.to(self.cfgs.device), labels.to(self.cfgs.device)
                imgs, labels = Variable(imgs), Variable(labels)		
                outputs = model(imgs) 		
                loss = self.loss_fn(outputs, labels)	        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()		 		
                losses.update(loss.item(), labels.size(0))
                # wandb.log({f"client{client_id}_loss": loss.item()})
                if (batch_idx + 1) % self.cfgs.print_freq == 0:
                    print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, 
                                        len(self.train_loader), losses.val, losses.avg))

class MMActionLocalUpdate:
    def __init__(self, train_loader, loss_fn, cfgs):
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.cfgs = cfgs
    
    def train(self, model, client_id):
        model.to(self.cfgs.device)
        model.train()

        # Rebuild optimizer everytime client receives new model weights from server
        if self.cfgs.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=self.cfgs.lr, momentum=0.9)
        else:
            raise ValueError(f'No implementation for {self.cfgs.optimizer} optimizer.')

        for epoch in range(self.cfgs.local_e):
            model.train()
            print(f'*************** Epoch {epoch} ***************')
            losses = AverageMeter("loss")
            for batch_idx, data in enumerate(self.train_loader):
                imgs, labels = data['imgs'], data['label']
                imgs = Variable(imgs).to(self.cfgs.device)
                labels = Variable(labels).to(self.cfgs.device)
                outputs = model(imgs) 
                loss = self.loss_fn(outputs, labels.squeeze(1))       
                optimizer.zero_grad()
                loss.backward()
                if hasattr(self.cfgs, 'clip_gradient') and self.cfgs.clip_gradient is not None:
                    _ = clip_grad_norm_(model.parameters(), self.cfgs.clip_gradient)
                optimizer.step()
                losses.update(loss.item(), labels.size(0))
                if (batch_idx + 1) % 50 == 0:
                    print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, 
                                    len(self.train_loader), losses.val, losses.avg))

