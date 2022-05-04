import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import AverageMeter

class VideoLocalUpdate:
    def __init__(self, train_loader, loss_fn, cfgs):
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.cfgs = cfgs 

    def train(self, model):
        model.to(self.cfgs.device)
        model.train()

        # Rebuild optimizer everytime client receives new model weights from server
        if self.cfgs.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=self.cfgs.lr, momentum=0.9)
        else:
            raise ValueError(f'No implementation for {self.cfgs.optimizer} optimizer.')

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
            if (batch_idx + 1) % self.cfgs.print_freq == 0:
                print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, 
                                    len(self.train_loader), losses.val, losses.avg))
