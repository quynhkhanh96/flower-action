import os
import torch
import torch.optim as optim
import numpy as np

from base_client import Client
import qsgd_utils

class QSGDClient(Client):
    def __init__(self, random, n_bit, lower_bit, no_cuda, q_down, fp_layers, **kwargs):
        super().__init__(**kwargs)

        self.quantizer = qsgd_utils.QSGDQuantizer(
            random, n_bit, no_cuda
        )

        self.q_down = q_down
        self.lower_bit = lower_bit if lower_bit != -1 else n_bit
        self.fp_layers = fp_layers.split(',')
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.W_old = {name: torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(self.cfgs.device) 
                        for name, value in self.W.items()}
    
    def load_weights(self, global_model):
        if self.q_down:
            weights = {}
            for lname, signature in global_model.items():
                weights[lname] = self.quantizer.dequantize(signature)
            self.model.load_state_dict(weights, strict=False)
        else:
            self.model.load_state_dict(global_model.state_dict())
        self.model.to(self.cfgs.device)
        self.W = {name: value for name, value in self.model.named_parameters()}

    def _keep_layer_full_precision(self, lname):
        for fp_layer in self.fp_layers:
            if fp_layer in lname:
                return True
        return False

    def compress_weight_update_up(self):
        res = {}
        s = self.quantizer.s
        for lname, lgrad in self.dW.items():
            # if 'bn' in lname:
            if self._keep_layer_full_precision(lname):
                res[lname] = [lgrad]
                continue

            if 'conv' in lname and 'bn' not in lname:
                self.quantizer.s = s
            else:
                self.quantizer.s = 2 ** self.lower_bit
            res[lname] = self.quantizer.quantize(lgrad)

        self.quantizer.s = s
        
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

class TopKQSGDClient(QSGDClient):
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def compress_weight_update_up(self):
        res = {}
        s = self.quantizer.s
        for lname, lgrad in self.dW.items():
            if self._keep_layer_full_precision(lname):
                res[lname] = [lgrad]
                continue

            if 'conv' in lname and 'bn' not in lname:
                self.quantizer.s = s
                n_elts = lgrad.numel()
                n_top = int(np.ceil(n_elts * self.k))
                with torch.no_grad():
                    inds = torch.argsort(torch.abs(lgrad).flatten(), descending=True)
                    
                    # compress and encode the top-k gradients with higher #bits
                    mask_top = torch.full((lgrad.numel(),), 0.).to(lgrad.device).scatter_(0,
                                        inds[:n_top], 1).view(*tuple(lgrad.shape))
                    signature_top = self.quantizer.quantize(lgrad * mask_top)

                    # and the rest with less #bits
                    self.quantizer.s = 2 ** self.lower_bit
                    mask_rest = torch.full((lgrad.numel(),), 0.).to(lgrad.device).scatter_(0,
                                        inds[n_top:], 1).view(*tuple(lgrad.shape))
                    signature_rest = self.quantizer.quantize(lgrad * mask_rest)

                res[lname] = [signature_top, signature_rest]
            
            else:
                self.quantizer.s = 2 ** self.lower_bit
                res[lname] = self.quantizer.quantize(lgrad)

        self.quantizer.s = s
        
        return res