import torch
from collections import OrderedDict
from base_client import Client
from federated_learning.client.update.video_base import VideoLocalUpdate

class FedBnClient(Client):

    def load_weights(self, global_model, client_id):
        try:
            last_local_weights = torch.load(
                self.work_dir + f'/client_{client_id}.pth')
            self.model.load(last_local_weights['state_dict'])
            state_dict = OrderedDict()
            global_weights = global_model.state_dict()
            for k in global_weights:
                if 'bn' not in k:
                    state_dict[k] = global_weights[k]
            self.model.load_state_dict(state_dict, strict=False)
        except:
            self.model = global_model

    def train(self, rnd, client_id, global_model):
        # load the new global weights
        self.load_weights(global_model, client_id)

        # create the data loaders of the current client
        train_loader, val_loader = self.get_data_loaders(client_id)
        local_trainer = VideoLocalUpdate(train_loader=train_loader,
                                        loss_fn=self.loss_fn, 
                                        cfgs=self.cfgs)
        # train loop
        local_trainer.train(self.model, client_id)

        # save current local weights
        torch.save({'state_dict': self.model.state_dict()},
            self.work_dir + f'/client_{client_id}.pth'
        )
        
        return self.get_weights(), len(train_loader.dataset)
