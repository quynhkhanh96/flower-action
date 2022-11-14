import sys, os
sys.path.insert(0, os.path.abspath('..'))
import torch
import copy
import operator
import torch.nn.functional as F
from torch.autograd import Variable
from base_server import Server
from fedpns_utils import (
    get_gradient, get_relation, get_average, 
    node_deleting, weight_average, convert_numpy_weights
)

class FedPNSServer(Server):
    def __init__(self, v, **kwargs):
        super(FedPNSServer, self).__init__(**kwargs)     
        self.v = int(v * self.cfgs.num_C * self.cfgs.frac) + 1
        print(f'For num_C = {self.num_clients}, frac = {self.cfgs.frac}: v = {self.v}.')
        self.model.eval()

    def get_test_loss(self):
        self.model.to(self.device)
        self.model.eval()

        test_loss = 0
        for data in self.test_loader:
            if isinstance(data, dict):
                imgs, label = data['imgs'], data['label']
            else:
                imgs, label, _ = data

            imgs = imgs.to(self.device)
            with torch.no_grad():
                log_probs = self.model(Variable(imgs))
                test_loss += F.cross_entropy(log_probs, label,
                                reduction='sum').item()

        test_loss /= len(self.test_loader.dataset)
        return test_loss       

    def test_part(self, w_locals, idxs_users, key):
        weight_all = weight_average(w_locals, idxs_users)
        net_all = convert_numpy_weights(self.model, weight_all)
        self.model.load_state_dict(net_all)
        loss_all = self.get_test_loss()

        idxs_users.remove(key)
        weight_part = weight_average(w_locals, idxs_users)
        net_part = convert_numpy_weights(self.model, weight_part)
        self.model.load_state_dict(net_part)
        loss_part = self.get_test_loss()

        return loss_all, loss_part, idxs_users

    def aggregate(self, w_locals, idxs_users, num_examples, test_count):
        full_users = copy.copy(idxs_users)

        gradient = {}
        for idx, client_weight in w_locals.items():
            g = get_gradient(self.model.state_dict(), client_weight, 
                            self.cfgs, num_examples[idx])
            gradient[idx] = copy.deepcopy(g)

        gradient['avg_grad'] = get_average(gradient)
        max_now = get_relation(gradient, idxs_users)
        gradient.pop('avg_grad')
        expect_list = {}
        labeled = []
        while len(w_locals) > self.v:
            expect_list = node_deleting(expect_list, max_now, idxs_users, gradient)
            key = max(expect_list.items(), key=operator.itemgetter(1))[0]
            if expect_list[key] <= expect_list["all"]:
                break
            else:
                labeled.append(key)
                test_count[key][1] += 1
                expect_list.pop("all")
                loss_all, loss_pop, idxs_users = self.test_part(w_locals, idxs_users, key)
                if loss_all < loss_pop:
                    # w_locals, idxs_users.append(key)
                    idxs_users.append(key)
                    break
                else:
                    w_locals.pop(key)
                    gradient.pop(key)
                    max_now = expect_list[key]
                    expect_list.pop(key)
        
        msg = 'After filtering adverse clients, take {} clients.'.format(len(idxs_users))
        print(msg)
        with open(self.work_dir + f'/node_C{self.num_clients}_frac{self.cfgs.frac}_logs.txt', 'a') as f:
            f.write(msg + '\n')
        
        w_glob = weight_average(w_locals, idxs_users)
        return w_glob, full_users, idxs_users, labeled, test_count