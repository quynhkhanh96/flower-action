import sys, os
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
import copy
import operator
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def model_convert(model_weights):
    ini = []
    if isinstance(model_weights, list):
        for weight in model_weights:
            # ini = ini + torch.flatten(weight).tolist()
            ini = ini + weight.flatten().tolist()
    else:
        for layer in model_weights.keys():
            ini = ini + torch.flatten(model_weights[layer]).tolist()
    return ini

def dot_sum(K, L):
    return round(sum(i[0] * i[1] for i in zip(K, L)), 2)

def get_average(grad_all):
    value_list = list(grad_all.values())
    w_avg = copy.deepcopy(value_list[0])
    for i in range(1, len(value_list)):
        w_avg += value_list[i]
    return w_avg / len(value_list)

def node_deleting(expect_list, expect_value, worker_ind, grads):
    for i in range(len(worker_ind)):
        worker_ind_del  = [n for n in worker_ind if n != worker_ind[i]]    
        grad_del = grads.copy()
        grad_del.pop(worker_ind[i])
        avg_grad_del = get_average(grad_del)
        grad_del['avg_grad'] = avg_grad_del
        expect_value_del = get_relation(grad_del, worker_ind_del)
        expect_list[worker_ind[i]] = expect_value_del
    expect_list['all'] =  expect_value

    return expect_list

def Diff(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))

# def weight_average(w, idxs_users):
#     w_avg = w[idxs_users[0]]
#     for k in w_avg.keys():
#         for i in range(1, len(idxs_users)):    
#             w_avg[k] += w[idxs_users[i]][k]
#         w_avg[k] = torch.div(w_avg[k], len(idxs_users))
#     return w_avg
def weight_average(w, idxs_users):
    w_avg = w[idxs_users[0]]
    for k in range(len(w_avg)):
        for i in range(1, len(idxs_users)):
            w_avg[k] += w[idxs_users[i]][k]
        w_avg[k] = w_avg[k] / len(idxs_users)
    return w_avg 

def convert_numpy_weights(net, np_weights):
    state_dict = OrderedDict()
    for i, layer in enumerate(net.state_dict()):
        state_dict[layer] = torch.Tensor(np_weights[i])
    return state_dict

def get_gradient(pre, now, cfgs, num_samples):
    grad = np.subtract(model_convert(pre), model_convert(now))
    return grad / (num_samples * cfgs.local_e * cfgs.lr / cfgs.train_bz)

def get_relation(avg_grad, idxs_users):
     innnr_value = {}
     for i in range(len(idxs_users)):
         innnr_value[idxs_users[i]] = dot_sum(avg_grad[idxs_users[i]], avg_grad['avg_grad'])
     return round(sum(list(innnr_value.values())), 3)

def probabilistic_selection(node_prob, node_count, act_indx, 
                            part_node_after, labeled, num_selected):
    remove_list = Diff(act_indx, part_node_after)
    
    for i in remove_list:
        node_count[i][2] += 1

    # rest_nodes = Diff(list(node_prob.keys()), remove_list)
    rest_nodes = Diff(list(node_prob.keys()), labeled)
    alpha, beta = 2.0, 0.7
    weight = 0
 
    ratio = {}
    for i in labeled:
        ratio[i] = node_count[i][1] / node_count[i][0]
        
    for i in labeled:
        prob_change =  node_prob[i] * min((ratio[i] + beta)**alpha, 1)
        weight += prob_change
        node_prob[i] =  node_prob[i] - prob_change
 
    for i in rest_nodes:
            node_prob[i] = node_prob[i] + weight / (len(rest_nodes))

    get_node = np.random.choice(list(node_prob.keys()), num_selected, 
                            replace=False, p=list(node_prob.values()))

    return get_node.tolist(), node_prob, node_count