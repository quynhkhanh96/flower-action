import sys, os
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
import torch
import copy
import operator
import torch.nn.functional as F
from torch.utils.data import DataLoader

from base_server import Server

def model_convert(model_weights):
    ini = []
    if isinstance(model_weights, list):
        for weight in model_weights:
            ini = ini + torch.flatten(weight).tolist()
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

def test_img(net_g, datatest, args, train_sampler):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs, sampler=train_sampler)
    # print(data_loader)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
    
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = correct.item() / 100

    return accuracy, test_loss

def weight_average(w, idxs_users):
    w_avg = w[idxs_users[0]]
    for k in w_avg.keys():
        for i in range(1, len(idxs_users)):    
            w_avg[k] += w[idxs_users[i]][k]
        w_avg[k] = torch.div(w_avg[k], len(idxs_users))
    return w_avg

def test_part(net_glob, w_locals, idxs_users, 
                key, dataset_test_part, test_sampler, args):
    net_all = weight_average(w_locals, idxs_users)
    
    net_glob.load_state_dict(net_all)
    acc, loss_all = test_img(net_glob, dataset_test_part, args, test_sampler)
    idxs_users.remove(key)
    net_part = weight_average(w_locals, idxs_users)
    net_glob.load_state_dict(net_part)
    acc, loss_part = test_img(net_glob, dataset_test_part, args, test_sampler)
    
    return loss_all, loss_part, idxs_users


# ========================== get_gradient() ==========================
'''
    args.num_sample = ?
    args.local_ep, args.local_bs
'''
# def get_gradient(args, pre, now, lr):
#     grad = np.subtract(model_convert(pre), model_convert(now)) 
#     return grad / (args.num_sample * args.local_ep * lr / args.local_bs)
def get_gradient(pre, now, cfgs, num_samples):
    grad = np.subtract(model_convert(pre), model_convert(now))
    return grad / (num_samples * cfgs.local_e * cfgs.lr / cfgs.train_bz)
# ========================== get_relation() ==========================
def get_relation(avg_grad, idxs_users):
    innnr_value = {}
    for i in range(len(idxs_users)):
        innnr_value[idxs_users[i]] = dot_sum(avg_grad[idxs_users[i]], avg_grad['avg_grad'])
    return round(sum(list(innnr_value.values())), 3)

# ========================== Feddel() ==========================
def fed_del(net_glob, w_locals, gradient, idxs_users, 
            max_now, dataset_test, test_sampler, args, test_count):
    full_user = copy.copy(idxs_users)
    # nr_th = len(idxs_users) * 0.7
    gradient.pop('avg_grad')
    expect_list = {}
    labeled = []
    while len(w_locals) > 8:
        expect_list = node_deleting(expect_list, max_now, idxs_users, gradient)
        key = max(expect_list.items(), key=operator.itemgetter(1))[0]
        if expect_list[key] <= expect_list["all"]:
            break
        else:
            labeled.append(key)
            test_count[key][1] += 1
            expect_list.pop("all")
            loss_all, loss_pop, idxs_users = test_part(net_glob, w_locals, idxs_users, 
                                                key, dataset_test, test_sampler, args)
            if loss_all < loss_pop:
                w_locals, idxs_users.append(key)
                break
            else:
                w_locals.pop(key)
                gradient.pop(key)
                max_now = expect_list[key]
                expect_list.pop(key)

    return w_locals, full_user, idxs_users, labeled, test_count

# ========================== probabilistic_selection() ==========================
def probabilistic_selection(node_prob, node_count, act_indx, part_node_after, labeled):
    remove_list = Diff(act_indx, part_node_after)
    
    for i in remove_list:
        node_count[i][2] += 1

    # rest_nodes = Diff(list(node_prob.keys()), remove_list)
    rest_nodes = Diff(list(node_prob.keys()), labeled)
    alpha, beta = 2.0, 0.7
    weight = 0
 
    ratio = {}
    for i in labeled:
        ratio[i] = node_count[i][1]/ node_count[i][0]
        
    for i in labeled:
        prob_change =  node_prob[i] * min((ratio[i] + beta)**alpha, 1)
        weight += prob_change
        node_prob[i] =  node_prob[i] - prob_change
 
    for i in rest_nodes:
            node_prob[i] = node_prob[i] + weight / (len(rest_nodes))

    get_node = np.random.choice(list(node_prob.keys()), 10, replace=False, p=list(node_prob.values()))

    return get_node.tolist(), node_prob, node_count