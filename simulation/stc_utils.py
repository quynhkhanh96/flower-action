import torch
import numpy as np
from functools import partial

def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()

def add(target, source):
    for name in target:
        target[name].data += source[name].data.clone()

def scale(target, scaling):
    for name in target:
        target[name].data = scaling * target[name].data.clone()

def subtract(target, source):
    for name in target:
        target[name].data -= source[name].data.clone()

def subtract_(target, minuend, subtrachend):
    for name in target:
        target[name].data = minuend[name].data.clone()-subtrachend[name].data.clone()

def average(target, sources):
    for name in target:
        target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()

def weighted_average(target, sources, weights):
    for name in target:
        summ = torch.sum(weights)
        n = len(sources)
        modify = [weight/summ*n for weight in weights]
        target[name].data = torch.mean(torch.stack([m*source[name].data for source, m in zip(sources, modify)]), dim=0).clone()

def majority_vote(target, sources, lr):
    for name in target:
        # threshs = torch.stack([torch.max(source[name].data) for source in sources])
        mask = torch.stack([source[name].data.sign() for source in sources]).sum(dim=0).sign()
        target[name].data = (lr*mask).clone()

def approx_v(T, p, frac):
    if frac < 1.0:
        n_elements = T.numel()
        n_sample = min(int(max(np.ceil(n_elements * frac), np.ceil(100/p))), n_elements)
        n_top = int(np.ceil(n_sample*p))

        i = 0 if n_elements == n_sample else np.random.randint(n_elements-n_sample)
        topk, _ = torch.topk(T.flatten()[i:i+n_sample], n_top)
        if topk[-1] == 0.0 or topk[-1] == T.max():
            return approx_v(T, p, 1.0)
    else:
        n_elements = T.numel()
        n_top = int(np.ceil(n_elements * p))
        topk, _ = torch.topk(T.flatten(), n_top)
    
    return topk[-1], topk

def none(T, hp):
    return T

def dgc(T, hp):
    '''
        "Deep Gradient Compression: Reducing the communication Bandwidth for Distributed Training, Lin et al."
    '''
    hp_ = {'p' : 0.001, 'approx' : 1.0}
    hp_.update(hp)
    device = T.device

    if hp_['p'] >= 1.0:
        return T

    T_abs = torch.abs(T)
    v, _ = approx_v(T_abs, hp_['p'], hp_['approx'])
    out = torch.where(T_abs >= v, T, torch.Tensor([0.0]).to(device))

    return out

def stc(T, hp):
    '''
        "Sparse Binary Compression: Towards Distributed Deep Learning with minimal Communication, Sattler et al."
    '''
    hp_ = {'p' : 0.001, 'approx' : 1.0}
    hp_.update(hp)
    device = T.device

    T_abs = torch.abs(T)
    v, topk = approx_v(T_abs, hp_["p"], hp_["approx"])
    mean = torch.mean(topk) 

    out_ = torch.where(T >= v, mean, torch.Tensor([0.0]).to(device))
    out = torch.where(T <= -v, -mean, out_)

    return out

def signsgd(T, hp):
    '''
        signSGD: Compressed Optimisation for non-convex Problems, Bernstein et al.
    '''
    return T.sign()

def compress(target, source, compress_fun):
    for name in target:
        target[name].data = compress_fun(source[name].data.clone())

def compression_function(name, hp=None):
    return partial(globals()[name], hp=hp)

def get_hp_compression(compression):

    c = compression[0]
    hp = compression[1]

    if c ==  "none" : 
        return  {"compression_up" : ["none", {}], "compression_down" : ["none", {}],
               "accumulation_up" : False, "accumulation_down" : False,  "aggregation" : "mean"}

    if c ==  "signsgd" : 
        return  {"compression_up" : ["signsgd", {}], "compression_down" : ["none", {}],
               "accumulation_up" : False, "accumulation_down" : False,  "aggregation" : "majority", "lr" : hp["lr"], "local_iterations" : 1}
    
    if c ==  "dgc_up" : 
        return  {"compression_up" : ["dgc", {"p" : hp["p_up"]}], "compression_down" : ["none", {}],
               "accumulation_up" : True, "accumulation_down" : False,  "aggregation" : "mean"}
    
    if c ==  "stc_up" : 
        return  {"compression_up" : ["stc", {"p" : hp["p_up"]}], "compression_down" : ["none", {}],
               "accumulation_up" : True, "accumulation_down" : False,  "aggregation" : "mean"}
    
    if c ==  "dgc_updown" : 
        return  {"compression_up" : ["dgc", {"p" : hp["p_up"]}], "compression_down" : ["dgc", {"p" : hp["p_down"]}],
               "accumulation_up" : True, "accumulation_down" : True,  "aggregation" : "mean"}    
    if c ==  "stc_updown" : 
        return {"compression_up" : ["stc", {"p" : hp["p_up"]}], "compression_down" : ["stc", {"p" : hp["p_down"]}],
               "accumulation_up" : True, "accumulation_down" : True,  "aggregation" : "mean"}
    
    if c ==  "fed_avg" : 
        return {"compression_up" : ["none", {}], "compression_down" : ["none", {}],
               "accumulation_up" : False, "accumulation_down" : False,  "aggregation" : "weighted_mean", "local_iterations" : hp["n"]}
