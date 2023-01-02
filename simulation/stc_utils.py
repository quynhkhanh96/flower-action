import torch

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
        target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()

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

def compress(target, source, compress_fun):
    for name in target:
        target[name].data = compress_fun(source[name].data.clone())

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
