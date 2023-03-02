import torch
import numpy as np 
from functools import reduce

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

@dataclass
class STCFitRes:
    """Fit response from a STC client."""
    msgs: List[bytes]
    signs: List[bytes]
    mus = List[bytes]
    num_examples: int
    num_examples_ceil: Optional[int] = None  # Deprecated
    fit_duration: Optional[float] = None  # Deprecated

def dec_to_base(num, base):  
    # Maximum base - 36
    base_num = ""
    while num > 0:
        dig = int(num % base)
        if dig < 10:
            base_num += str(dig)
        else:
            base_num += chr(ord('A') + dig - 10)  # Using uppercase letters
        num //= base
    base_num = base_num[::-1]  # To reverse the string
    if len(base_num) < base:
        base_num = base_num.zfill(base)
    return base_num

def base_to_dec(num_str, base):
    num_str = num_str[::-1]
    num = 0
    for k in range(len(num_str)):
        dig = num_str[k]
        if dig.isdigit():
            dig = int(dig)
        else:    
            # Assuming its either number or alphabet only
            dig = ord(dig.upper()) - ord('A') + 10
        num += dig * (base**k)
    return num

def encode_index_distance(q, r, b_star):
    q_part = '1' * q + '0'
    r_part = dec_to_base(r, b_star)
    return q_part + r_part
    

def golomb_position_encode(T, p):
    '''
        T: sparse tensor
        p: sparsity factor, p << 1.0, for eg. p = 0.01
    '''
    # indices as 1-indexed
    T_flat = T.flatten()
    indices = [idx.item()+1 for idx in torch.nonzero(T_flat)] 
    golden = (np.sqrt(5)+1)/2
    b_star = int(1 + np.floor(np.log2(np.log(golden-1)/np.log(1-p))))
    if b_star == 0:
        raise ValueError('b_star must be greater than 0., consider lower the value of p.')
    
    msg = ''
    N = len(indices)
    if N == 0:
        mu = 0.
    else:
        mu = abs(T_flat[indices[0]-1])
    
    indices = [0] + indices
    signs = ''
    for i in range(1, N+1):
        d = indices[i] - indices[i-1]
        q, r = divmod(d-1, 2**b_star)
        msg += encode_index_distance(q, r, b_star)
        signs += str(int(T_flat[indices[i]-1].item() > 0))
    
    return msg, signs, mu

def golomb_position_decode(msg, mu, signs, p, T_shape):
    golden = (np.sqrt(5)+1)/2
    b_star = int(1 + np.floor(np.log2(np.log(golden-1)/np.log(1-p))))
    if b_star == 0:
        raise ValueError('b_star must be greater than 0., consider lower the value of p.')
    
    i, q, j = 0, 0, 0
    indices = []
    while i < len(msg):
        if msg[i] == '0':
            r = base_to_dec(msg[i+1: i+b_star+1], b_star)
            d = q * (2**b_star) + r + 1
            j += d
            indices.append(j-1)
            q = 0
            i = i + b_star + 1
        else:
            q += 1
            i += 1
    
    T = torch.zeros(reduce(lambda r, c: r*c, T_shape))
    for i, idx in enumerate(indices):
        if signs[i] == '1':
            T[idx] = mu
        else:
            T[idx] = mu*(-1)
    
    return T.reshape(T_shape)