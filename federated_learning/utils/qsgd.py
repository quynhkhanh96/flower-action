import torch
from math import log
import numpy as np
import struct
from functools import partial, reduce

def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()

def subtract_(target, minuend, subtrachend):
    with torch.no_grad():
        for name in target:
            target[name].data = minuend[name].data.clone()-subtrachend[name].data.clone()

def average(target, sources):
    with torch.no_grad():
        for name in target:
            target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()

def weighted_average(target, sources, weights):
    with torch.no_grad():
        for name in target:
            summ = torch.sum(weights)
            n = len(sources)
            modify = [weight/summ*n for weight in weights]
            target[name].data = torch.mean(torch.stack([m*source[name].data for source, m in zip(sources, modify)]), dim=0).clone()

unary = lambda n: n * '0' + '1'
binary = lambda n, l = 1:("{0:0%db}" % l).format(n)

def elias(n):
    if not n: return '0'

    x = int(log(n, 2))
    return unary(x) + binary(n - 2**x, x)

def unelias(s, x):
    '''
        x: number of leading zeros
    '''
    return 2**x + int(s[x+1:], 2)

def bitstring_to_bytes(s):
    # Credit: https://stackoverflow.com/questions/32675679/convert-binary-string-to-bytearray-in-python-3
    b = bytearray()
    for i in range(0, len(s), 8):
        b.append(int(s[i:i+8], 2) & 0xff)
    r = len(s) % 8
    b.append(r & 0xff)
    return bytes(b)

def bytes_to_bitstring(bs):
    res = ''
    for i in range(len(bs)-2):
        res += str(bin(bs[i]))[2:].zfill(8)
    r = bs[-1]
    if r != 0:
        res += str(bin(bs[-2]))[2:].zfill(r)
    return res

class QSGDCoder:
    def __init__(self, s):
        self.s = s

    def encode(self, norm, signs, epsilon):
        inds = np.where(epsilon > 0)[0]

        msg = ''
        # Iteratively encode each non-zero element with Elias code
        msg += elias(inds[0]+2)
        if signs[inds[0]] == 1: msg += '1'
        else: msg += '0'
        msg += elias(epsilon[inds[0]]+2)

        for i in range(1, len(inds)):
            dist = elias(inds[i] - inds[i-1]+2)
            sign = '1' if signs[inds[i]] == 1 else '0'
            eps = elias(epsilon[inds[i]]+2)
            msg += (dist + sign + eps)

        # The first 32 bits (4 bytes) are for norm
        return struct.pack('!f', norm) + bitstring_to_bytes(msg)

    def decode(self, msg, d):
        # norm back to float32
        norm = struct.unpack('!f', msg[:4])[0]

        # nonzero information back to bit string
        nonzeros = bytes_to_bitstring(msg[4:])

        # indices, values => signs, epsilon
        indices, values = [0], []
        i, j, N = 0, 0, len(nonzeros)
        track = 0 # track = 0 means we are processing an index, 1 means a value
        while i < N:
            if nonzeros[i] == '0':
                i += 1
            else:
                x = i-j
                if not track:
                    ind = indices[-1] + unelias(nonzeros[j:i+x+1], x)-2
                    indices.append(ind)
                    i += (x+2)
                    j = i
                    # sign bit at nonzeros[i-1]
                    sign = 1 if nonzeros[i-1] == '1' else -1
                else:
                    val = unelias(nonzeros[j:i+x+1], x)-2
                    values.append(val * sign)
                    i += (x+1)
                    j = i
                track = not track

        _ = indices.pop(0)

        v = np.zeros(d)
        for i, ind in enumerate(indices):
            v[ind] = norm * values[i] / self.s

        return v

# https://github.com/xinyandai/gradient-quantization/blob/master/compressors/qsgd_compressor.py
class QSGDQuantizer:
    def __init__(self, random, n_bit, no_cuda):
        '''
            random: True if it is stochastic quantization, False otherwise
            n_bit: number of bits that we want to quantize to
            no_cuda: True if we're using GPUs, False otherwise

        '''
        self.random = random
        self.bit = n_bit
        assert self.bit > 0

        self.cuda = not no_cuda
        self.s = 2 ** self.bit

        self.code_dtype = torch.int32

    def quantize(self, vec):
        """
        :param vec: torch tensor
        :return: norm, signs, quantized_intervals
        """
        w_shape = tuple(vec.shape)
        w_dim = reduce(lambda x, y: x*y, w_shape)
        vec = vec.view(-1, w_dim)
        norm = torch.max(torch.abs(vec), dim=1, keepdim=True)[0]
        normalized_vec = vec / norm

        scaled_vec = torch.abs(normalized_vec) * self.s
        l = torch.clamp(scaled_vec, 0, self.s-1).type(self.code_dtype)

        if self.random:
            # l[i] <- l[i] + 1 with probability |v_i| / ||v|| * s - l
            probabilities = scaled_vec - l.type(torch.float32)
            r = torch.rand(l.size())
            if self.cuda:
                r = r.to(vec.device)
            l[:] += (probabilities > r).type(self.code_dtype)

        signs = torch.sign(vec) > 0
        return [norm, signs.view(w_shape), l.view(w_shape)]

    def dequantize(self, signature):
        [norm, signs, l] = signature
        assert l.shape == signs.shape
        w_shape = tuple(l.shape)
        w_dim = reduce(lambda x, y: x*y, w_shape)
        scaled_vec = l.type(torch.float32) * (2 * signs.type(torch.float32) - 1)
        compressed = (scaled_vec.view((-1, w_dim))) * norm / self.s
        return compressed.view(w_shape)