import torch
import numpy as np
from functools import partial, reduce

def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()

def subtract_(target, minuend, subtrachend):
    with torch.no_grad():
        for name in target:
            target[name].data = minuend[name].data.clone()-subtrachend[name].data.clone()

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
                r = r.cuda()
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