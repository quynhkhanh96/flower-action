from typing import List

import torch
import torch.nn as nn

__all__ = ['ConcatenateFusionBlock']


class ConcatenateFusionBlock(nn.Module):

    def __init__(self,
                 num_streams: int,
                 dim: int = 1,
                 normalize=False):
        super(ConcatenateFusionBlock, self).__init__()
        self.num_streams = num_streams
        self.dim = dim
        self.normalize = normalize

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        for stream_id in range(self.num_streams):
            if self.normalize:
                xs[stream_id] = xs[stream_id] / xs[stream_id].norm(dim=self.dim, keepdim=True)
        return torch.cat(xs, dim=self.dim)
