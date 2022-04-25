from typing import List

import torch
import torch.nn as nn

__all__ = ['SlowFastLateralConnection']


class SlowFastLateralConnection(nn.Module):

    def __init__(self,
                 slow_dim: int,
                 fast_dim: int,
                 fusion_kernel: int = 5,
                 alpha: int = 4,
                 order: str = 'slow_fast'
                 ):
        super(SlowFastLateralConnection, self).__init__()
        if order == 'slow_fast':
            self.reversed = False
        elif order == 'fast_slow':
            self.reversed = True
        else:
            raise ValueError(f'order must be slow_fast or fast_slow; got {order}.')

        self.conv_f2s = nn.Conv3d(
            fast_dim,
            slow_dim,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = nn.BatchNorm3d(
            num_features=slow_dim,
            eps=1e-5,
            momentum=0.1,
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn_out = nn.BatchNorm3d(
            num_features=slow_dim,
            eps=1e-5,
            momentum=0.1,
        )

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        if not self.reversed:
            x_slow, x_fast = xs
        else:
            x_fast, x_slow = xs
        fuse = self.conv_f2s(x_fast)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        # x_slow_fuse = torch.cat([x_slow, fuse], dim=1)
        x_slow_fuse = self.bn_out(x_slow + fuse)
        return [x_slow_fuse, x_fast]
