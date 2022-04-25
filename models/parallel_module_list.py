from typing import List, Iterable, Optional

import torch


class ParallelModuleList(torch.nn.ModuleList):

    def __init__(self, modules: Optional[Iterable[torch.nn.Module]] = None):
        super(ParallelModuleList, self).__init__(modules)

    # noinspection PyMethodOverriding
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for stream_id, module in enumerate(self):
            x[stream_id] = module(x[stream_id])
        return x


class ForkModuleList(torch.nn.ModuleList):

    def __init__(self, modules: Optional[Iterable[torch.nn.Module]] = None):
        super(ForkModuleList, self).__init__(modules)

    # noinspection PyMethodOverriding
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        xs = []
        for stream_id, module in enumerate(self):
            xs[stream_id] = module(x)
        return xs
