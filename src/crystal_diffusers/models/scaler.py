from typing import Sequence

import torch
from torch import nn


class StandardScaler(nn.Module):
    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
    ):
        super().__init__()
        mean = torch.tensor(mean, dtype=torch.float32)

        eps = torch.finfo(mean.dtype).eps
        std = torch.tensor(std, dtype=torch.float32).clamp(min=eps)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std
