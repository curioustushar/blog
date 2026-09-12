"""Width-scalable MLP for optimizer experiments."""

from __future__ import annotations

import torch
from torch import nn


class WidthMLP(nn.Module):
    def __init__(self, input_dim: int, width: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, width)
        self.fc2 = nn.Linear(width, output_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def layer_names(self) -> list[str]:
        return ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
