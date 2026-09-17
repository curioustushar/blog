"""Configurable deep MLP used as the demo model."""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class DeepMLP(nn.Module):
    """Linear -> [ReLU -> Linear] * depth -> output.

    Parameter count is dominated by the ``depth`` hidden linear layers, so
    per-rank memory comparisons are easy to reason about analytically.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(cfg.input_dim, cfg.hidden_dim), nn.ReLU()]
        for _ in range(max(0, cfg.depth - 1)):
            layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(cfg.hidden_dim, cfg.output_dim))
        self.net = nn.Sequential(*layers)
        self._cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_bytes(self) -> int:
        return self.param_count() * self._cfg.dtype_bytes
