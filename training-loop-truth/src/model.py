"""Small char-level language model for transparent training experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import TrainConfig


class TinyCharLM(nn.Module):
    """Embedding + 2-layer MLP head — small enough to inspect every tensor."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.fc1 = nn.Linear(config.embed_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.vocab_size, bias=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T]
        x = self.embed(tokens)  # [B, T, D]
        x = torch.relu(self.fc1(x))  # [B, T, H]
        logits = self.fc2(x)  # [B, T, V]
        return logits

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
