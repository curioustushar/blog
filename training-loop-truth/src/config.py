"""Experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    vocab_size: int = 64
    embed_dim: int = 32
    hidden_dim: int = 64
    seq_len: int = 16
    batch_size: int = 8
    lr: float = 1e-2
    train_steps: int = 120
    seed: int = 42
    device: str = "cpu"
