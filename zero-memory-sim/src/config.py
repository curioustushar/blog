"""Configuration for the ZeRO memory simulator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """A small, easy-to-reason-about MLP so memory numbers are dominated by params."""

    input_dim: int = 512
    hidden_dim: int = 1024
    output_dim: int = 512
    depth: int = 4  # number of hidden layers
    dtype_bytes: int = 4  # fp32


@dataclass(frozen=True)
class WorldConfig:
    """One logical world = a set of `world_size` virtual ranks on a single process."""

    world_size: int = 32
    optimizer_states_per_param: int = 2  # Adam keeps m and v => 2 * P scalars
