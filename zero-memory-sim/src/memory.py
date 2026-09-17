"""Per-rank model-state memory accounting for DP / ZeRO-1 / ZeRO-2 / ZeRO-3.

Every function returns bytes. The formulas are derived in the notebook and
README; here we implement them once so the notebook, sanity checks, and
plots all read from the same source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .config import ModelConfig, WorldConfig


@dataclass(frozen=True)
class RankMemory:
    strategy: str
    param_bytes: int
    grad_bytes: int
    optimizer_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.param_bytes + self.grad_bytes + self.optimizer_bytes


def _shard(total: int, world_size: int) -> int:
    """Ceil-shard so that world_size shards cover the whole tensor."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    return ceil(total / world_size)


def data_parallel(P: int, mcfg: ModelConfig, wcfg: WorldConfig) -> RankMemory:
    """Every rank stores full params, grads and optimizer states."""
    return RankMemory(
        strategy="data_parallel",
        param_bytes=P * mcfg.dtype_bytes,
        grad_bytes=P * mcfg.dtype_bytes,
        optimizer_bytes=wcfg.optimizer_states_per_param * P * mcfg.dtype_bytes,
    )


def zero_1(P: int, mcfg: ModelConfig, wcfg: WorldConfig) -> RankMemory:
    """Params and grads replicated; optimizer states partitioned across ranks."""
    return RankMemory(
        strategy="zero_1",
        param_bytes=P * mcfg.dtype_bytes,
        grad_bytes=P * mcfg.dtype_bytes,
        optimizer_bytes=_shard(
            wcfg.optimizer_states_per_param * P, wcfg.world_size
        ) * mcfg.dtype_bytes,
    )


def zero_2(P: int, mcfg: ModelConfig, wcfg: WorldConfig) -> RankMemory:
    """Params replicated; grads and optimizer states partitioned."""
    return RankMemory(
        strategy="zero_2",
        param_bytes=P * mcfg.dtype_bytes,
        grad_bytes=_shard(P, wcfg.world_size) * mcfg.dtype_bytes,
        optimizer_bytes=_shard(
            wcfg.optimizer_states_per_param * P, wcfg.world_size
        ) * mcfg.dtype_bytes,
    )


def zero_3(P: int, mcfg: ModelConfig, wcfg: WorldConfig) -> RankMemory:
    """Params, grads and optimizer states all partitioned across ranks."""
    return RankMemory(
        strategy="zero_3",
        param_bytes=_shard(P, wcfg.world_size) * mcfg.dtype_bytes,
        grad_bytes=_shard(P, wcfg.world_size) * mcfg.dtype_bytes,
        optimizer_bytes=_shard(
            wcfg.optimizer_states_per_param * P, wcfg.world_size
        ) * mcfg.dtype_bytes,
    )


STRATEGIES = {
    "data_parallel": data_parallel,
    "zero_1": zero_1,
    "zero_2": zero_2,
    "zero_3": zero_3,
}


def all_strategies(P: int, mcfg: ModelConfig, wcfg: WorldConfig) -> dict[str, RankMemory]:
    return {name: fn(P, mcfg, wcfg) for name, fn in STRATEGIES.items()}
