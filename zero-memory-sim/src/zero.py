"""Tensor-level ZeRO 1/2/3 rank simulation.

This module builds one flat parameter vector for the model and then constructs
per-rank *shard views* of parameters, gradients and optimizer states according
to the ZeRO stage. The shards are real ``torch.Tensor`` objects — so
``tensor.element_size() * tensor.numel()`` gives us honest per-rank bytes we
can compare against the analytical formulas in ``memory.py``.

No real cross-process communication happens. Instead we compute the *volume*
of data that would flow over the network per step (in bytes) using the
standard ZeRO/ring-collective accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Callable

import torch

from .config import ModelConfig, WorldConfig


@dataclass
class RankState:
    rank: int
    param_shard: torch.Tensor  # what this rank stores for parameters
    grad_shard: torch.Tensor   # what this rank stores for gradients
    optim_shard: torch.Tensor  # what this rank stores for optimizer state (m, v concatenated)

    def bytes(self) -> dict[str, int]:
        return {
            "param": self.param_shard.element_size() * self.param_shard.numel(),
            "grad": self.grad_shard.element_size() * self.grad_shard.numel(),
            "optim": self.optim_shard.element_size() * self.optim_shard.numel(),
        }


def _make_shard_slices(P: int, world_size: int) -> list[tuple[int, int]]:
    shard = ceil(P / world_size)
    return [(r * shard, min((r + 1) * shard, P)) for r in range(world_size)]


def _build_ranks(
    P: int,
    mcfg: ModelConfig,
    wcfg: WorldConfig,
    param_shape: Callable[[int, int], int],
    grad_shape: Callable[[int, int], int],
    optim_shape: Callable[[int, int], int],
) -> list[RankState]:
    """Build the per-rank tensors.

    Each ``*_shape`` callable takes ``(rank, world_size)`` and returns
    how many elements this rank should hold for that category. Optimizer
    state counts m and v separately, so it returns ``2 * <param_slice_size>``
    (or 0 if partitioned differently, but here every strategy shards the
    optimizer state).
    """
    dtype = torch.float32  # fp32 matches ModelConfig.dtype_bytes=4
    ranks: list[RankState] = []
    for r in range(wcfg.world_size):
        p_n = param_shape(r, wcfg.world_size)
        g_n = grad_shape(r, wcfg.world_size)
        o_n = optim_shape(r, wcfg.world_size)
        ranks.append(
            RankState(
                rank=r,
                param_shard=torch.zeros(p_n, dtype=dtype),
                grad_shard=torch.zeros(g_n, dtype=dtype),
                optim_shard=torch.zeros(o_n, dtype=dtype),
            )
        )
    return ranks


def data_parallel_ranks(P: int, mcfg: ModelConfig, wcfg: WorldConfig) -> list[RankState]:
    return _build_ranks(
        P,
        mcfg,
        wcfg,
        param_shape=lambda r, N: P,
        grad_shape=lambda r, N: P,
        optim_shape=lambda r, N: wcfg.optimizer_states_per_param * P,
    )


def zero1_ranks(P: int, mcfg: ModelConfig, wcfg: WorldConfig) -> list[RankState]:
    slices = _make_shard_slices(P, wcfg.world_size)
    opt_slices = _make_shard_slices(
        wcfg.optimizer_states_per_param * P, wcfg.world_size
    )
    return _build_ranks(
        P,
        mcfg,
        wcfg,
        param_shape=lambda r, N: P,
        grad_shape=lambda r, N: P,
        optim_shape=lambda r, N: opt_slices[r][1] - opt_slices[r][0],
    )


def zero2_ranks(P: int, mcfg: ModelConfig, wcfg: WorldConfig) -> list[RankState]:
    slices = _make_shard_slices(P, wcfg.world_size)
    opt_slices = _make_shard_slices(
        wcfg.optimizer_states_per_param * P, wcfg.world_size
    )
    return _build_ranks(
        P,
        mcfg,
        wcfg,
        param_shape=lambda r, N: P,
        grad_shape=lambda r, N: slices[r][1] - slices[r][0],
        optim_shape=lambda r, N: opt_slices[r][1] - opt_slices[r][0],
    )


def zero3_ranks(P: int, mcfg: ModelConfig, wcfg: WorldConfig) -> list[RankState]:
    slices = _make_shard_slices(P, wcfg.world_size)
    opt_slices = _make_shard_slices(
        wcfg.optimizer_states_per_param * P, wcfg.world_size
    )
    return _build_ranks(
        P,
        mcfg,
        wcfg,
        param_shape=lambda r, N: slices[r][1] - slices[r][0],
        grad_shape=lambda r, N: slices[r][1] - slices[r][0],
        optim_shape=lambda r, N: opt_slices[r][1] - opt_slices[r][0],
    )


BUILDERS = {
    "data_parallel": data_parallel_ranks,
    "zero_1": zero1_ranks,
    "zero_2": zero2_ranks,
    "zero_3": zero3_ranks,
}


def summarize_ranks(ranks: list[RankState]) -> dict[str, int]:
    per_rank = ranks[0].bytes()
    return {
        "param_bytes_rank0": per_rank["param"],
        "grad_bytes_rank0": per_rank["grad"],
        "optim_bytes_rank0": per_rank["optim"],
        "total_bytes_rank0": sum(per_rank.values()),
        "total_bytes_across_ranks": sum(sum(r.bytes().values()) for r in ranks),
    }
