"""Analytical communication volume per training step, per rank.

Using the standard ring-collective accounting for a world of ``N`` ranks:

* ``all-reduce(P)`` = ``2 * (N - 1) / N * P`` bytes moved per rank
  (reduce-scatter + all-gather). Under the common approximation of large N,
  this is ~``2P``.
* ``reduce-scatter(P)`` = ``(N - 1) / N * P`` bytes per rank (~``P``).
* ``all-gather(P)``    = ``(N - 1) / N * P`` bytes per rank (~``P``).

We report both the exact and the "~large N" approximations. Volumes are in
bytes, not elements.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ModelConfig, WorldConfig


@dataclass(frozen=True)
class CommsPerStep:
    strategy: str
    forward_bytes: int
    backward_bytes: int
    optimizer_step_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.forward_bytes + self.backward_bytes + self.optimizer_step_bytes


def _all_reduce_bytes(P_elems: int, world_size: int, dtype_bytes: int) -> int:
    if world_size <= 1:
        return 0
    return int(2 * (world_size - 1) / world_size * P_elems * dtype_bytes)


def _reduce_scatter_bytes(P_elems: int, world_size: int, dtype_bytes: int) -> int:
    if world_size <= 1:
        return 0
    return int((world_size - 1) / world_size * P_elems * dtype_bytes)


def _all_gather_bytes(P_elems: int, world_size: int, dtype_bytes: int) -> int:
    if world_size <= 1:
        return 0
    return int((world_size - 1) / world_size * P_elems * dtype_bytes)


def comms_for(strategy: str, P: int, mcfg: ModelConfig, wcfg: WorldConfig) -> CommsPerStep:
    N = wcfg.world_size
    d = mcfg.dtype_bytes
    if strategy == "data_parallel":
        # Backward: all-reduce all gradients across ranks.
        return CommsPerStep(
            strategy=strategy,
            forward_bytes=0,
            backward_bytes=_all_reduce_bytes(P, N, d),
            optimizer_step_bytes=0,
        )
    if strategy == "zero_1":
        # Optimizer states sharded => backward does an all-reduce on grads
        # (still needed because every rank must know full grads to send its own
        # slice to the owner) *or* a reduce-scatter + all-gather split.
        # Total volume is the same as data-parallel: ~2P per rank.
        return CommsPerStep(
            strategy=strategy,
            forward_bytes=0,
            backward_bytes=_all_reduce_bytes(P, N, d),
            optimizer_step_bytes=0,
        )
    if strategy == "zero_2":
        # Gradients sharded: reduce-scatter grads (rank r keeps its shard),
        # then optimizer step is local. Params are still full, so no
        # forward all-gather.
        return CommsPerStep(
            strategy=strategy,
            forward_bytes=0,
            backward_bytes=_reduce_scatter_bytes(P, N, d),
            optimizer_step_bytes=0,
        )
    if strategy == "zero_3":
        # Params sharded: forward and backward each need an all-gather to
        # materialize the full parameter set for compute, then discard.
        # Grads: reduce-scatter.
        return CommsPerStep(
            strategy=strategy,
            forward_bytes=_all_gather_bytes(P, N, d),
            backward_bytes=_all_gather_bytes(P, N, d) + _reduce_scatter_bytes(P, N, d),
            optimizer_step_bytes=0,
        )
    raise ValueError(strategy)


def all_comms(P: int, mcfg: ModelConfig, wcfg: WorldConfig) -> dict[str, CommsPerStep]:
    return {s: comms_for(s, P, mcfg, wcfg) for s in ("data_parallel", "zero_1", "zero_2", "zero_3")}
