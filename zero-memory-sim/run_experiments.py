#!/usr/bin/env python3
"""Run all ZeRO simulation experiments and write results.json + figures/."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.comms import all_comms
from src.config import ModelConfig, WorldConfig
from src.memory import STRATEGIES, all_strategies
from src.model import DeepMLP
from src.zero import BUILDERS, summarize_ranks

ROOT = Path(__file__).parent
FIG = ROOT / "figures"


STRAT_ORDER = ["data_parallel", "zero_1", "zero_2", "zero_3"]
STRAT_LABEL = {
    "data_parallel": "Data Parallel",
    "zero_1": "ZeRO-1",
    "zero_2": "ZeRO-2",
    "zero_3": "ZeRO-3",
}


def gib(bytes_: int) -> float:
    return bytes_ / (1024**3)


def mib(bytes_: int) -> float:
    return bytes_ / (1024**2)


def _bar_memory_by_strategy(P: int, mcfg: ModelConfig, wcfg: WorldConfig, path: Path) -> None:
    breakdown = all_strategies(P, mcfg, wcfg)
    params = [breakdown[s].param_bytes / (1024**2) for s in STRAT_ORDER]
    grads = [breakdown[s].grad_bytes / (1024**2) for s in STRAT_ORDER]
    opts = [breakdown[s].optimizer_bytes / (1024**2) for s in STRAT_ORDER]

    x = range(len(STRAT_ORDER))
    plt.figure(figsize=(8, 5))
    plt.bar(x, params, label="Parameters")
    plt.bar(x, grads, bottom=params, label="Gradients")
    plt.bar(
        x,
        opts,
        bottom=[p + g for p, g in zip(params, grads)],
        label="Optimizer states",
    )
    plt.xticks(list(x), [STRAT_LABEL[s] for s in STRAT_ORDER])
    plt.ylabel("Per-rank memory (MiB)")
    plt.title(f"Per-rank model-state memory, N={wcfg.world_size}")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_world_scaling(P: int, mcfg: ModelConfig, path: Path) -> None:
    world_sizes = [1, 2, 4, 8, 16, 32]
    plt.figure(figsize=(8, 5))
    for strat in STRAT_ORDER:
        y = []
        for N in world_sizes:
            wcfg = WorldConfig(world_size=N)
            rm = STRATEGIES[strat](P, mcfg, wcfg)
            y.append(rm.total_bytes / (1024**2))
        plt.plot(world_sizes, y, marker="o", label=STRAT_LABEL[strat], linewidth=2)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(world_sizes, [str(n) for n in world_sizes])
    plt.xlabel("World size (number of ranks)")
    plt.ylabel("Per-rank model-state memory (MiB, log)")
    plt.title("How per-rank memory scales with world size")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_comms(P: int, mcfg: ModelConfig, wcfg: WorldConfig, path: Path) -> None:
    comms = all_comms(P, mcfg, wcfg)
    totals = [comms[s].total_bytes / (1024**2) for s in STRAT_ORDER]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(STRAT_ORDER)), totals)
    plt.xticks(range(len(STRAT_ORDER)), [STRAT_LABEL[s] for s in STRAT_ORDER])
    plt.ylabel("Communication volume per rank per step (MiB)")
    plt.title(f"Analytical communication volume, N={wcfg.world_size}")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_total_vs_perrank(P: int, mcfg: ModelConfig, wcfg: WorldConfig, path: Path) -> None:
    breakdown = all_strategies(P, mcfg, wcfg)
    per_rank = [breakdown[s].total_bytes / (1024**2) for s in STRAT_ORDER]
    total_cluster = [
        breakdown[s].total_bytes * wcfg.world_size / (1024**2) for s in STRAT_ORDER
    ]
    x = range(len(STRAT_ORDER))
    width = 0.4
    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], per_rank, width=width, label="Per rank")
    plt.bar(
        [i + width / 2 for i in x], total_cluster, width=width, label=f"× N={wcfg.world_size} (cluster)"
    )
    plt.xticks(list(x), [STRAT_LABEL[s] for s in STRAT_ORDER])
    plt.ylabel("Memory (MiB)")
    plt.title("Per-rank vs whole-cluster model-state memory")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _sanity_check(P: int, mcfg: ModelConfig, wcfg: WorldConfig) -> dict[str, bool]:
    breakdown = all_strategies(P, mcfg, wcfg)
    dp = breakdown["data_parallel"]
    z1 = breakdown["zero_1"]
    z2 = breakdown["zero_2"]
    z3 = breakdown["zero_3"]

    N = wcfg.world_size
    tol = 1 / N  # allow one shard's worth of ceil rounding
    return {
        "zero1_optimizer_partitioned": abs(z1.optimizer_bytes * N - dp.optimizer_bytes)
        / dp.optimizer_bytes
        <= tol,
        "zero2_grad_partitioned": abs(z2.grad_bytes * N - dp.grad_bytes) / dp.grad_bytes
        <= tol,
        "zero3_param_partitioned": abs(z3.param_bytes * N - dp.param_bytes) / dp.param_bytes
        <= tol,
        "zero1_params_replicated": z1.param_bytes == dp.param_bytes,
        "zero2_params_replicated": z2.param_bytes == dp.param_bytes,
        "zero3_grads_partitioned": z3.grad_bytes * N >= dp.grad_bytes - dp.grad_bytes * tol,
    }


def _measure_tensor_bytes(P: int, mcfg: ModelConfig, wcfg: WorldConfig) -> dict:
    measured = {}
    for name in STRAT_ORDER:
        ranks = BUILDERS[name](P, mcfg, wcfg)
        measured[name] = summarize_ranks(ranks)
    return measured


def _time_local_step(mcfg: ModelConfig, iters: int = 20) -> dict:
    """A tiny wall-clock measurement so we have honest 'observed compute' numbers.

    This times the FORWARD + BACKWARD + OPTIMIZER STEP of a full replica on
    one process. Real distributed training also incurs communication, which
    we account for separately in ``comms.py``.
    """
    torch.manual_seed(0)
    model = DeepMLP(mcfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(32, mcfg.input_dim)
    y = torch.randn(32, mcfg.output_dim)
    loss_fn = torch.nn.MSELoss()

    # warmup
    for _ in range(3):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    fwd = bwd = step = 0.0
    for _ in range(iters):
        opt.zero_grad()
        t0 = time.perf_counter()
        out = model(x)
        loss = loss_fn(out, y)
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        opt.step()
        t3 = time.perf_counter()
        fwd += t1 - t0
        bwd += t2 - t1
        step += t3 - t2
    return {
        "iterations": iters,
        "forward_ms_per_iter": 1e3 * fwd / iters,
        "backward_ms_per_iter": 1e3 * bwd / iters,
        "optimizer_ms_per_iter": 1e3 * step / iters,
        "total_ms_per_iter": 1e3 * (fwd + bwd + step) / iters,
    }


def main() -> None:
    FIG.mkdir(exist_ok=True)
    mcfg = ModelConfig()
    wcfg = WorldConfig()

    torch.manual_seed(0)
    model = DeepMLP(mcfg)
    P = model.param_count()

    breakdown = all_strategies(P, mcfg, wcfg)
    comms = all_comms(P, mcfg, wcfg)
    measured = _measure_tensor_bytes(P, mcfg, wcfg)
    checks = _sanity_check(P, mcfg, wcfg)
    timings = _time_local_step(mcfg)

    _bar_memory_by_strategy(P, mcfg, wcfg, FIG / "per_rank_memory.png")
    _plot_world_scaling(P, mcfg, FIG / "world_size_scaling.png")
    _plot_comms(P, mcfg, wcfg, FIG / "comms_per_step.png")
    _plot_total_vs_perrank(P, mcfg, wcfg, FIG / "cluster_vs_rank.png")

    results = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu (32 virtual ranks; no physical multi-GPU used)",
        },
        "config": {"model": asdict(mcfg), "world": asdict(wcfg)},
        "param_count": P,
        "param_bytes": P * mcfg.dtype_bytes,
        "per_rank_memory_bytes": {s: asdict(breakdown[s]) for s in STRAT_ORDER},
        "per_rank_memory_measured_tensor_bytes": measured,
        "communication_per_step_bytes": {s: asdict(comms[s]) for s in STRAT_ORDER},
        "sanity_checks": checks,
        "local_timing_single_process": timings,
        "notes": (
            "Memory numbers are model-state only (params + grads + optimizer states). "
            "Activations, buffers, framework overhead and allocator fragmentation are "
            "excluded. Communication volumes are analytical, using ring-collective "
            "accounting."
        ),
    }

    (ROOT / "results.json").write_text(json.dumps(results, indent=2, default=str))
    for k, v in checks.items():
        assert v, f"sanity check failed: {k}"
    print("Wrote", ROOT / "results.json")
    print("P =", P)
    print("Per-rank total (MiB):", {STRAT_LABEL[s]: round(breakdown[s].total_bytes / (1024**2), 2) for s in STRAT_ORDER})


if __name__ == "__main__":
    main()
