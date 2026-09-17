#!/usr/bin/env python3
"""Generate notebook.ipynb for the ZeRO memory simulator.

The notebook follows the assignment's 12-section structure with markdown
before every code cell. Regenerate with:

    python build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
NB_PATH = ROOT / "notebook.ipynb"


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _join(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _join(lines),
    }


def _join(lines: tuple[str, ...]) -> list[str]:
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def build() -> dict:
    cells = [
        md(
            "# ZeRO Memory Simulation — 32 Virtual Ranks",
            "",
            "**What this notebook demonstrates:** how DeepSpeed-style ZeRO stages 1, 2 and 3",
            "change per-rank memory (and, at the analytical level, communication) versus",
            "conventional data-parallel training.",
            "",
            "**What is real and what is simulated:**",
            "",
            "| Aspect | Real | Simulated |",
            "|--------|:----:|:---------:|",
            "| PyTorch model, parameters, gradients, Adam state | ✅ | |",
            "| Per-rank shard tensors and their `element_size * numel` bytes | ✅ | |",
            "| 32 separate CUDA GPUs | | ✅ (32 logical ranks in one process) |",
            "| Cross-process collectives | | ✅ (analytical volumes, not `torch.distributed`) |",
            "",
            "Every simplification is called out explicitly. The point is understanding, not",
            "reproducing DeepSpeed.",
        ),
        md("## 1. Environment"),
        code(
            "import platform, sys, torch",
            "print('Python :', platform.python_version())",
            "print('PyTorch:', torch.__version__)",
            "print('Device backend:', 'cuda' if torch.cuda.is_available() else 'cpu')",
            "print('Physical CUDA devices:', torch.cuda.device_count())",
            "print('World size (virtual ranks): 32')",
        ),
        md("## 2. Configuration and the demo model",
           "",
           "A simple deep MLP. Depth/width and dtype are configurable in `src/config.py`.",
           "",
           "Parameter counts (in bytes):",
           "",
           "- **P** = number of scalar parameters (`model.param_count()`)",
           "- **b** = bytes per scalar (`ModelConfig.dtype_bytes`; fp32 → 4)",
           "- **k** = number of optimizer states per parameter (Adam/AdamW → 2 for `m` and `v`)"),
        code(
            "from src.config import ModelConfig, WorldConfig",
            "from src.model import DeepMLP",
            "",
            "mcfg = ModelConfig()",
            "wcfg = WorldConfig()",
            "torch.manual_seed(0)",
            "model = DeepMLP(mcfg)",
            "P = model.param_count()",
            "print(f'World size N = {wcfg.world_size}')",
            "print(f'Parameters P = {P:,}')",
            "print(f'Bytes per scalar b = {mcfg.dtype_bytes} (fp32)')",
            "print(f'Optimizer states per param k = {wcfg.optimizer_states_per_param} (Adam m, v)')",
            "print(f'Full parameter set = {P * mcfg.dtype_bytes / 1024**2:.2f} MiB')",
        ),
        md(
            "## 3. Baseline: conventional data parallelism",
            "",
            "In vanilla data-parallel training, **every rank stores everything**:",
            "",
            "- Parameters:  `P · b` bytes",
            "- Gradients:   `P · b` bytes",
            "- Optimizer:   `k · P · b` bytes",
            "",
            "For Adam (k=2) and fp32 (b=4), that's `4P + 4P + 8P = 16P` bytes per rank.",
            "",
            "The **cluster** as a whole therefore stores `N × 16P` bytes — most of it duplicated.",
            "ZeRO exists to remove that duplication.",
        ),
        code(
            "from src.memory import all_strategies",
            "from run_experiments import STRAT_ORDER, STRAT_LABEL, mib",
            "",
            "breakdown = all_strategies(P, mcfg, wcfg)",
            "for s in STRAT_ORDER:",
            "    m = breakdown[s]",
            "    print(f\"{STRAT_LABEL[s]:>15}: params={mib(m.param_bytes):.2f} MiB  \"",
            "          f\"grads={mib(m.grad_bytes):.2f} MiB  \"",
            "          f\"optim={mib(m.optimizer_bytes):.2f} MiB  \"",
            "          f\"total={mib(m.total_bytes):.2f} MiB per rank\")",
        ),
        md(
            "## 4. ZeRO-1 — partition **optimizer states**",
            "",
            "Concept in this simulation: every rank still stores the full `P` parameters and",
            "the full `P` gradients, but the `2P` scalars of Adam state are split across the",
            "`N=32` ranks. Each rank owns approximately `2P/N` optimizer scalars.",
            "",
            "Formula: `M_Z1 = (P + P + 2P/N) · b` bytes per rank.",
        ),
        code(
            "from src.zero import BUILDERS, summarize_ranks",
            "",
            "ranks = BUILDERS['zero_1'](P, mcfg, wcfg)",
            "s = summarize_ranks(ranks)",
            "print(f\"ZeRO-1 rank 0: params={mib(s['param_bytes_rank0']):.2f} MiB, \"",
            "      f\"grads={mib(s['grad_bytes_rank0']):.2f} MiB, \"",
            "      f\"optim={mib(s['optim_bytes_rank0']):.2f} MiB\")",
            "assert abs(s['optim_bytes_rank0'] * wcfg.world_size - 2 * P * mcfg.dtype_bytes) < 2 * mcfg.dtype_bytes * wcfg.world_size",
            "print('OK — ZeRO-1 optimizer state ≈ 2P·b / N per rank')",
        ),
        md(
            "## 5. ZeRO-2 — partition **gradients** too",
            "",
            "Now gradients are also split across ranks. Every rank still holds all `P`",
            "parameters (needed for the forward pass), but each rank keeps only its slice",
            "of the gradient tensor.",
            "",
            "Formula: `M_Z2 = (P + P/N + 2P/N) · b` bytes per rank.",
        ),
        code(
            "ranks = BUILDERS['zero_2'](P, mcfg, wcfg)",
            "s = summarize_ranks(ranks)",
            "print(f\"ZeRO-2 rank 0: params={mib(s['param_bytes_rank0']):.2f} MiB, \"",
            "      f\"grads={mib(s['grad_bytes_rank0']):.2f} MiB, \"",
            "      f\"optim={mib(s['optim_bytes_rank0']):.2f} MiB\")",
            "assert abs(s['grad_bytes_rank0'] * wcfg.world_size - P * mcfg.dtype_bytes) < mcfg.dtype_bytes * wcfg.world_size",
            "print('OK — ZeRO-2 gradient ≈ P·b / N per rank')",
        ),
        md(
            "## 6. ZeRO-3 — partition **parameters** too",
            "",
            "The final stage: parameters, gradients and optimizer states are all sharded.",
            "Each rank stores roughly `1/N` of every model tensor.",
            "",
            "Formula: `M_Z3 = (P/N + P/N + 2P/N) · b = 4P·b / N` per rank.",
            "",
            "To do a forward pass, ranks need to *temporarily* reconstruct the full parameter",
            "set via an `all-gather`, then discard the non-owned parts. That is where ZeRO-3's",
            "extra communication comes from — see section 8.",
        ),
        code(
            "ranks = BUILDERS['zero_3'](P, mcfg, wcfg)",
            "s = summarize_ranks(ranks)",
            "print(f\"ZeRO-3 rank 0: params={mib(s['param_bytes_rank0']):.2f} MiB, \"",
            "      f\"grads={mib(s['grad_bytes_rank0']):.2f} MiB, \"",
            "      f\"optim={mib(s['optim_bytes_rank0']):.2f} MiB\")",
            "assert abs(s['param_bytes_rank0'] * wcfg.world_size - P * mcfg.dtype_bytes) < mcfg.dtype_bytes * wcfg.world_size",
            "print('OK — ZeRO-3 param ≈ P·b / N per rank')",
        ),
        md("## 7. Memory comparison — theoretical formula vs measured tensor bytes"),
        code(
            "rows = []",
            "for s in STRAT_ORDER:",
            "    theo = breakdown[s]",
            "    meas = summarize_ranks(BUILDERS[s](P, mcfg, wcfg))",
            "    rows.append((STRAT_LABEL[s], theo.param_bytes, meas['param_bytes_rank0'],",
            "                 theo.grad_bytes, meas['grad_bytes_rank0'],",
            "                 theo.optimizer_bytes, meas['optim_bytes_rank0']))",
            "",
            "hdr = f\"{'Strategy':>15} | {'params (MiB)':>18} | {'grads (MiB)':>18} | {'optim (MiB)':>18}\"",
            "print(hdr)",
            "print('-' * len(hdr))",
            "for name, tp, mp, tg, mg, to, mo in rows:",
            "    print(f'{name:>15} | {mib(tp):8.2f} vs {mib(mp):5.2f} | {mib(tg):8.2f} vs {mib(mg):5.2f} | {mib(to):8.2f} vs {mib(mo):5.2f}')",
            "print()",
            "print('theoretical  vs  measured (bytes of the actual per-rank tensors)')",
        ),
        md(
            "## 8. Communication cost per step",
            "",
            "Using ring-collective accounting for `N` ranks:",
            "",
            "- **all-reduce(P)** ≈ `2·(N-1)/N · P · b` bytes per rank ≈ `2·P·b`",
            "- **reduce-scatter(P)** ≈ `(N-1)/N · P · b`",
            "- **all-gather(P)** ≈ `(N-1)/N · P · b`",
            "",
            "| Strategy | Forward | Backward | Total per rank per step |",
            "|----------|:-------:|:--------:|:-----------------------:|",
            "| Data Parallel | — | all-reduce grads | ~2P |",
            "| ZeRO-1 | — | all-reduce grads | ~2P |",
            "| ZeRO-2 | — | reduce-scatter grads | ~P |",
            "| ZeRO-3 | all-gather params | all-gather params + reduce-scatter grads | ~3P |",
            "",
            "So **memory drops** through the stages, but **ZeRO-3 costs an extra all-gather**.",
            "That is the fundamental compute/memory trade-off.",
        ),
        code(
            "from src.comms import all_comms",
            "comms = all_comms(P, mcfg, wcfg)",
            "for s in STRAT_ORDER:",
            "    c = comms[s]",
            "    print(f'{STRAT_LABEL[s]:>15}: fwd={mib(c.forward_bytes):6.2f} MiB  '",
            "          f'bwd={mib(c.backward_bytes):6.2f} MiB  '",
            "          f'step={mib(c.optimizer_step_bytes):6.2f} MiB  '",
            "          f'total={mib(c.total_bytes):6.2f} MiB per rank per step')",
        ),
        md(
            "## 9. World-size scaling",
            "",
            "As `N` grows, ZeRO stages diverge:",
            "",
            "- Data Parallel stays flat (nothing sharded).",
            "- ZeRO-1 drops toward the `2P` params+grads floor.",
            "- ZeRO-2 drops toward `P` (only params replicated).",
            "- ZeRO-3 drops linearly with `N` (nothing replicated).",
        ),
        code(
            "from IPython.display import Image, display",
            "display(Image('figures/world_size_scaling.png'))",
        ),
        md(
            "## 10. Local timing (single process)",
            "",
            "Since we cannot spawn 32 real GPUs, we time forward/backward/optimizer on **one**",
            "full replica. The numbers are *not* the wall-clock time of a real 32-GPU cluster",
            "— that would require adding the communication times from §8, which are network",
            "hardware dependent.",
        ),
        code(
            "from run_experiments import _time_local_step",
            "timings = _time_local_step(mcfg)",
            "for k, v in timings.items():",
            "    print(f'{k}: {v}')",
        ),
        md(
            "## 11. Sanity checks — the formulas hold",
            "",
            "For `N = 32`:",
            "",
            "- ZeRO-1: optimizer bytes/rank × N ≈ 2P·b",
            "- ZeRO-2: gradient bytes/rank × N ≈ P·b",
            "- ZeRO-3: parameter bytes/rank × N ≈ P·b",
        ),
        code(
            "from run_experiments import _sanity_check",
            "checks = _sanity_check(P, mcfg, wcfg)",
            "for k, v in checks.items():",
            "    print(f'{k}: {\"PASS\" if v else \"FAIL\"}')",
            "assert all(checks.values())",
        ),
        md(
            "## 12. Simulation vs real hardware",
            "",
            "**Faithfully represented**",
            "",
            "- Per-rank memory footprint of params/grads/optimizer state (real tensor bytes).",
            "- The 1/N scaling law for whichever category each ZeRO stage partitions.",
            "- Analytical communication volume per collective.",
            "",
            "**Not represented**",
            "",
            "- Real network latency and bandwidth (no NCCL / no gloo transport in this notebook).",
            "- Activation memory and framework overhead (excluded so formulas stay clean).",
            "- Overlap of communication with compute (ZeRO implementations hide much of the comms).",
            "- CUDA allocator behaviour and fragmentation.",
        ),
        md(
            "## 13. Conclusions",
            "",
            "- Memory savings go **DP → ZeRO-1 → ZeRO-2 → ZeRO-3** and grow with world size.",
            "- Communication is roughly constant for DP/ZeRO-1/ZeRO-2 (~2P or ~P per step) and",
            "  higher for ZeRO-3 (~3P) because it must all-gather parameters for compute.",
            "- The choice of ZeRO stage is therefore a **memory vs communication** trade-off,",
            "  not a free lunch.",
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build()
    NB_PATH.write_text(json.dumps(nb, indent=2))
    print(f"Wrote {NB_PATH}")
