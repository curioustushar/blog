---
title: "ZeRO Memory Truth"
slug: "zero-memory-sim"
date: 2026-09-17T02:15:00-05:00
categories: ["machine-learning"]
tags:
  ["zero", "deepspeed", "distributed", "memory", "notebook"]
author: Tushar Gupta
description: "A 32-rank ZeRO simulator built to prove to myself that I understand DP / ZeRO-1 / ZeRO-2 / ZeRO-3 and their memory-vs-communication trade-off."
---

<div class="post-summary">

Data-parallel training gives every GPU a full copy of the model. On 32 GPUs
that is 32 copies of parameters, gradients, and optimizer states — mostly
duplicated bytes. **DeepSpeed ZeRO** removes that duplication in three stages.

To make sure I actually understood the mechanics rather than the marketing, I
built a 32-rank simulator on a single CPU. Each rank owns *real* shards of
the model. Communication volumes are computed analytically. Every simplification
is called out — no fabricated benchmark numbers.

<div class="plan-status" role="status" aria-label="Project status">
  <span class="status-badge status-ok">DP + ZeRO-1/2/3 modeled</span>
  <span class="status-badge status-ok">Notebook runs on CPU</span>
  <span class="status-badge status-ok">World-size scaling plot</span>
</div>

</div>

---

## Links

| | URL |
|---|---|
| **GitHub** | [zero-memory-sim/](https://github.com/curioustushar/blog/tree/master/zero-memory-sim) |
| **Notebook** | [`notebook.ipynb`](https://github.com/curioustushar/blog/blob/master/zero-memory-sim/notebook.ipynb) |
| **Full README** | [README.md](https://github.com/curioustushar/blog/blob/master/zero-memory-sim/README.md) |

---

## The three lists

Every training run keeps three things per rank:

- **Parameters** — the weights (`P` scalars).
- **Gradients** — one number per weight, produced by backward.
- **Optimizer states** — Adam needs `m` and `v`, so `2P` scalars.

For fp32, that is `4P + 4P + 8P = 16P` bytes per rank. On 32 GPUs the cluster
stores `32 × 16P` — most of it identical copies. ZeRO's job is to stop paying
that duplication tax.

## What each ZeRO stage partitions

| Stage | Params | Gradients | Optimizer |
|-------|:------:|:---------:|:---------:|
| Data Parallel | full | full | full |
| ZeRO-1 | full | full | **sharded** |
| ZeRO-2 | full | **sharded** | **sharded** |
| ZeRO-3 | **sharded** | **sharded** | **sharded** |

Formulas (per rank):

\[
M_{DP} = 4P \cdot b, \quad
M_{Z1} = \left(2P + \frac{2P}{N}\right) b, \quad
M_{Z2} = \left(P + \frac{3P}{N}\right) b, \quad
M_{Z3} = \frac{4P}{N} \cdot b
\]

## Numbers from the simulator

For a 4-hidden-layer MLP with `P = 4,198,912` params, fp32, `N = 32`:

| Strategy | Per-rank total | Cluster-wide |
|----------|---------------:|-------------:|
| Data Parallel | **64.07 MiB** | 2,050 MiB |
| ZeRO-1 | **33.04 MiB** | 1,057 MiB |
| ZeRO-2 | **17.52 MiB** | 561 MiB |
| ZeRO-3 | **2.00 MiB** | 64 MiB |

ZeRO-3 stores exactly `1/32` of the parameters, gradients, and optimizer state
per rank. Sanity checks assert this — see `run_experiments.py`.

![Per-rank memory by strategy](../../zero-memory-sim/per_rank_memory.png)

## Communication is not free

ZeRO-3's memory savings come with an extra parameter all-gather every forward
and every backward.

| Strategy | Bytes/rank/step |
|----------|----------------:|
| Data Parallel | ~2P |
| ZeRO-1 | ~2P |
| ZeRO-2 | ~P |
| ZeRO-3 | **~3P** |

So the picture is:

- Memory drops **DP → Z1 → Z2 → Z3**.
- Comms is flat-ish for **DP/Z1/Z2** and higher for **Z3**.

Pick a stage based on the GPU-memory-to-network-bandwidth ratio of your
cluster, not by cargo culting.

![World-size scaling](../../zero-memory-sim/world_size_scaling.png)

## What I did NOT simulate

- No `torch.distributed` process groups — 32 GPUs are 32 rank objects on one CPU.
- No activation memory, no allocator fragmentation.
- No wall-clock benchmark of a real 32-GPU cluster.

Every simplification is stated in the README; the point of this project was
understanding the mechanics, not building DeepSpeed.

## Reproduce

```bash
git clone https://github.com/curioustushar/blog.git
cd blog/zero-memory-sim
pip install -r requirements.txt
python run_experiments.py
jupyter notebook notebook.ipynb
```
