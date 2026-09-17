# ZeRO Memory Simulation — 32 Virtual Ranks

A small, honest simulator for **DeepSpeed ZeRO stages 1, 2 and 3** on top of a
32-rank virtual world. I wrote this to convince *myself* that I understand the
memory story of ZeRO, not to build a production-grade distributed trainer.

**Blog:** [ZeRO Memory Truth](https://curioustushar.github.io/blog/posts/zero-memory-sim/)  
**Notebook:** [`notebook.ipynb`](./notebook.ipynb) — runs end-to-end on CPU in under a minute.

### One-line descriptions (for sharing)

| Where | Line |
|-------|------|
| **Git README** (this file) | Runnable ZeRO 1/2/3 simulator on 32 virtual ranks — real per-rank tensor bytes and analytical collective volumes on a single CPU. |
| **Blog** | I built a 32-rank ZeRO simulator to understand exactly what each stage partitions, why memory drops, and what it costs in communication. |
| **GitHub project** | [`zero-memory-sim/`](https://github.com/curioustushar/blog/tree/master/zero-memory-sim) — notebook + `src/` helpers regenerate memory tables and figures for DP / ZeRO-1 / ZeRO-2 / ZeRO-3. |

---

## 1. What this project demonstrates

Distributed training divides work across many GPUs. Vanilla **data parallelism**
gives every GPU a full copy of the model — parameters, gradients, and optimizer
states — and reduces gradients across ranks. It works, but every rank duplicates
`~16P` bytes for a fp32 Adam model. **ZeRO** ("Zero Redundancy Optimizer")
removes that duplication one category at a time.

I simulate 32 virtual GPUs in one Python process. Each rank is a small object
that owns its shard of `parameters / gradients / optimizer state`. I measure
memory by asking each rank's tensors for `element_size() * numel()`, and I
compute cross-rank communication volumes analytically using the standard
ring-collective formulas.

## 2. What "32 virtual GPUs" means here

**Real:**

- A real PyTorch model (`DeepMLP`), real parameters, real Adam optimizer.
- Real per-rank tensors sized to whatever the stage says that rank should own.
- Real byte accounting: shard size × dtype size.

**Simulated:**

- 32 separate CUDA devices → **32 logical rank objects** on one CPU process.
- `torch.distributed` collectives → **analytical volumes** (ring reduce-scatter,
  all-gather, all-reduce). I do not spawn 32 gloo processes because that would
  make the notebook slow and painful to run in Colab CPU, and it would not
  change the memory numbers.

No physical multi-GPU cluster was used. The mechanics that this simulation
faithfully represents are:

1. Which category of state is replicated vs partitioned at each stage.
2. Per-rank footprint of the categories, exactly.
3. Total collective volume per iteration.

The mechanics that are **not** simulated: network latency, activation memory,
CUDA allocator fragmentation, framework overhead, communication/computation
overlap.

## 3. How I understand ZeRO (student voice)

I keep three lists in my head:

- **Parameters** — the weights the model uses in forward.
- **Gradients** — one number per parameter, produced by backward.
- **Optimizer states** — Adam needs `m` and `v` per parameter, so `2P` scalars.

Data parallel gives every rank all three lists. That is `16P` bytes per rank
for fp32 Adam. On 32 ranks the cluster stores `32 × 16P` — most of it identical
copies.

**ZeRO-1** notices that the optimizer step is trivially local. Each rank can
own a *slice* of `m` and `v`, do the update on its slice, and then broadcast
the resulting parameter slice back. So we shard the `2P` optimizer scalars
across 32 ranks: `~2P/32` optimizer bytes per rank. Params and grads are still
full.

**ZeRO-2** goes one step further. The gradient tensor is only needed to feed
the optimizer step and can be sharded too. Each rank keeps its slice of the
gradient; the backward pass ends with a **reduce-scatter** instead of an
all-reduce, which naturally leaves each rank holding its slice. Params are
still full.

**ZeRO-3** is the "everything is sharded" stage. Each rank stores `1/N` of
the parameters, gradients, and optimizer states. Because forward and backward
need the *full* parameter set for compute, ranks must **all-gather** parameters
temporarily and then discard them. That extra all-gather is where ZeRO-3 pays
for its memory savings.

The formula summary is:

```
data_parallel  M/rank = (P + P + 2P) · b = 16P bytes (fp32)
ZeRO-1         M/rank = (P + P + 2P/N) · b
ZeRO-2         M/rank = (P + P/N + 2P/N) · b
ZeRO-3         M/rank = (P/N + P/N + 2P/N) · b = 4P·b / N
```

with `P` = parameter count, `b` = bytes per scalar, `N` = world size.

I like ZeRO-3's number: on 32 ranks it stores `1/8` of what data parallel
stores per rank, at the cost of extra all-gather traffic every step.

## 4. Experimental setup

| Item | Value |
|------|-------|
| Python | 3.11 |
| PyTorch | 2.2 |
| Device | CPU (32 virtual ranks) |
| Model | `DeepMLP` (4 hidden layers × 1024 hidden) |
| Params `P` | **4,198,912** (~4.2 M) |
| dtype | fp32 (`b = 4`) |
| Optimizer | AdamW (`k = 2`) |
| World size `N` | 32 |

## 5. Results (verified — see `results.json`)

Per-rank model-state memory:

| Strategy | Params | Grads | Optim | **Total per rank** |
|----------|-------:|------:|------:|-------------------:|
| Data Parallel | 16.02 MiB | 16.02 MiB | 32.03 MiB | **64.07 MiB** |
| ZeRO-1 | 16.02 MiB | 16.02 MiB | 1.00 MiB | **33.04 MiB** |
| ZeRO-2 | 16.02 MiB | 0.50 MiB | 1.00 MiB | **17.52 MiB** |
| ZeRO-3 | 0.50 MiB | 0.50 MiB | 1.00 MiB | **2.00 MiB** |

Analytical communication per rank per step:

| Strategy | Forward | Backward | Total |
|----------|--------:|---------:|------:|
| Data Parallel | 0 | all-reduce grads ≈ 2P·b | 31.03 MiB |
| ZeRO-1 | 0 | all-reduce grads ≈ 2P·b | 31.03 MiB |
| ZeRO-2 | 0 | reduce-scatter grads ≈ P·b | 15.52 MiB |
| ZeRO-3 | all-gather params | all-gather + reduce-scatter | 46.55 MiB |

Local single-process timing (informational only — a real 32-rank cluster
would add network time):

```
forward   : ~2.8 ms / iter
backward  : ~4.7 ms / iter
optimizer : ~17.5 ms / iter
total     : ~25.0 ms / iter
```

Sanity checks (from `run_experiments.py`, all pass):

- `zero1_optimizer_partitioned` — ZeRO-1 optimizer bytes/rank × N ≈ `2P·b`
- `zero2_grad_partitioned` — ZeRO-2 gradient bytes/rank × N ≈ `P·b`
- `zero3_param_partitioned` — ZeRO-3 parameter bytes/rank × N ≈ `P·b`
- `zero{1,2}_params_replicated` — ZeRO-1 and -2 still hold full parameters
- `zero3_grads_partitioned` — ZeRO-3 shards gradients too

## 6. World-size scaling

I re-ran the theoretical formulas for `N ∈ {1, 2, 4, 8, 16, 32}` and plotted:

- Data Parallel is flat — nothing shrinks.
- ZeRO-1 flattens once `2P/N` becomes negligible next to `2P`.
- ZeRO-2 flattens once both `P/N` and `2P/N` shrink.
- ZeRO-3 keeps dropping — it scales linearly with `N`.

That plot is `figures/world_size_scaling.png`.

## 7. Simulation vs real hardware

**Faithful:** per-rank memory footprint, category-by-category, and the exact
1/N scaling law for whichever category each stage partitions.

**Approximated:** communication is expressed as bytes per rank per step
using ring accounting. Actual wall-clock time depends on interconnect
bandwidth, latency, and how well the framework overlaps comms with compute.

**Not modeled:** activations, forward/backward buffers, allocator
fragmentation, mixed-precision master weights, offloading, real
`torch.distributed` process groups.

## 8. Repository structure

```
zero-memory-sim/
├── README.md               # this file
├── requirements.txt
├── run_experiments.py      # regenerates results.json + figures/
├── build_notebook.py       # regenerates notebook.ipynb
├── notebook.ipynb          # the primary deliverable
├── figures/                # generated PNGs
└── src/
    ├── config.py           # ModelConfig, WorldConfig
    ├── model.py            # DeepMLP
    ├── memory.py           # analytical per-rank byte formulas
    ├── zero.py             # rank-level tensor shard builders
    └── comms.py            # analytical collective volumes
```

## 9. How to run

```bash
git clone https://github.com/curioustushar/blog.git
cd blog/zero-memory-sim
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_experiments.py       # results.json + figures/
jupyter notebook notebook.ipynb # or open in Colab
```

Colab: upload `notebook.ipynb` plus the `src/` folder, run all. CPU runtime
is enough — the whole notebook finishes in under a minute.

## 10. Limitations

- No `torch.distributed` process groups — the collectives are counted, not run.
- No CUDA memory measurements; results depend on tensor bytes only.
- Activation memory is out of scope; adding it would require a specific batch
  size and layer-by-layer accounting.
- No offloading (ZeRO-Infinity-style CPU or NVMe tiers).
- No mixed precision (fp16 / bf16 master + fp32 gradient accumulators). Those
  change the constants in the formulas but not the shape of the story.

## 11. Conclusions

- Data-parallel replication is memory-wasteful and gets worse with world size.
- ZeRO-1/2/3 remove that redundancy one class at a time.
- **ZeRO-3 is not a free lunch**: memory drops linearly with `N` but requires
  an extra parameter all-gather per forward and per backward.
- Beyond a small world size, the ZeRO-2 vs ZeRO-3 choice becomes a memory vs
  communication trade-off — pick based on the ratio of GPU memory to network
  bandwidth in your cluster.
