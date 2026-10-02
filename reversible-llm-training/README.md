# Evaluating Reversible Training in a 20M-Parameter LLM

An end-to-end experiment comparing standard training against reversible layers on a 20M-parameter causal language model for 50M tokens. This project measures exactly how much GPU VRAM is saved by recomputing activations during the backward pass instead of caching them, and experimentally pushes the batch size to its absolute limit on a single GPU.

## 1. Project Objective

The goal of this experiment is to implement, run, and measure the memory and throughput trade-offs of **reversible training**. By comparing a standard 20M-parameter baseline model against a reversible variant, we aim to discover how much we can push the maximum batch size on a single GPU without hitting Out-of-Memory (OOM) limits.

### Notebooks

| # | Notebook | Open in Colab |
|:--|:---------|:--------------|
| 1 | Baseline Training — 20M LLM, 50M tokens | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[your-github]/reversible-llm-training/blob/main/notebooks/01_baseline.ipynb) |
| 2 | Reversible Training — Euler & Midpoint variants | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[your-github]/reversible-llm-training/blob/main/notebooks/02_reversible.ipynb) |
| 3 | Max-Batch Reversible Training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[your-github]/reversible-llm-training/blob/main/notebooks/03_max_batch_reversible.ipynb) |

## 2. Hardware and Environment

All experiments were run on a single instance to ensure a fair comparison. 
*(Update these placeholders with your actual Colab/hardware details after running)*

* **GPU:** `[e.g., NVIDIA T4]`
* **GPU VRAM:** `[e.g., 16 GB]`
* **CPU / RAM:** `[e.g., 2vCPU, 12.7 GB RAM]`
* **Frameworks:** `PyTorch 2.x, Transformers, Datasets`
* **Precision:** `[e.g., bfloat16]`

## 3. Model Architecture & Tokenizer

We trained a small Causal Language Model modeled on modern transformer architectures. 

* **Architecture Type:** Causal Transformer (Decoder-only)
* **Parameter Count:** `~19.23M` 
* **Dataset:** `TinyStories` (streamed via HuggingFace)
* **Tokenizer:** `GPT-2 (vocab size 50,257)`
* **Sequence Length:** `256`
* **Total Token Budget:** `50M tokens`

## 4. Methodology

### 4.1 Baseline Methodology
The baseline is a standard implementation of a Transformer (Notebook 01). During the forward pass, all intermediate activations are saved to memory so they can be used during the backward pass to compute gradients. This is fast but highly memory-intensive.

### 4.2 Reversible Methodology
To save memory, we modified the architecture to use **Reversible Blocks** (Notebook 02). Instead of storing the input to every layer, we only store the input to the first block. During the backward pass, the inputs to each subsequent layer are mathematically reconstructed on-the-fly.
We implemented and tested both **Euler** and **Midpoint** reversible integration variants.

## 5. Results and Measurements

> **Measurement Note:** GPU memory was measured using `torch.cuda.max_memory_allocated()`. Throughput (tokens/sec) was calculated strictly over the active training loops, excluding initial dataset download and model initialization times. 

### 5.1 Fair Comparison Table

| Experiment | Params | Batch | Seq Len | Tokens | Final Loss | Tokens/s | Peak VRAM | Time |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Baseline** | `~19.2M` | `[X]` | 256 | 50M | `[Loss]` | `[T/s]` | `[VRAM]` | `[Time]` |
| **Reversible – Euler** | `~19.2M` | `[X]` | 256 | 50M | `[Loss]` | `[T/s]` | `[VRAM]` | `[Time]` |
| **Reversible – Midpoint** | `~19.2M` | `[X]` | 256 | 50M | `[Loss]` | `[T/s]` | `[VRAM]` | `[Time]` |
| **Reversible – Max Batch** | `~19.2M` | `[Max]` | 256 | 50M | `[Loss]` | `[T/s]` | `[VRAM]` | `[Time]` |

### 5.2 The Maximum-Batch Experiment
For our final run (Notebook 03), we took our successful Reversible implementation and aggressively increased the batch size until we hit the GPU memory limit using an automated binary-search step-up process. 

## 6. Reproducibility

### Structure
```text
repo/
├── README.md
├── notebooks/
│   ├── 01_baseline.ipynb
│   ├── 02_reversible.ipynb
│   └── 03_max_batch_reversible.ipynb
```

### Quick Start (Google Colab)
1. Clone this repository or open the notebooks directly via the Colab badges above.
2. Ensure runtime is set to GPU (T4).
3. Run `01_baseline.ipynb` first to generate baseline metrics.
4. Run `02_reversible.ipynb` and `03_max_batch_reversible.ipynb`. 

## 7. Conclusions

*(Fill this in based on the observed measurements in the comparison table)*
1. **Memory:** Reversible training reduced peak VRAM consumption by `[X]%` for the same batch size.
2. **Throughput:** Recomputing activations comes with a compute penalty, reducing tokens/sec by roughly `[X]%`.
3. **Scaling:** The memory savings allowed us to increase our maximum batch size by a factor of `[X]`. 
