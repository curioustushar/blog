---
title: "Evaluating Reversible Training in a 20M-Parameter LLM"
date: 2026-09-25T18:12:00-05:00
categories: ["machine-learning"]
tags: ["llm", "distributed-training", "reversibility", "optimization", "gpu"]
author: Tushar Gupta
description: "An end-to-end experiment comparing standard training against reversible layers on a 20M-parameter language model for 50M tokens, focusing on memory efficiency and throughput."
---

<div class="post-summary">

In this post, we build and document an end-to-end experiment that trains a ~20M-parameter language model for 50M tokens. We compare standard causal language model training against training with **reversible layers**. The core objective is to practically measure how reversibility allows us to bypass storing activations for the backward pass, and to experimentally discover how much we can push the batch size limit on a single GPU.

</div>

---

## 1. Project Objective

The goal of this experiment is to implement, run, and measure the memory and throughput trade-offs of **reversible training**. By comparing a standard 20M-parameter baseline model against a reversible variant, we aim to measure exactly how much GPU VRAM is saved by recomputing activations during the backward pass instead of caching them, and whether this theoretically enables a significantly larger maximum batch size without hitting Out-of-Memory (OOM) limits.

**Final Repository:** [Insert GitHub Link Here]

### Notebooks

| # | Notebook | Open in Colab |
|:--|:---------|:--------------|
| 1 | Baseline Training — 20M LLM, 50M tokens | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[your-github]/reversible-llm-training/blob/main/notebooks/01_baseline.ipynb) |
| 2 | Reversible Training — Euler & Midpoint variants | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[your-github]/reversible-llm-training/blob/main/notebooks/02_reversible.ipynb) |
| 3 | Max-Batch Reversible Training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[your-github]/reversible-llm-training/blob/main/notebooks/03_max_batch_reversible.ipynb) |

## 2. Hardware and Environment

All experiments were run on a single instance to ensure a fair comparison. 
*(Update these placeholders with your actual Colab/hardware details)*

* **GPU:** `[e.g., NVIDIA T4 / A100]`
* **GPU VRAM:** `[e.g., 16 GB]`
* **CPU / RAM:** `[e.g., 2vCPU, 12.7 GB RAM]`
* **Frameworks:** `[e.g., PyTorch 2.x, Transformers]`
* **Precision:** `[e.g., bfloat16 / fp16 / fp32]`

## 3. Model Architecture

We trained a small Causal Language Model modeled roughly on modern transformer architectures. 

* **Architecture Type:** Causal Transformer (Decoder-only)
* **Parameter Count:** `~20M` 
* **Parameter Calculation:** 
  *(Explain your parameter count here. For example: `Vocab Size * Embed Dim + Layers * (Attn_Params + MLP_Params) = Total`)*

## 4. Dataset and Tokenizer

* **Dataset:** `[e.g., TinyShakespeare, OpenWebText sample, etc.]`
* **Tokenizer:** `[e.g., GPT-2 BPE, SentencePiece]`
* **Sequence Length:** `[e.g., 512 / 1024]`
* **Total Token Budget:** `50M tokens`

## 5. Training Configuration

To maintain a fair comparison, the following hyperparameters were kept constant across baseline and reversible runs (until the max-batch experiment).

* **Batch Size (Baseline/Reversible):** `[e.g., 16]`
* **Gradient Accumulation Steps:** `[e.g., 1 or 4]`
* **Effective Batch Size:** `[e.g., 64]`
* **Tokens per Optimizer Step:** `[e.g., Seq_Len * Eff_Batch_Size]`
* **Optimizer:** `[e.g., AdamW (lr=3e-4)]`
* **Learning-rate schedule:** `[e.g., Cosine with warmup]`

---

## 6. Methodology

### 6.1 Baseline Methodology
The baseline is a standard implementation of a Transformer. During the forward pass, all intermediate activations are saved to memory so they can be used during the backward pass to compute gradients. This is fast but highly memory-intensive as the sequence length and batch size grow.

### 6.2 Reversible Methodology
To save memory, we modified the architecture to use **Reversible Blocks**. Instead of storing the input to every layer, we only store the input to the first block. During the backward pass, the inputs to each subsequent layer are mathematically reconstructed on-the-fly.

#### Reversible Formulation Used
*(Specify which formulation you ended up using)*
We utilized the **[Euler / Midpoint]** reversible integration. 
* **Explanation:** In this formulation, a block is split into two streams `(x1, x2)`. The forward pass looks like:
  `y1 = x1 + F(x2)`
  `y2 = x2 + G(y1)`
  This makes the backward reconstruction trivial:
  `x2 = y2 - G(y1)`
  `x1 = y1 - F(x2)`
* **Why this method:** `[Explain why you picked Euler or Midpoint, e.g., stability, ease of implementation, etc.]`

---

## 7. Results and Measurements

> **Measurement Note:** GPU memory was measured using `torch.cuda.max_memory_allocated()`. Throughput (tokens/sec) was calculated strictly over the active training loops, excluding initial dataset download and model initialization times. 

### 7.1 Fair Comparison Table

| Experiment | Params | Batch | Seq Len | Tokens | Final Loss | Tokens/s | Peak VRAM | Time |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Baseline** | `~20M` | `[X]` | `[L]` | 50M | `[Loss]` | `[T/s]` | `[VRAM]` | `[Time]` |
| **Reversible – [Variant]** | `~20M` | `[X]` | `[L]` | 50M | `[Loss]` | `[T/s]` | `[VRAM]` | `[Time]` |
| **Reversible – Max Batch** | `~20M` | `[Max]` | `[L]` | 50M | `[Loss]` | `[T/s]` | `[VRAM]` | `[Time]` |

### 7.2 The Maximum-Batch Experiment
For our final run, we took our successful Reversible implementation and aggressively increased the batch size until we hit the GPU memory limit. 
* **How Max Batch was determined:** `[Explain your binary search or step-up process for finding the OOM ceiling]`
* **Result:** We successfully scaled the batch size from `[Baseline Batch]` to `[Max Batch]`, demonstrating the true practical benefit of reversible layers.

### 7.3 Loss Curves
*(Insert your matplotlib/wandb loss curves here)*
`![Loss Curves](/path/to/image.png)`

---

## 8. Reproducibility

The complete codebase, including the Jupyter notebooks designed for Google Colab, is available in the repository. 

### Structure
```text
repo/
├── README.md
├── notebooks/
│   ├── 01_baseline.ipynb
│   ├── 02_reversible.ipynb
│   └── 03_max_batch_reversible.ipynb
├── src/
│   ├── model.py
│   ├── reversible.py
│   ├── train.py
│   └── ...
```

### Quick Start (Google Colab)
1. Clone the repository: `git clone [Your Repo URL]`
2. Upload the notebooks in `/notebooks` to Google Colab.
3. Ensure runtime is set to GPU (T4).
4. Run all cells in `01_baseline.ipynb` and `02_reversible.ipynb`.

---

## 9. Observations and Challenges

### Reconstruction and Numerical Stability
`[Did you face any NaN/Inf issues? How did you test that your reconstruction was accurate? Mention your numerical reconstruction test here.]`

### Problems Encountered and Fixes
* **OOMs on Baseline:** `[Describe if your baseline OOM'd during initial testing and how you found the safe batch size]`
* **Reversibility Overhead:** `[Describe the compute overhead of re-calculating activations on the backward pass and how it affected tokens/sec]`

### Limitations
* `[Mention limitations, e.g., the reversible implementation required strict residual connections and didn't allow certain normalization placements]`
* `[Mention if precision scaling (like bf16) changed the reconstruction error]`

## 10. Conclusions

Based strictly on our measurements:
1. **Memory:** Reversible training reduced peak VRAM consumption by `[X]%` for the same batch size.
2. **Throughput:** Recomputing activations comes with a compute penalty, reducing tokens/sec by roughly `[X]%`.
3. **Scaling:** The memory savings allowed us to increase our maximum batch size by a factor of `[X]`, which ultimately `[improved/worsened]` overall hardware utilization. 

For highly memory-constrained environments where activation memory dwarfs parameter memory, reversible architectures provide a massive scaling advantage, though they trade compute time for memory space.
