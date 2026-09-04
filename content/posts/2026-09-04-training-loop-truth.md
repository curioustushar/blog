---
title: "Training Loop Truth"
slug: "training-loop-truth"
date: 2026-09-04T13:30:00-05:00
categories: ["machine-learning"]
tags:
  ["training", "gradients", "MFU", "bf16", "fp8", "notebook"]
author: Tushar Gupta
description: "A small model and a real training loop — shapes, hand-verified gradients, broken accumulation, grad norms, honest MFU, and floating-point bit patterns."
---

<div class="post-summary">

The loss curve can lie. The loop can look healthy while averaging wrong, accumulating wrong, or running at 0.03% MFU.

This post accompanies a notebook that makes a **tiny char LM** tell the truth about itself: every tensor shape explained, one gradient nudged by hand, gradient accumulation broken on purpose, grad norms logged every step, MFU computed honestly, and `0.1` written out in FP32, BF16, and FP8 E4M3.

<div class="plan-status" role="status" aria-label="Project status">
  <span class="status-badge status-ok">Shapes + grad check</span>
  <span class="status-badge status-ok">Accumulation gap plotted</span>
  <span class="status-badge status-ok">MFU + float bits</span>
</div>

</div>

---

## Links

| | URL |
|---|---|
| **GitHub** | [training-loop-truth/](https://github.com/curioustushar/blog/tree/master/training-loop-truth) |
| **Notebook** | [`notebook.ipynb`](https://github.com/curioustushar/blog/blob/master/training-loop-truth/notebook.ipynb) |
| **Full write-up** | [docs/training-loop-truth-writeup.md](https://github.com/curioustushar/blog/blob/master/docs/training-loop-truth-writeup.md) |

---

## Configuration

TinyCharLM — **8,320 parameters**. Batch 8, sequence 16, vocab 64. Adam lr=0.01. 120 training steps.

---

## What the notebook proves

| Experiment | Result |
|------------|--------|
| Tensor shapes | `tokens [8,16]` → `logits [8,16,64]` → scalar loss |
| Grad check (`fc2.bias[0]`) | Numerical **-0.0143** vs autograd **-0.0150** (4.5% rel diff) |
| Accumulation (10 vs 100 tokens) | Wrong **4.196** vs correct **4.210** |
| Grad norm event | Step **91**: norm 0.198→0.172 before loss moves |
| MFU | **~0.03%** (CPU vs 15 TFLOP/s reference) |
| 0.1 in FP32/BF16/FP8 | `0x3DCCCCCD` / `0x3DCC` / `0x2A` |

![Gradient accumulation: wrong vs correct averaging](../../training-loop-truth/gradient_accumulation.png)

---

## Gradient accumulation gap

Micro-batch A has **10** tokens; micro-batch B has **100**. Averaging the two micro-batch means gives equal weight to each — the short batch is overweighted by 10×. The correct loss divides by total tokens (220), not by 2.

---

## MFU honesty

\[
\text{MFU} = \frac{6 \times N \times \text{tokens/sec}}{\text{peak FLOP/s}}
\]

On this toy CPU run: **0.03%**. Not because the formula is wrong — because an 8k-param model on 120 tokens/step never saturates a GPU. Distance to 40%: model too small, batch too small, Python overhead, no fused kernels.

---

## Floating point: 0.1

| Format | Bits (sign \| exp \| frac) | Stored |
|--------|---------------------------|--------|
| FP32 | 0 \| 01111011 \| 10011001100110011001101 | 0.1000000015 |
| BF16 | 0 \| 01111011 \| 1001100 | 0.099609375 |
| FP8 E4M3 | 0 \| 0101 \| 010 | 0.1015625 |

**Train in BF16** with FP32 optimizer states. FP8 for throughput-critical matmuls with scaling — not everywhere.

---

## Reproduce

```bash
git clone https://github.com/curioustushar/blog.git
cd blog/training-loop-truth
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```
