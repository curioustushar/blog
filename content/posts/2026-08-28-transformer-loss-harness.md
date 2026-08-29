---
title: "Transformer Loss Harness"
slug: "transformer-loss-harness"
date: 2026-08-28T18:00:00-05:00
categories: ["machine-learning"]
tags:
  ["transformers", "loss", "cross-entropy", "perplexity", "notebook"]
author: Tushar Gupta
description: "One notebook, one loss harness — every step from model output to scalar loss, verified with decoded token strings."
---

<div class="post-summary">

The few lines between `model(tokens)` and `cross_entropy(...)` are where serious training bugs live. They do not always raise an exception. They often produce a **beautiful loss curve**.

This post accompanies a Colab notebook that makes every step — shift, padding mask, document boundary, perplexity, tied weights, chunked CE memory, and a two-token-ahead head — **observable and assertable**.

<div class="plan-status" role="status" aria-label="Project status">
  <span class="status-badge status-ok">Part 1: 7 harness checks ✓</span>
  <span class="status-badge status-ok">Part 2: dual-head training ✓</span>
  <span class="status-badge status-ok">Part 3: wrong-shift demo ✓</span>
</div>

</div>

---

## Links

| | URL |
|---|---|
| **Colab (verified run)** | [Open in Colab](https://colab.research.google.com/drive/1B0FOUHvzvu-mt8Wdswnegv2gL7cvMx3H?usp=sharing) |
| **GitHub** | [transformer-loss-harness/](https://github.com/curioustushar/blog/tree/master/transformer-loss-harness) |
| **Full write-up** | [docs/transformer-loss-harness-writeup.md](https://github.com/curioustushar/blog/blob/master/docs/transformer-loss-harness-writeup.md) |

---

## Configuration

| Setting | Value |
|---------|-------|
| `vocab_size` | 256 |
| `hidden_dim` | 128 |
| `num_layers` | 2 |
| `num_heads` | 4 |
| `batch_size` | 4 |
| `seq_len` | 32 |
| `train_steps` | 40 |
| `lr` | 3e-4 |
| `seed` | 42 |

---

## The harness

```python
hidden = model(tokens)
logits = output_head(hidden)
loss = cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    tokens[:, 1:].reshape(-1),
)
```

Seven checks in Part 1:

1. **Shapes** — every tensor, every dimension labeled  
2. **Strings** — inputs and targets decoded side by side (not IDs)  
3. **Padding mask** — contributing token count + loss with/without mask  
4. **Wrong denominator** — divide by valid tokens, not `B×T`  
5. **Document boundary** — packed sequences mask the cross-doc prediction  
6. **Perplexity** — untrained ppl ≈ vocab size (within-tokenizer only)  
7. **Memory** — ordinary full `log_softmax` vs chunked CE, absolute GiB/MiB + ratio  

---

## Part 1 results (verified Colab run)

| # | Check | Result |
|---|-------|--------|
| 1 | Tensor shapes | `tokens [4,32]` → `shifted_logits [124,256]` |
| 2 | Valid tokens (pad mask) | **20** of 124 |
| 3 | Loss without / with pad mask | **5.557 / 5.658** |
| 3b | Loss wrong denominator (`B×T`) | **5.557** |
| 4 | Packed loss before / after boundary | **5.707 / 5.749** |
| 5 | Initial perplexity | **286.5** (vocab 256, ratio 1.12) |
| 6 | Tied / untied params | **429,312 / 462,080** (diff 32,768) |
| 7 | Memory ordinary / chunked / ratio | **624 B / 1200 B / 0.52** |

Pad mask changes the mean (5.557 → 5.658) because only 20 of 124 positions contribute. Wrong denominator (5.557) matches the unmasked mean — dividing by `B×T` instead of valid tokens.

### Boundary mask

Packed with `<eos>` between documents. Without masking, the model learns **`<eos>` → `The'`** — end of document A predicting start of document B.

### Perplexity caveat

Perplexity is a good **within-tokenizer** training signal. Comparing perplexity across different tokenizers is misleading — use bits-per-byte or bits-per-character for cross-tokenizer comparison.

### Memory (§1.7)

Verified run: ordinary **624 B**, chunked **1200 B**, ratio **0.52** (tracemalloc on CPU fallback workload). For GiB-scale VRAM numbers, set **Runtime → Change runtime type → GPU** and re-run §1.7 — ordinary should dominate chunked when materializing `[tokens × vocab]` log-probs.

---

## Part 2 — two-token-ahead head

After 40 training steps:

| Loss | Value |
|------|-------|
| Next-token (t+1) | **0.196** |
| Two-ahead (t+2) | **0.752** |
| Combined (sum) | **0.948** |

![Dual-head training loss over 40 steps](../../transformer-loss-harness/dual-head-training.png)

**Head 1** drops fast. **Head 2** stays higher throughout — predicting `t+2` needs longer-range structure. **Combined** tracks the sum.

---

## Part 3 — wrong-shift demonstration

```
WRONG shift (position t predicts t-1):
Input tokens : ['cat', 'sat', 'on', 'the', 'mat', '.']
Target tokens: ['The', 'cat', 'sat', 'on', 'the', 'mat']

Position | Wrong input | Wrong target
       0 | cat         | The
       1 | sat         | cat
       ...
```

| Loss | Value |
|------|-------|
| Wrong-shift | **5.630** |
| Correct-shift | **5.658** |

Shapes match. Both losses are finite and close — only the **strings** reveal that position *t* is predicting *t−1* instead of *t+1*.

---

## The lesson

> Print the strings. Many serious training bugs live in the few lines between the model output and the scalar, and they do not always raise an exception.

Three silent failures the harness catches:

1. **Wrong shift** — beautiful loss curve, model learns to copy  
2. **Wrong denominator** — loss scaled by padding fraction, changes every batch  
3. **Missing boundary mask** — model learns unrelated documents are adjacent  

---

## Reproduce

```bash
git clone https://github.com/curioustushar/blog.git
cd blog/transformer-loss-harness
python build_notebook.py
python run_harness.py
```

Or open the [verified Colab notebook](https://colab.research.google.com/drive/1B0FOUHvzvu-mt8Wdswnegv2gL7cvMx3H?usp=sharing) and run top to bottom.
