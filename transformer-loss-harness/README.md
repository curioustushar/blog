# Transformer Loss Harness

One notebook, one loss harness, and one thing you have to get right by reading rather than by guessing.

Makes the path from `model(tokens)` → shifted targets → masking → scalar loss fully observable.

**Live write-up:** [blog post](https://curioustushar.github.io/blog/posts/transformer-loss-harness/)  
**GitHub:** [transformer-loss-harness/](https://github.com/curioustushar/blog/tree/master/transformer-loss-harness)  
**Colab (verified):** [Open in Colab](https://colab.research.google.com/drive/1B0FOUHvzvu-mt8Wdswnegv2gL7cvMx3H?usp=sharing)  
**Notebook (GitHub):** [`transformer_loss_harness.ipynb`](./transformer_loss_harness.ipynb)

## What it covers

**Part 1 — Loss harness**
- Tensor shape inspection with per-dimension labels
- Target shift verified with decoded token **strings** (not IDs)
- Padding mask + loss with/without mask + contributing token counts
- Wrong denominator check (divide by valid tokens, not `B×T`)
- Packed-document boundary mask (with `<eos>` separator)
- Perplexity sanity check + within-tokenizer caveat
- Tied vs untied parameter counts
- Ordinary vs chunked cross-entropy — absolute GiB/MiB + ratio

**Part 2 — Two-token-ahead head**
- Head 1: predict `t+1`, Head 2: predict `t+2`
- Separate losses, combined sum, training curves

**Part 3 — Wrong-shift demonstration**
- Deliberate off-by-one in the wrong direction
- Compare scalar losses; strings reveal the bug

## Run locally

```bash
cd transformer-loss-harness
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch matplotlib
python build_notebook.py          # regenerate .ipynb
python run_harness.py             # CPU sanity check
```

Run §1.7 on **Colab GPU** for meaningful VRAM numbers (ordinary ≫ chunked).

## Verified results (Colab)

| Part 1 | Value |
|--------|-------|
| Valid tokens (pad mask) | 20 / 124 |
| Loss without / with pad mask | 5.557 / 5.658 |
| Loss wrong denominator | 5.557 |
| Packed loss before/after boundary | 5.707 / 5.749 |
| Initial perplexity | 286.5 |
| Tied / untied params | 429,312 / 462,080 |
| Memory ordinary / chunked / ratio | 624 B / 1200 B / 0.52 |

| Part 2 (40 steps) | Value |
|-------------------|-------|
| Next-token loss | 0.196 |
| Two-ahead loss | 0.752 |
| Combined | 0.948 |

| Part 3 | Value |
|--------|-------|
| Wrong-shift loss | 5.630 |
| Correct-shift loss | 5.658 |

## Key lesson

> A wrong target shift can produce a beautiful loss curve without raising an exception. Print the strings.
