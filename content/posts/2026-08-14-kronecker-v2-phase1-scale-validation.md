---
title: "Kronecker V2: Phase 1 — Scale Validation & Algebraic Decoding"
slug: "kronecker-v2-phase1-scale-validation"
date: 2026-08-14T17:30:00-05:00
categories: ["machine-learning", "research"]
tags: ["embeddings", "kronecker", "invertibility", "research", "deep-learning"]
author: Tushar Gupta
description: "Phase 1: Does Kronecker injectivity hold at scale through z-norm and projection? We implement an algebraic decoder and benchmark O(D) decoding vs O(V) search."
---

<div class="post-summary">

**One-Line Summary:** Phase 1 validated the complete Kronecker pipeline (raw κ + z-norm + projection) at 1K–50K tokens, implemented an algebraic decoder with 98.9% unseen-token reconstruction, and confirmed O(D) decoding ~23,440× faster than vocabulary search.

<div class="plan-status" role="status" aria-label="Research status">
  <span class="status-badge status-ok">Phase 1 complete</span>
  <span class="status-badge status-ok">0 collisions at 50K</span>
  <span class="status-badge status-ok">Algebraic decoder works</span>
  <span class="status-badge status-pending">LM integration pending</span>
</div>

</div>

---

## Update from Phase 0

[Phase 0](/blog/posts/kronecker-v2-invertibility-phase0/) asked whether Kronecker embeddings are invertible. On 89 tokens, raw κ showed zero collisions and an algebraic decoder looked theoretically viable.

**But Phase 0 left three open questions:**

1. Does injectivity hold at real vocabulary scales (50K+)?
2. Do z-normalization and projection destroy uniqueness?
3. Does the algebraic decoder work on **unseen** tokens, or only memorized vocabulary?

**Phase 1 answers all three** — with one important correction to the Phase 0 hypothesis.

---

## What We Tested

Phase 1 extended Phase 0 by testing the **complete pipeline**:

```
token → UTF-8 bytes → κ(b) → z-norm → projection W·κ̄ → e
```

### Configuration

```python
encoder = KroneckerEncoderStaged(
    char_dim=256,
    pos_dim=32,
    d_model=768,
    apply_length_norm=True,
    apply_z_norm=True,        # Phase 0: untested
    apply_projection=True,    # Phase 0: untested
    projection_seed=42,
)
```

**Scales tested:** 1K, 10K, 50K tokens  
**Precision:** float64, 12 decimal places for collision detection  
**Seeds:** 42 (training vocab), 99 (unseen tokens)

---

## Key Finding 1: Z-Norm Does NOT Break Uniqueness

### Phase 0 Hypothesis (Revised)

**Phase 0 claimed:** Z-normalization destroys invertibility because the mean μ is subtracted and lost.

**Phase 1 finding:** Z-norm does **not** introduce collisions over finite Kronecker vocabularies.

| Stage | Dimension | Unique (1K) | Collisions |
|-------|-----------|-------------|------------|
| Raw Kronecker | 8,192 | 1,000 | 0 |
| Z-normalized | 8,192 | 1,000 | 0 |
| Projected (d=768) | 768 | 1,000 | 0 |

**Why it still works:** Z-norm is not globally invertible over arbitrary ℝ^D vectors. But Kronecker representations form a **restricted subset**:

- Sparse (≤32 non-zeros)
- Fixed magnitude pattern (1/√L per entry)
- Fixed index pattern (`byte_val × 32 + position`)

Within this domain, z-norm preserves uniqueness. We do **not** need to skip z-norm or store auxiliary μ, σ parameters.

---

## Key Finding 2: Projection Preserves Separation

**Concern from Phase 0:** Projecting 8,192 → 768 dimensions might cause collisions.

**Result:** Zero collisions at all tested scales. Minimum pairwise distance actually **increased** after projection (1.00 → 25.5 at 1K scale).

```
Projection matrix: W ∈ ℝ^{768 × 8192}
Rank: 768 (full rank)
Condition number: ~O(10²)
```

The effective dimensionality of Kronecker representations is far below 8,192 (sparse, structured), so projection to d=768 preserves full separation.

---

## Key Finding 3: Algebraic Decoder Works

### The Algorithm

Since κ(b) is sparse, we extract bytes directly from non-zero positions:

```python
def algebraic_decode_raw_kronecker(kappa, char_dim=256, pos_dim=32):
    nonzero_indices = np.where(np.abs(kappa) > tol)[0]
    for idx in nonzero_indices:
        byte_val = idx // pos_dim
        position = idx % pos_dim
        byte_position_pairs.append((position, byte_val))
    byte_position_pairs.sort()
    return bytes([b for (_, b) in byte_position_pairs])
```

**Complexity:** O(D) = O(8,192), **independent of vocabulary size V**.

### Training Vocabulary

| Vocab Size | Exact Reconstruction | Mean Decode Time |
|------------|----------------------|------------------|
| 1,000 | 1,000 / 1,000 (100%) | 23.6 μs |

### Unseen Tokens (Never in Training Vocab)

**Test:** 178 tokens never seen during vocabulary construction — random strings, uncommon words, 100+ char strings, Unicode ("münchen", "東京", "🚀").

| Metric | Result |
|--------|--------|
| Exact reconstructions | 176 / 178 (98.9%) |
| Correct after truncation | 2 (1.1%) |
| Failures | 0 (0.0%) |

**Interpretation:** The decoder uses **mathematical structure**, not memorization. This is structural invertibility — it works on arbitrary byte strings within the codec's constraints.

---

## Key Finding 4: O(D) Decoding vs O(V) Search

We benchmarked algebraic decoding against brute-force nearest-neighbor vocabulary search:

| Vocab Size | Algebraic (μs) | Search (μs) | Speedup |
|------------|----------------|-------------|---------|
| 100 | 28.3 | 2,054 | 72× |
| 1,000 | 35.1 | 20,152 | 575× |
| 10,000 | 32.5 | 204,177 | 6,284× |
| **50,000** | **41.2** | **964,812** | **23,440×** |

**Algebraic decoder:** ~35 μs mean, approximately constant across vocabulary sizes.  
**Vocabulary search:** O(V) — linear scaling confirmed.

At V=50K, algebraic decoding is **23,440× faster** than enumerating the full vocabulary.

---

## What Phase 1 Proved — and What It Didn't

### Proven ✅

```
token → κ → algebraic_decode → token
```

| Claim | Status |
|-------|--------|
| Injectivity at 1K–50K | ✅ 0 collisions |
| Z-norm preserves uniqueness | ✅ Confirmed |
| Projection preserves uniqueness | ✅ Confirmed |
| Algebraic decoder on training tokens | ✅ 100% |
| Algebraic decoder on unseen tokens | ✅ 98.9% |
| O(D) decoding independent of V | ✅ ~35 μs constant |

### Not Proven ⚠️

```
h_t → κ(token_t) → token_t
```

**The architectural question Phase 1 explicitly leaves open:**

A language model produces `h ∈ ℝ^{d_model}`, not a Kronecker representation. Standard decoding:

```
h → W_out @ h → softmax(V logits) → token     (W_out ∈ ℝ^{V × d})
```

**Can we replace this with:**

```
h → ??? → κ(token) → algebraic_decode → token
```

**Without vocabulary-sized parameters?**

Phase 1 proves the **codec** works. It does **not** prove the **output head** can be eliminated. That is Phase 2's question.

---

## Phase 1 Verdict

**Status:** ✅ **GREEN — Strong**

The Kronecker representation (including z-norm and projection) is practically injective at tested scales. Algebraic decoding is viable, fast, and structurally grounded.

**Important correction:** Phase 0's z-norm concern was overstated for finite Kronecker vocabularies. The complete pipeline works.

**Honest boundary:**

| Problem | Status |
|---------|--------|
| Token representation decoding | ✅ Solved |
| Language model output prediction | ⚠️ Unsolved (Phase 2) |

---

## Limitations

- **Scale:** Tested up to 50K tokens; 100K–1M pending
- **Precision:** float64 only; float32/float16 not yet tested
- **Proof:** Injectivity is empirical + sketch, not formal
- **LM integration:** Not tested — no Transformer, no perplexity

---

## Code & Reproducibility

**Repository:** [kronecker-v2-invertibility](https://github.com/curioustushar/kronecker-v2-invertibility)

```bash
cd kronecker-v2-invertibility
python3 experiments/phase1_scale_validation.py   # Collision tests at scale
python3 experiments/test_unseen_tokens.py        # Unseen token decoding
python3 experiments/benchmark_decoder_scaling.py   # O(D) vs O(V) benchmark
```

**Full technical report:** [phase1_report.md](https://github.com/curioustushar/kronecker-v2-invertibility/blob/main/docs/phase1_report.md)

---

## What Happened Next

**[Phase 2](/blog/posts/kronecker-v2-phase2-complete/)** tackled the unresolved architectural question: can a neural network learn `h_t → bytes → κ → token_t` without a V×d output head?

**Spoiler:** Structured byte prediction works on synthetic tasks. Practical LM utility remains Phase 3.

---

## Citation

```bibtex
@misc{gupta2026kronecker_phase1,
  title={Kronecker Embedding V2: Scale Validation and Algebraic Decoding},
  author={Gupta, Tushar},
  year={2026},
  howpublished={\url{https://curioustushar.github.io/blog/}},
  note={Phase 1: Injectivity validated at 1K-50K scale}
}
```

---

**Status:** Phase 1 complete (codec validated at scale), Phase 2 complete (structured prediction feasible)

**Series:** [Phase 0](/blog/posts/kronecker-v2-invertibility-phase0/) → [Phase 1](/blog/posts/kronecker-v2-phase1-scale-validation/) (this post) → [Phase 2](/blog/posts/kronecker-v2-phase2-complete/)
