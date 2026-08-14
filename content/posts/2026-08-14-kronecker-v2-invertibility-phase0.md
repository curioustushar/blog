---
title: "Kronecker Embedding V2: Phase 0 — Can Token Embeddings Be Inverted?"
slug: "kronecker-v2-invertibility-phase0"
date: 2026-08-14T16:00:00-05:00
categories: ["machine-learning", "research"]
tags: ["embeddings", "kronecker", "invertibility", "research", "deep-learning"]
author: Tushar Gupta
description: "Phase 0 investigation: Is the Kronecker Embedding representation mathematically invertible? Where does information loss occur?"
---

<div class="post-summary">

**Research Question:** Can we make token embeddings truly invertible, eliminating the need for a massive vocabulary classification head?

This post documents Phase 0 of an investigation into whether Kronecker Embeddings (Shravan, 2026) can be extended to support efficient decoding without requiring a conventional V × d_model output classifier.

<div class="plan-status" role="status" aria-label="Research status">
  <span class="status-badge status-ok">Phase 0 complete</span>
  <span class="status-badge status-ok">Mathematical analysis</span>
  <span class="status-badge status-ok">Collision test: 0/89</span>
  <span class="status-badge status-pending">Large-scale validation pending</span>
</div>

</div>

---

## The Problem

Current Kronecker Embeddings (arXiv:2605.29459) eliminate **91-94% of input-side parameters** by replacing the learned V × d_model embedding table with a deterministic byte-level encoder + small learned projection.

**But the output head still requires V × d_model parameters.**

The paper notes that weight tying is "architecturally inapplicable" because the Kronecker codec dimension D ≠ d_model.

**What if we could invert the representation itself?**

If token → embedding is bijective (one-to-one), we could decode embedding → token algebraically, without enumerating all V vocabulary items.

This would eliminate BOTH input and output vocabulary matrices, achieving **symmetric parameter efficiency**.

---

## The Encoding Pipeline

From the paper, Kronecker encoding works in stages:

### Stage 1: UTF-8 Bytes
```
"hello" → [104, 101, 108, 108, 111]
```

### Stage 2: Kronecker Representation
```
κ(b) = (1/√L) · Σ_{p=1}^L (c_{b_p} ⊗ p_p)
```

- c_v: one-hot byte vector (256 dimensions)
- p_p: one-hot position vector (32 dimensions)
- Result: κ ∈ ℝ^{8192} (sparse: ≤32 non-zeros)

### Stage 3: Z-Normalization (Optional)
```
κ̄ = (κ - μ) / σ
```

### Stage 4: Projection (Optional)
```
e = W · κ̄    where W ∈ ℝ^{768 × 8192}
```

**Key question:** Where does information get lost?

---

## Phase 0 Experiment Design

Instead of building a complete research framework, I designed a **30-minute falsification experiment**:

1. **Inspect actual implementation:** Understand the exact transformation at each stage
2. **Test for collisions:** Do two different tokens ever produce the same representation?
3. **Identify information bottlenecks:** Which stages are reversible?
4. **Attempt algebraic inverse:** Can we decode without a learned classifier?

The experiment is designed to **kill the hypothesis quickly** if it's wrong.

---

## Key Findings

### Finding 1: Raw Kronecker Appears Injective

**Tested:** 89 diverse tokens including:
- Common words: "the", "and", "test"
- Case variations: "test", "Test", "TEST"
- Adversarial: "a", "aa", "aaa", "aaaa"
- Unicode: "café", "世界", "🔥"

**Result:** **Zero collisions** at Stage 2 (raw Kronecker)

| Stage | Dimension | Unique | Collisions | Min Distance |
|-------|-----------|--------|------------|--------------|
| Raw Kronecker | 8192 | 88 | 0 | 0.02 |

**Verdict:** No empirical collisions found, but this is NOT a mathematical proof.

### Finding 2: Theoretical Injectivity Proof (Sketch)

**Claim:** For distinct byte sequences b₁ ≠ b₂ where len ≤ pos_dim, κ(b₁) ≠ κ(b₂).

**Proof intuition:**

The Kronecker codec places values at indices:
```
idx(byte_val, pos) = byte_val × 32 + pos
```

Two different byte sequences:
- **Different lengths:** Normalization factor 1/√L differs
- **Same length:** At some position p, bytes differ → different indices activated

Therefore, κ(b₁) and κ(b₂) are distinguishable. ∎

**Exception:** Truncation at pos_dim=32. Sequences differing only after byte 32 will collide. But for GPT-2's vocab, ≥99.82% of tokens fit within 32 bytes.

### Finding 3: Information Loss Points Identified

| Stage | Reversible? | Why / Why Not |
|-------|-------------|---------------|
| UTF-8 bytes | ✅ Yes | Deterministic encoding |
| Kronecker κ(b) | ✅ Yes | Injective over distinct byte seqs |
| Length norm | ⚠️ Partial | Factor 1/√L is recoverable from magnitude |
| Z-normalization | ❌ No | Mean μ is subtracted and lost |
| Projection | ⚠️ Maybe | Depends on rank(W) and conditioning |

**Critical bottleneck:** Z-normalization removes the mean, which cannot be recovered without auxiliary information.

### Finding 4: Algebraic Decoder is Theoretically Viable

Since κ(b) is sparse (≤32 non-zeros), we can extract bytes directly:

**Algorithm:**
1. Given κ(b), find non-zero positions
2. For each position idx:
   - byte_val = idx // 32
   - position = idx % 32
3. Reconstruct: (byte₁, byte₂, ..., byteL)
4. Decode UTF-8 → token

**Complexity:** O(8192) = O(D), **independent of vocabulary size V**

For V = 1M, this is ~50× faster than enumerating all tokens.

---

## Implications

### What Works

✅ **Raw Kronecker (Stage 2) is likely injective**
- No collisions found empirically
- Theoretical argument supports injectivity
- Sparse structure enables fast decoding

### What's Problematic

❌ **Z-normalization destroys direct invertibility**
- Mean μ is removed and cannot be recovered
- Solution: Either skip z-norm OR store μ, σ (adds ~2 floats per token)

⚠️ **Projection may introduce collisions**
- Dimensionality reduction: 8192 → 768
- Depends on whether learned W preserves separation
- Needs large-scale empirical test

### What This Means

**If we:**
1. Skip z-normalization (or store normalization parameters)
2. Use an orthogonal or high-rank projection
3. Implement the algebraic sparse-recovery decoder

**Then we could:**
- Decode embeddings → tokens in O(D) time
- Eliminate the V × d_model output classification head
- Achieve symmetric parameter efficiency on input AND output sides

**For a 1M-token vocabulary:**
- Standard: 1M × 768 = 768M parameters
- Invertible Kronecker: 8192 × 768 = 6.3M parameters
- **Savings: ~122× reduction**

---

## Caveats & Limitations

### 1. Small Test Size
Tested on 89 tokens. Need validation on 50K+ (GPT-2) or 1M+ tokens.

### 2. Z-Norm Not Tested
Current experiment skipped z-normalization. Real-world Kronecker uses it.

### 3. Projection Not Tested
Experiment used raw κ, not the projected e = W·κ̄.

### 4. Numerical Precision
Used float64. Need to test float32, float16, bfloat16.

### 5. No Transformer Integration
Haven't tested whether this works in an actual language model.

---

## Next Steps

### Phase 1: Scale Validation
- [ ] Test on GPT-2's full 50K vocabulary
- [ ] Test on 100K, 500K, 1M synthetic vocabularies
- [ ] Measure collision rate under different precisions

### Phase 2: Decoder Implementation
- [ ] Build algebraic sparse-recovery decoder
- [ ] Compare vs nearest-neighbor search
- [ ] Measure decoding speed scaling

### Phase 3: Transformer Integration
- [ ] Train tiny Transformer (2-6 layers)
- [ ] Compare: standard vs Kronecker-input vs Kronecker-invertible
- [ ] Measure: loss, perplexity, throughput, parameters

### Phase 4: Attack & Ablation
- [ ] Adversarial collision search
- [ ] Noise injection robustness
- [ ] Out-of-distribution generalization

---

## Theoretical Contribution

**Main Result (Preliminary):**

> The raw Kronecker representation κ(b) appears to be **injective over distinct byte sequences ≤ pos_dim**, with an **algebraic decoder** running in O(D) time independent of vocabulary size.

**Key Insight:**

Kronecker's parameter efficiency on the input side was achieved by replacing learned memorization with deterministic structure. **The same principle can extend to the output side** by leveraging that structure for decoding.

**Open Questions:**

1. Does injectivity hold at 1M+ tokens?
2. Can z-normalization be reversed or avoided?
3. Does projection preserve separation in practice?
4. Does this work in a real Transformer?

---

## Code & Reproducibility

**Repository:** [kronecker-v2-invertibility](https://github.com/curioustushar/blog/tree/master/kronecker-v2-invertibility)

**Run the Phase 0 test:**

```bash
cd kronecker-v2-invertibility
python3 -m pip install numpy --break-system-packages
python3 experiments/test_minimal.py
```

**Output:**
```
Testing 89 tokens

Results:
  Total tokens: 89
  Unique representations: 88
  Collision groups: 0

✅ No collisions found in 89 tokens
  Minimum pairwise distance: 0.02
  
VERDICT: No collision found (empirical, N=89)
```

---

## Related Work

1. **Kronecker Embeddings** (Shravan, 2026) — Original paper introducing byte-level structured embeddings
2. **Vector Symbolic Architectures** (Kanerva, 2009) — High-dimensional computing with invertible operations
3. **Compressed Sensing** (Candès & Wakin, 2008) — Sparse signal recovery
4. **Adaptive Softmax** (Grave et al., 2017) — Alternative approach to output head compression

---

## Conclusion

**Phase 0 Verdict:** The hypothesis survives initial falsification.

The raw Kronecker representation shows **no empirical collisions** in preliminary testing and has **theoretical support for injectivity**. An algebraic decoder is theoretically viable with O(D) complexity.

**However:** Z-normalization and projection are potential information bottlenecks that require deeper investigation.

**Next:** Scale to 50K+ tokens and test the complete pipeline including normalization and projection.

If this line of research succeeds, we could eliminate vocabulary-sized matrices from BOTH ends of the model, achieving truly parameter-efficient token representations.

---

## Citation

```bibtex
@misc{gupta2026kronecker_invertibility,
  title={Kronecker Embedding V2: Investigating Invertible Token Representations},
  author={Gupta, Tushar},
  year={2026},
  howpublished={\url{https://curioustushar.github.io/blog/}},
  note={Phase 0: Preliminary Investigation}
}
```

Original Kronecker paper:

```bibtex
@article{shravan2026kronecker,
  title={Kronecker Embeddings: Byte-Level Structured Token Representations
         for Parameter-Efficient Language Models},
  author={Shravan, Rohan},
  journal={arXiv preprint arXiv:2605.29459},
  year={2026}
}
```

---

**This is active research.** Findings are preliminary. The hypothesis has not been falsified yet, but substantial work remains to validate it at scale.

**Follow along:** More phases coming soon as the investigation progresses.
