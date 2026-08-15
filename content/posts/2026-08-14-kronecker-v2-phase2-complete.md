---
title: "Kronecker V2: Phase 2 Complete — Structured Byte Prediction Works"
slug: "kronecker-v2-phase2-complete"
date: 2026-08-14T19:30:00-05:00
categories: ["machine-learning", "research"]
tags: ["embeddings", "kronecker", "structured-prediction", "research", "transformers"]
author: Tushar Gupta
description: "Phase 2 complete: Structured byte prediction achieves 100% exact reconstruction on synthetic tasks, validating feasibility of V×d-free output heads."
---

<div class="post-summary">

**One-Line Summary:** Phase 2 demonstrated feasibility of structured byte prediction as a V×d-free output interface under controlled synthetic experiments; practical language-modeling utility remains unvalidated.

<div class="plan-status" role="status" aria-label="Research status">
  <span class="status-badge status-ok">Phase 2 complete</span>
  <span class="status-badge status-ok">100% exact accuracy</span>
  <span class="status-badge status-ok">Zero p^L degradation</span>
  <span class="status-badge status-pending">Phase 3 designed</span>
</div>

</div>

---

## Update from Phase 0

[Phase 0](/blog/posts/kronecker-v2-invertibility-phase0/) investigated whether raw Kronecker representations could be inverted. We found the encoding was theoretically invertible but identified z-normalization as a bottleneck.

**Phase 2 takes a different approach:** Instead of inverting continuous embeddings, we predict the **discrete byte factors** that construct the Kronecker representation.

This eliminates the continuous→discrete gap that plagued earlier attempts.

---

## Update from Phase 1

[Phase 1](/blog/posts/kronecker-v2-phase1-scale-validation/) addressed the bottlenecks identified in Phase 0 by testing the **complete pipeline** at scale (1K–50K tokens).

| Aspect | Result |
|--------|--------|
| Z-norm preserves uniqueness | ✅ 0 collisions (revised Phase 0 hypothesis) |
| Projection preserves uniqueness | ✅ 0 collisions at d=768 |
| Algebraic decoder (training) | ✅ 100% exact reconstruction |
| Algebraic decoder (unseen) | ✅ 98.9% exact reconstruction |
| Decoder scaling | ✅ O(D) — ~23,440× faster than O(V) search |

**Key insight:** Z-norm is not globally invertible over ℝ^D, but preserves uniqueness over the restricted Kronecker vocabulary subset.

**What Phase 1 established:** `token → κ → decode → token` works reliably at tested scales.

**What Phase 2 asks:** Can a neural network learn `h_t → bytes → κ → token_t` without a V×d output head?

---

## The Core Insight

### Phase 2A: Continuous Regression Failed

**Approach:** `h → κ_pred` (neural network predicts continuous κ)

**Problem:** The algebraic decoder requires **exact** discrete structure (sparse, integer indices). Neural noise destroyed this.

**Result:** ❌ 0% reconstruction accuracy

### Phase 2C: Discrete Factors Succeeded

**Approach:** `h → bytes` (neural network predicts discrete byte classes)

**Advantage:** Predicted bytes **construct** κ deterministically, guaranteeing valid structure.

**Result:** ✅ 100% exact reconstruction

---

## What We Proved

### Phase 2C: Interface is Learnable

**Task:** Predict 8 bytes for each token (fixed length)  
**Model:** Simple linear projection  
**Data:** 50 tokens, deterministic mapping  

**Result:** **100% exact sequence accuracy**

Even the simplest model achieved perfect reconstruction.

### Phase 2D: Scales Across Lengths

**Task:** Test L ∈ {1, 2, 4, 8, 16, 32}  
**Model:** Linear (same architecture)  

**Results:**

| Length | Exact Acc | Convergence |
|--------|-----------|-------------|
| 1      | 100%      | 100 epochs  |
| 2      | 100%      | 100 epochs  |
| 4      | 100%      | 200 epochs  |
| 8      | 100%      | 300 epochs  |
| 16     | 100%      | 400+ epochs |
| 32     | 100%      | 500+ epochs |

**Key finding:** Zero p^L degradation. Byte errors don't compound.

### Phase 2E: Variable-Length Works

**Task:** Mixed lengths (L=1-8) with EOS token  
**Model:** Linear with EOS=256  

**Results:**
- Length prediction: **100%**
- Content prediction: **100%**
- Exact sequences: **100%**

The model learned both **what** bytes to predict and **when** to stop.

### Phase 2F: Transformer Compatible

**Task:** Attach byte head to 2-layer Transformer  
**Result:** Architecturally compatible, codec functional  

No fundamental barriers to integration.

---

## Parameter Comparison

The key advantage: structured bytes scale with L (max token length), not V (vocabulary size).

| Vocabulary | Softmax Params | Structured Bytes | Savings |
|------------|----------------|------------------|---------|
| 10K        | 5.1M           | ~1M              | **5×**  |
| 50K        | 25.6M          | ~1M              | **26×** |
| 100K       | 51.2M          | ~1M              | **51×** |
| 500K       | 256M           | ~1M              | **256×** |
| 1M         | 512M           | ~1M              | **512×** |

For a d=512, L=8 model:
- Softmax: V × 512 parameters
- Structured bytes: 512 × 8 × 257 = 1.05M parameters

**Complexity:** O(L) instead of O(V)

---

## Technical Details

### The Pipeline

```python
# 1. Transformer produces hidden state
h = transformer(tokens)  # (batch, seq, d_model)

# 2. Predict bytes (independent positions)
byte_logits = byte_head(h)  # (batch, seq, L, 257)
# 257 classes: 0-255 (bytes) + 256 (EOS)

bytes_pred = argmax(byte_logits)  # (batch, seq, L)

# 3. Extract sequence (stop at EOS)
for each position:
    find first EOS
    extract bytes before EOS
    
# 4. Construct κ (deterministic)
κ = construct_kappa(bytes)
# Sparse structure: only L non-zeros
# Normalized: 1/√L factor

# 5. Decode token (algebraic)
token = algebraic_decode(κ)
# O(D) time, independent of V
```

### Why It Works

**Discrete predictions guarantee valid structure:**
- Bytes are exact (no rounding errors)
- Indices are exact (no noise)
- κ is always on the "Kronecker manifold"
- Decoder always succeeds

**Continuous regression failed because:**
- Neural outputs have noise
- Even tiny errors destroyed sparse structure
- Decoder couldn't recover from corruption

---

## What We Did NOT Prove

Phase 2 established **feasibility**, not **utility**.

**Still unknown:**
- ❓ Perplexity on real LM tasks (WikiText, etc.)
- ❓ Speed vs softmax baseline (wall-clock time)
- ❓ Memory efficiency in practice
- ❓ Generation quality (BLEU, human eval)
- ❓ Scaling to V > 100K

**These require Phase 3** (PyTorch + GPU + real dataset).

---

## Methodology Highlights

### Clear Scope

We **don't claim:**
- Better than softmax (unproven)
- Production-ready (not tested)
- Practical utility (requires validation)

We **do claim:**
- Feasibility validated (synthetic tasks)
- Parameter scaling eliminated (in principle)
- Interface is learnable (100% accuracy)

**This distinction strengthens the science.**

### Systematic Debugging

When initial attempts showed 0% accuracy, we:
1. Isolated components (codec, extraction, training)
2. Tested with fake logits (perfect inputs)
3. Overfitted single examples
4. Fixed bugs before drawing conclusions

**Result:** "0% accuracy" was implementation bugs, not architectural failure.

### Pre-Registration

Phase 3 protocol specified **before** experiments:
- Dataset, metrics, success criteria
- Prevents p-hacking
- Ensures fair comparison

---

## Phase 3: Designed, Not Executed

### Three-Stage Plan

**3A: Experimental Specification** ✅
- WikiText-103 benchmark
- Matched Transformer backbone  
- Pre-specified success criteria

**3B: PyTorch Proxy** ⏳
- Reproduce Phase 2 in PyTorch
- 6 validation tests
- Catch bugs before expensive training

**3C: Full Benchmark** ⏳
- Softmax vs structured bytes
- Perplexity, memory, throughput
- Vocabulary scaling (10K → 100K)

### Success Criteria

**Strong success:**
- Test perplexity ≤ 102% of softmax
- Memory ≤ 90% OR Throughput ≥ 90%

**Partial success:**
- Test perplexity ≤ 105%
- Parameter savings realized, mixed efficiency

**Failure:**
- Test perplexity > 105%
- OR Resources worse despite fewer parameters

### Timeline

- Phase 3B: 3-5 days (PyTorch validation)
- Phase 3C: 5-8 weeks (full training)
- **Total: ~2 months**

### Requirements

- PyTorch 2.0+
- GPU (A100 or equivalent)
- WikiText-103 dataset

---

## Key Insights

### 1. Discrete > Continuous

Predicting discrete factors that construct κ eliminates noise problems.

This is a general principle: when your decoder needs exact structure, predict the factors, not the structure.

### 2. No p^L Degradation

Byte errors are independent. 100% byte accuracy → 100% exact accuracy.

This was surprising! We expected compound errors across positions.

### 3. Simple Models Work

Linear projections achieved perfect results. No fancy architectures needed.

**Takeaway:** The right interface matters more than model complexity.

### 4. Variable-Length is Learnable

EOS mechanism worked perfectly. The model learned both content and termination.

---

## Reproducibility

All Phase 2 code runs in < 5 minutes on CPU:

```bash
cd kronecker-v2-invertibility/experiments/phase2

python phase2c_minimal.py      # 30 seconds - validation
python phase2d_ultra_fast.py   # 80 seconds - scaling
python phase2e_fixed.py        # 30 seconds - variable-length
```

**Expected output:** 100% exact accuracy on all tests.

**Requirements:** Python 3.8+, numpy

---

## Limitations

### Phase 2 Limitations

- Synthetic tasks only (not real LM)
- Small scale (V=50, L≤32)
- Simple models (linear, 2-layer MLP)
- numpy implementation (no GPU)

### Requires Phase 3

- Real language modeling benchmarks
- Perplexity comparison vs softmax
- Computational efficiency measurement  
- Large vocabulary scaling validation

---

## Related Work

### Parameter-Efficient LMs

- **Adaptive Softmax** (Grave et al., 2017) - Hierarchical softmax with varying capacity
- **Mixture of Softmaxes** (Yang et al., 2018) - Multiple specialized output heads
- **Character-level models** (Kim et al., 2016) - Eliminate vocabulary entirely

**Our contribution:** Structured prediction with algebraic decoding.

### Structured Prediction

- **CTC** (Graves et al., 2006) - Sequence prediction with alignment
- **Pointer Networks** (Vinyals et al., 2015) - Output as input selection
- **Copy Mechanisms** (Gu et al., 2016) - Hybrid generation/copying

**Our difference:** Predict factors that construct structured representations.

---

## Implications (If Phase 3 Succeeds)

### For Research

- Alternative to vocabulary-sized heads
- Insights into discrete vs continuous prediction
- Structured output representations

### For Practice

- Multilingual models (100K+ tokens)
- Memory-constrained deployment
- Faster model loading
- Parameter-efficient fine-tuning

### For Theory

- Discrete factors > continuous regression
- Algebraic decoders for structured outputs
- When to use classification vs regression

---

## FAQ

**Q: Should I use this instead of softmax?**  
A: Not yet. Wait for Phase 3 validation on real tasks.

**Q: Does it work for large vocabularies?**  
A: Theoretically yes (savings scale with V). Empirically untested beyond V=50.

**Q: What about generation quality?**  
A: Unknown. Phase 2 tested reconstruction, not generation.

**Q: Can I reproduce the results?**  
A: Yes! All code runs in < 5 minutes. See repository.

**Q: Why didn't you test on real LMs?**  
A: Phase 2 focused on feasibility validation. Phase 3 will test utility.

---

## Next Steps

### For This Project

1. Implement Phase 3B (PyTorch proxy)
2. Execute Phase 3C (WikiText-103 benchmark)
3. Analyze results
4. Write paper (if successful)

### For the Community

**If you're interested in:**
- PyTorch implementation
- Benchmark execution
- Optimization improvements
- Additional experiments

**Wait for Phase 3 specs** or contribute to validation.

---

## Conclusion

**Phase 2 Verdict:** Feasibility validated.

Structured byte prediction:
- ✅ Works in principle (100% accuracy)
- ✅ Scales across lengths (L=1 to 32)
- ✅ Handles variable length (EOS functional)
- ✅ Transformer-compatible
- ✅ Parameter-efficient (5× to 512× savings)

**But:** Practical utility for real language modeling remains an open empirical question requiring Phase 3 validation.

**Either outcome is valuable science.**

If it works in practice → new parameter-efficient architecture  
If it doesn't → well-documented negative result with clear failure modes

---

## Documentation

**Full technical report:** [Phase 2 Summary](https://github.com/curioustushar/kronecker-v2-invertibility/blob/main/docs/phase2_final_summary.md)

**Phase 3 protocol:** [Phase 3 Roadmap](https://github.com/curioustushar/kronecker-v2-invertibility/blob/main/docs/phase3_roadmap.md)

**Project status:** [Complete Overview](https://github.com/curioustushar/kronecker-v2-invertibility/blob/main/PROJECT_STATUS.md)

---

## Citation

```bibtex
@misc{gupta2026kronecker_phase2,
  title={Structured Byte Prediction for Parameter-Efficient Language Models:
         Phase 2 Feasibility Study},
  author={Gupta, Tushar},
  year={2026},
  howpublished={\url{https://curioustushar.github.io/blog/}},
  note={Phase 2: Feasibility validated under synthetic conditions}
}
```

---

**Status:** Phase 2 complete (feasibility validated), Phase 3 designed (not executed)

**Repository:** [kronecker-v2-invertibility](https://github.com/curioustushar/kronecker-v2-invertibility)

**Next milestone:** Phase 3B PyTorch proxy implementation

---

**This is active research.** Phase 2 established feasibility. Phase 3 will determine practical utility.

**Series:** [Phase 0](/blog/posts/kronecker-v2-invertibility-phase0/) → [Phase 1](/blog/posts/kronecker-v2-phase1-scale-validation/) → [Phase 2](/blog/posts/kronecker-v2-phase2-complete/) (this post)
