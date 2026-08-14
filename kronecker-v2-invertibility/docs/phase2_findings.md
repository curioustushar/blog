# Phase 2 Initial Findings

**Date:** August 14, 2026  
**Status:** Core architectural problem identified

---

## Executive Summary

**Question:** Can h→κ regression replace vocabulary-sized output head?

**Initial Answer:** NO - not with current algebraic decoder approach.

**Reason:** Fundamental incompatibility between continuous neural predictions and discrete algebraic decoding.

---

## Experiment Design

Minimal proof-of-concept comparing two approaches:

### Approach A: Standard Softmax
```
h ∈ ℝ^d → W_out @ h → logits ∈ ℝ^V → argmax → token
Parameters: V × d
```

### Approach C: Kronecker Regression
```
h ∈ ℝ^d → W_regress @ h → κ_pred ∈ ℝ^D → algebraic_decode → token
Parameters: d × D
```

**Configuration:**
- Vocabulary: 100 tokens
- Hidden dim: 128
- Training: 100 epochs, random sequences

---

## Results

| Metric | Softmax | Kronecker |
|--------|---------|-----------|
| Output parameters | 12,800 | 1,048,576 |
| Final loss | 4.67 | 0.000062 |
| Accuracy | 1.9% | N/A |
| Decode accuracy | N/A | **0.0%** |
| Training time | 2.7s | 73.6s |

### Key Observation

**Low MSE does NOT mean successful decoding!**

MSE = 0.000062 suggests learning occurred, but decode accuracy = 0%.

---

## Diagnostic Analysis

### Test 1: Decoder on Ground-Truth κ
- **Result:** 100% accuracy ✓
- **Conclusion:** Decoder implementation is correct

### Test 2: κ Properties
```
Token    nnz  ||κ||   max
the      3    1.000   0.577
hello    5    1.000   0.447
test     4    1.000   0.500
```
- **Structure:** Sparse (3-7 nonzero), normalized, precise

### Test 3: Noise Sensitivity
```
Token    σ=0.000  σ=0.001  σ=0.010
the      ✓        None     None
hello    ✓        None     None
test     ✓        None     None
```
- **Critical finding:** Even σ=0.001 noise breaks the decoder
- **Implication:** Decoder requires near-perfect predictions

### Test 4: Random Vectors
- **Result:** 0/10 decoded
- **Conclusion:** Decoder rejects non-structured inputs

### Test 5: Near-Zero Predictions
```
Token    MSE        Decoded
the      0.000221   None
hello    0.000222   None
test     0.000220   None
```
- **Critical finding:** Low MSE but no decoding
- **Explanation:** Model learns to predict near-zero vectors (minimizes MSE on sparse targets)

---

## Root Cause Analysis

### Why Decoding Fails

**The algebraic decoder requires:**
1. Sparse structure (few nonzero entries)
2. Precise values (within tight tolerance)
3. Correct positions (specific (byte,position) pairs)

**Neural networks produce:**
1. Dense predictions (continuous-valued)
2. Approximate values (inherent noise)
3. Blurred structure (no sharp sparsity)

**The fundamental mismatch:**

```
True κ(the) = [0, 0, 0, ..., 0.577, 0, ..., 0.577, 0, ..., 0.577, 0, ...]
                               ↑pos=0          ↑pos=1        ↑pos=2

Learned κ  = [0.01, -0.02, 0.03, ..., 0.12, ..., 0.08, ..., 0.05, ...]
              ↑ No clear sparse structure
```

The algebraic decoder cannot identify which entries correspond to actual bytes vs noise.

---

## Why This is a Fundamental Problem

### The Algebraic Decoder Algorithm:
```python
1. Find nonzero indices: where |κ[i]| > tolerance
2. Extract (byte, position) from index: byte = i // pos_dim, pos = i % pos_dim
3. Sort by position
4. Assemble bytes into token
```

**Critical assumption:** Nonzero entries correspond EXACTLY to (byte, position) pairs.

**Reality with neural predictions:** ALL entries have some nonzero value (noise).

### The Tolerance Dilemma:
- **Tolerance too low:** Includes noise, decoder finds invalid structure
- **Tolerance too high:** Misses true signal, decoder finds no structure

**Current tolerance:** 1e-6  
**Typical prediction noise:** 0.01-0.1

The noise is **10,000× larger** than the tolerance!

---

## Parameter Count Paradox

For V=100:
- Standard: 12,800 parameters
- Kronecker: 1,048,576 parameters

**Kronecker is 82× WORSE for small vocabularies!**

The crossover point is approximately V ≈ 10,000:
- V=1,000: Standard wins
- V=10,000: ~Break-even
- V=50,000: Kronecker 6× better
- V=1,000,000: Kronecker 122× better

**Implication:** This approach only makes sense for LARGE vocabularies.

---

## Proposed Solutions

### Solution 1: Learned Decoder (Hybrid Approach)
Instead of algebraic decoding, learn a decoder that can handle continuous κ.

**Architecture:**
```
h → W_regress @ h → κ_pred → Decoder_network → token_logits → softmax
```

**Problems:**
- Defeats the purpose (still need V-sized final layer)
- Just adds extra complexity

**Verdict:** Not promising

---

### Solution 2: Quantization / Sparsification
Force predicted κ to be sparse through:
- Top-k selection
- Thresholding
- Straight-through estimator
- Gumbel-softmax

**Architecture:**
```
h → κ_continuous → sparsify(κ) → κ_sparse → algebraic_decode
```

**Challenges:**
- Non-differentiable (need gradient estimator)
- Doesn't guarantee valid structure
- May need separate loss per dimension

**Verdict:** Worth exploring

---

### Solution 3: Structured Prediction
Predict (byte, position) pairs directly rather than full κ.

**Architecture:**
```
h → byte_classifier (256 classes)
h → position_classifier (32 classes)
h → length_predictor (1-32)
```

**Advantages:**
- Direct prediction of structured information
- Differentiable
- Parameter count: d × (256 + 32 + 32) ≈ d × 320

**Disadvantages:**
- Still needs V-dependent information somehow
- How to know which byte goes where?

**Verdict:** Interesting but unclear

---

### Solution 4: Contrastive / Retrieval Approach
Don't decode at all. Instead, retrieve from precomputed κ bank.

**Architecture:**
```
h → κ_pred
Find nearest: argmin_token ||κ_pred - κ(token)||
```

**Advantages:**
- Doesn't require exact structure
- Works with approximate predictions
- Differentiable (through temperature-scaled softmax)

**Disadvantages:**
- Requires storing/comparing all V tokens
- O(V) inference cost
- Doesn't eliminate vocabulary dependence

**Verdict:** Misses the point

---

### Solution 5: Two-Stage Decoding
Stage 1: Coarse prediction (identify likely tokens)  
Stage 2: Refinement + algebraic decode

**Architecture:**
```
h → κ_coarse → find k nearest tokens → {κ₁, κ₂, ..., κₖ}
h → κ_fine → algebraic_decode with candidates
```

**Advantages:**
- Combines strengths of both
- Can use approximate matching for coarse stage

**Disadvantages:**
- Complex
- Still needs some form of vocabulary search

**Verdict:** Possible but complex

---

## Most Promising Direction

**Solution 2 with auxiliary losses:**

```python
# Predict continuous κ
κ_continuous = W_regress @ h

# Enforce sparsity during training
sparsity_loss = ||κ_continuous||_1  # L1 regularization

# Enforce structure
structure_loss = sum of distances to valid (byte,pos) patterns

# Standard MSE
regression_loss = ||κ_continuous - κ_target||²

# Combined loss
total_loss = regression_loss + λ₁ * sparsity_loss + λ₂ * structure_loss

# At inference: sparsify before decoding
κ_sparse = top_k(κ_continuous, k=token_length)
token = algebraic_decode(κ_sparse)
```

This requires:
1. Learning the correct sparse structure
2. Additional losses to enforce it
3. Careful hyperparameter tuning
4. Top-k or thresholding at inference

**Risk:** May still fail if structure doesn't emerge.

---

## Critical Research Question

**Can a neural network learn to predict DISCRETE, SPARSE, STRUCTURED representations?**

This is fundamentally different from:
- Standard regression (continuous targets)
- Classification (one-hot targets)
- Standard LM prediction (distribution over tokens)

We're asking the network to predict a **structured object** (sparse vector with specific positional encoding) that must satisfy:
- Sparsity: Only k nonzero entries
- Structure: Nonzero indices encode (byte, position) pairs
- Precision: Values must be exact enough for algebraic decode

**This may be asking too much.**

---

## Revised Phase 2 Plan

### Option A: Explore Sparsification (1 week)
- Implement top-k / thresholding
- Add sparsity + structure losses
- Test on small scale
- If works → scale up
- If fails → pivot

### Option B: Hybrid Approach (3 days)
- Add small V-sized classifier on top of κ
- Measure actual parameter savings
- Compare to baseline
- Document findings

### Option C: Alternative Architecture (2 weeks)
- Investigate structured prediction
- Test byte-position direct prediction
- Explore other decompositions
- May require new mathematical framework

---

## Preliminary Verdict

**Question:** Can h→κ regression with algebraic decoder replace V×d output head?

**Answer (preliminary):** NO, not without significant architectural modifications.

**Reason:** Fundamental incompatibility between continuous neural predictions and discrete algebraic decoding.

**Path forward:**
1. Test sparsification approach (high priority)
2. If fails, reconsider entire V2 hypothesis
3. Alternative: Focus on INPUT side efficiency (existing paper)

---

## What We Learned

### Positive:
1. Phase 1 algebraic decoder works perfectly on ground-truth κ ✓
2. κ representation is provably invertible ✓
3. token ↔ κ bidirectionality is solid ✓

### Negative:
1. h→κ regression produces non-structured predictions ✗
2. Algebraic decoder requires precision beyond neural network capability ✗
3. Small-vocab parameter count is WORSE than standard ✗

### Critical Insight:
**The problem is not invertibility (solved in Phase 1).**  
**The problem is learning to produce decodable κ from h.**

This is a **representation learning** problem, not an invertibility problem.

---

## Next Steps

**Immediate (today):**
1. Implement and test sparsification approach
2. Measure decode accuracy with top-k
3. If >80% accuracy → continue
4. If <80% accuracy → document failure and pivot

**Short-term (this week):**
1. If sparsification works: scale to V=1K, 10K, 50K
2. If sparsification fails: investigate hybrid or alternative approaches
3. Document findings in detail

**Phase 2 completion:**
1. Final report answering: "Can we eliminate V×d output head?"
2. If YES: provide working implementation + experiments
3. If NO: explain exactly why and what alternatives exist

---

## Status

**Phase 2 Progress:** 30% complete  
**Time spent:** 3 hours  
**Critical blocker identified:** YES  
**Path forward:** CLEAR (test sparsification)

**Next action:** Implement Solution 2 and test decode accuracy.
