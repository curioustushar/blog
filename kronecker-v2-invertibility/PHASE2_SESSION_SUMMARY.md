# Phase 2 Session Summary - August 14, 2026

## Objective
Determine if invertible Kronecker representation can replace vocabulary-sized output head in language models.

**Core Question:** Can `h_t → κ(token_t)` work without V×d parameters?

---

## What We Built Today

### 1. Tiny Transformer Architecture ✓
- Clean numpy implementation for clarity
- 4 layers, 512-dim, 8 heads, 2048 FFN
- ~18M parameters (backbone)
- Supports both standard and Kronecker input embeddings

**File:** `experiments/phase2/tiny_transformer.py`

### 2. Three Output Head Variants ✓
- **Standard Softmax:** h → W_out @ h → logits (V×d params)
- **Kronecker Regression:** h → W_regress @ h → κ_pred → decode (d×D params)
- **Hybrid:** Both softmax and Kronecker (for future experiments)

**File:** `experiments/phase2/output_heads.py`

### 3. Minimal Training Experiment ✓
- Proof-of-concept comparing softmax vs Kronecker regression
- 100-token vocabulary, 128-dim hidden, 100 epochs
- Full backpropagation implementation

**File:** `experiments/phase2/train_minimal.py`

### 4. Decoder Diagnostics ✓
- Systematic testing of algebraic decoder behavior
- Noise sensitivity analysis
- Ground-truth vs learned predictions

**File:** `experiments/phase2/diagnose_decoder.py`

---

## Critical Discovery

### The Fundamental Problem

**Algebraic decoder requires:**
- Precise sparse structure
- Exact (byte, position) encoding
- Tolerance: ~1e-6

**Neural networks produce:**
- Continuous predictions
- Inherent noise: ~0.01-0.1
- No guaranteed sparsity

**Result:** 0% decode accuracy despite MSE=0.000062

### Diagnostic Results

| Test | Result | Implication |
|------|--------|-------------|
| Decoder on ground-truth κ | 100% ✓ | Decoder implementation correct |
| Decoder on learned κ | 0% ✗ | Predictions not structured enough |
| Noise sensitivity (σ=0.001) | Fails | Requires near-perfect precision |
| Random vectors | All fail | Decoder rejects non-structured inputs |
| Near-zero predictions | MSE low, decode fails | Low MSE ≠ decodable |

### The Mismatch

```
True κ("the"):    [0, 0, ..., 0.577, 0, ..., 0.577, 0, ...]
                                ↑              ↑
                           (byte=t, pos=0) (byte=h, pos=1)

Learned κ:        [0.01, -0.02, 0.03, ..., 0.12, 0.08, 0.05, ...]
                   ↑ All entries have some value (noise)
                   ↑ No clear sparse structure to extract
```

Algebraic decoder cannot identify which entries are signal vs noise.

---

## Parameter Count Reality Check

For V=100:
- Standard softmax: **12,800 params**
- Kronecker regression: **1,048,576 params**
- **Kronecker is 82× WORSE!**

Crossover point: V ≈ 10,000
- V=10K: Break-even (~1.2×)
- V=50K: Kronecker 6× better
- V=1M: Kronecker 122× better

**Implication:** Only beneficial for LARGE vocabularies (50K+)

---

## Why This Matters

Phase 1 proved: **token ↔ κ is invertible** ✓

Phase 2 reveals: **h → κ is not directly learnable** ✗

The problem is NOT invertibility (solved).  
The problem IS representation learning (unsolved).

**We're asking the network to predict a DISCRETE, SPARSE, STRUCTURED object.**

This is fundamentally different from:
- Standard regression (continuous targets)
- Classification (one-hot targets)
- LM prediction (distribution over tokens)

---

## Proposed Solutions

### Solution 1: Sparsification (Most Promising)
Force predictions to be sparse through:
- Top-k selection at inference
- L1 regularization during training
- Structure-aware losses
- Gumbel-softmax or straight-through estimators

**Architecture:**
```
h → κ_continuous → sparsify(κ, k=token_length) → κ_sparse → decode
```

**Challenges:**
- Need differentiable approximation for training
- May not guarantee valid structure
- Requires careful tuning

### Solution 2: Hybrid Approach
Combine Kronecker with small classifier:
```
h → κ_pred (structure guidance)
h → small_classifier (final decision)
```

Defeats some of the parameter savings but may work better.

### Solution 3: Structured Prediction
Predict (byte, position, length) directly instead of full κ:
```
h → byte_logits (256 classes)
h → position_logits (32 classes)  
h → length_predictor
```

Different mathematical framework, unclear if it helps.

### Solution 4: Retrieval
Don't decode algebraically, retrieve nearest token:
```
h → κ_pred
token = argmin ||κ_pred - κ(token)||
```

Still O(V) cost, doesn't eliminate vocabulary dependence.

---

## Next Steps (Priority Order)

### 1. Test Sparsification (HIGH PRIORITY)
- [ ] Implement top-k sparsification at inference
- [ ] Add L1 sparsity loss during training
- [ ] Measure decode accuracy
- [ ] **Decision point:** If >80% → continue, else pivot

### 2. If Sparsification Works
- [ ] Scale to V=1K, 10K, 50K
- [ ] Compare parameters vs standard approach
- [ ] Measure inference speed
- [ ] Document parameter crossover points

### 3. If Sparsification Fails
- [ ] Test hybrid approach
- [ ] Investigate structured prediction
- [ ] Consider alternative architectures
- [ ] Document why V2 hypothesis fails

### 4. Final Phase 2 Report
- [ ] Answer: "Can we eliminate V×d output head?"
- [ ] Provide evidence (experiments, plots, tables)
- [ ] Explain limitations
- [ ] Suggest future work

---

## Key Insights for Research Report

### What Works ✓
1. **Phase 1 algebraic decoder:** 100% accurate on ground-truth κ
2. **Invertibility:** token ↔ κ is bijective (with truncation caveat)
3. **O(D) complexity:** Decoder is vocabulary-independent
4. **Unseen tokens:** 98.9% reconstruction on never-before-seen tokens

### What Doesn't Work ✗
1. **Direct h→κ regression:** 0% decode accuracy
2. **Noise sensitivity:** Decoder requires precision beyond neural nets
3. **Small-vocab parameter count:** Worse than standard for V<10K
4. **Structured learning:** Networks don't naturally learn discrete sparse structure

### The Core Trade-off
**Precision vs Generalization**
- Algebraic decoder: Precise but brittle
- Neural decoder: Robust but requires parameters
- **No free lunch**

---

## Time Spent

- Phase 2 design: 30 min
- Tiny Transformer implementation: 1 hour
- Output heads implementation: 30 min
- Training experiment: 1 hour
- Diagnostics: 30 min
- Documentation: 30 min

**Total:** ~4 hours

---

## Status

**Phase 2 Progress:** ~40% complete  
**Critical blocker identified:** YES ✓  
**Path forward:** CLEAR  
**Confidence in understanding problem:** HIGH  

**Phase 2 ETA:** 2-3 more days to test sparsification and finalize report

---

## Files Created This Session

```
docs/
├── phase2_design.md          # Full Phase 2 plan
└── phase2_findings.md         # Detailed analysis of failure mode

experiments/phase2/
├── tiny_transformer.py        # Transformer implementation
├── output_heads.py            # Three output head variants
├── train_minimal.py           # Training experiment
└── diagnose_decoder.py        # Decoder diagnostics

PHASE2_SESSION_SUMMARY.md      # This file
```

---

## Verdict So Far

**Question:** Can invertible Kronecker replace V×d output head?

**Answer (preliminary):** NO - not with current algebraic decoder.

**Confidence:** 70% (pending sparsification test)

**Reason:** Fundamental incompatibility between continuous neural predictions and discrete algebraic decoding.

**But:** Phase 1 was still valuable - proved invertibility exists.

**Impact:** If V2 fails, Phase 1 algebraic decoder is still novel research contribution demonstrating mathematical properties of Kronecker embeddings.

---

## What This Means for the Paper

### If Sparsification Works (20% probability):
- **Title:** "Kronecker Embeddings with Invertible Output Heads"
- **Contribution:** Vocabulary-independent decoding for LMs
- **Impact:** HIGH - enables 100× parameter reduction for large vocabs

### If Sparsification Fails (80% probability):
- **Title:** "On the Invertibility of Kronecker Embeddings"
- **Contribution:** Mathematical analysis of Kronecker structure + impossibility result
- **Impact:** MEDIUM - theoretical contribution, explains architectural constraint

**Either way, we have a publishable result.**

Positive results are exciting, but **rigorous negative results are also valuable scientific contributions.**

---

## User Action Required

**None - continuing with Phase 2 automatically.**

Will implement sparsification approach next and report findings.

If you want to change direction or priorities, let me know!

---

**Next:** Implement `train_with_sparsification.py` and test decode accuracy.
