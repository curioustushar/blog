# Kronecker V2 — Phase 2 Design: Transformer Integration

**Objective:** Determine if invertible Kronecker representation can replace vocabulary-sized output head.

---

## The Core Question

**Phase 1 proved:** `token ↔ κ(token)` is bidirectional via algebraic decoder

**Phase 2 must answer:** Can `h_t → κ(token_t)` work without V×d parameters?

---

## Three Approaches to Test

### Approach A: Standard Softmax (Baseline)

```
Architecture:
  Transformer → h_t ∈ ℝ^d → W_out @ h_t → logits ∈ ℝ^V → softmax → token

Parameters:
  W_out: V × d_model

Loss:
  CrossEntropy(logits, target_token_id)

Decoding:
  argmax(logits)
```

**This is the standard approach.** Every token prediction requires computing V logits.

---

### Approach B: Kronecker Input + Standard Output (Current Paper)

```
Architecture:
  token → κ(token) → projection → e_in
  Transformer(e_in) → h_t
  h_t → W_out @ h_t → logits ∈ ℝ^V → softmax → token

Parameters:
  Input: D × d_model (Kronecker projection)
  Output: V × d_model (standard head)

Loss:
  CrossEntropy(logits, target_token_id)
```

**This is what the existing Kronecker paper does.** Input side uses Kronecker, output side uses standard softmax.

Parameter savings on INPUT only.

---

### Approach C: Kronecker-Invertible (V2 Hypothesis)

```
Architecture:
  token → κ(token) → projection → e_in
  Transformer(e_in) → h_t ∈ ℝ^d
  h_t → W_regress @ h_t → κ_pred ∈ ℝ^D
  κ_pred → algebraic_decode → token

Parameters:
  Input: D × d_model (shared with Approach B)
  Output: d_model × D (regression head)

Loss:
  MSE(κ_pred, κ_target) or similar

Decoding:
  algebraic_decode(κ_pred)
```

**This is the V2 hypothesis.** Learn h → κ via regression, then use algebraic decoder.

**Key question:** Does W_regress ∈ ℝ^{d_model × D} provide enough capacity to predict κ without vocabulary-specific parameters?

**Parameter comparison:**
- Standard: V × d_model
- Kronecker-invertible: d_model × D

For V=50K, d=512, D=8192:
- Standard: 50,000 × 512 = 25.6M
- Invertible: 512 × 8,192 = 4.2M
- **Savings: 6.1× reduction**

For V=1M:
- Standard: 512M
- Invertible: 4.2M
- **Savings: 122× reduction**

---

## Phase 2 Experimental Design

### Architecture: Tiny Transformer

```python
config = {
    'n_layers': 4,
    'd_model': 512,
    'n_heads': 8,
    'd_ff': 2048,
    'dropout': 0.1,
    'max_seq_len': 128,
    'vocab_size': 10_000,  # Start small
}
```

**Why small:** Fast iteration, clear signal, less confounded by scale.

### Dataset: TinyStories Subset

- Small, clean, simple language
- Fast training (hours, not days)
- Sufficient to test the architectural question

**Size:** ~10M tokens training, 1M validation

### Training Protocol

**Three independent runs:**
1. Baseline (standard softmax)
2. Kronecker input + standard output
3. Kronecker input + invertible output (V2)

**Matched conditions:**
- Same architecture (except output head)
- Same dataset
- Same hyperparameters
- Same number of training steps
- Same random seeds

**Metrics:**
- Training loss
- Validation loss
- Perplexity
- Top-1 accuracy
- Top-5 accuracy
- Parameter count
- Inference throughput (tokens/sec)
- Memory footprint

---

## Critical Design Decisions

### Decision 1: What is κ_target?

When training the regressor h → κ, what is the target?

**Option 1:** Raw Kronecker κ(token)
- Most direct
- Sparse, well-defined

**Option 2:** Z-normalized κ̄(token)
- Matches existing pipeline
- But Phase 1 showed z-norm is injective, so this should work

**Option 3:** Projected e(token) = W @ κ(token)
- This is what the model actually sees as input
- Might create a shortcut through the Transformer

**Recommendation:** Start with Option 1 (raw κ), test Option 2 as ablation.

### Decision 2: What loss function?

**MSE (Mean Squared Error):**
```python
loss = ||κ_pred - κ_target||²
```

**Pros:** Simple, differentiable, matches regression formulation  
**Cons:** Doesn't directly optimize token prediction accuracy

**Cosine Similarity:**
```python
loss = 1 - (κ_pred · κ_target) / (||κ_pred|| ||κ_target||)
```

**Pros:** Direction matters more than magnitude  
**Cons:** May not preserve sparse structure

**Hybrid:**
```python
loss = MSE(κ_pred, κ_target) + λ * CrossEntropy(decode(κ_pred), target_id)
```

**Pros:** Directly optimizes decode accuracy  
**Cons:** Requires differentiable decoder or REINFORCE

**Recommendation:** Start with MSE, add token-level accuracy as auxiliary metric.

### Decision 3: How to handle decoding during training?

**Problem:** Algebraic decoder is not differentiable (it uses argmax, thresholding).

**Solution 1:** Train with MSE on κ, evaluate with algebraic decoder
- Simple, clean separation
- Decoder is used only at inference

**Solution 2:** Gumbel-Softmax or straight-through estimator
- Make decoder approximately differentiable
- More complex

**Recommendation:** Solution 1. Train h → κ with MSE, use algebraic decoder at inference.

### Decision 4: What if h → κ regression fails?

**Possible outcomes:**

**Outcome A:** Regression works perfectly
- Model learns to predict κ with high accuracy
- Algebraic decoder reconstructs tokens
- Perplexity matches baseline
- **V2 hypothesis CONFIRMED**

**Outcome B:** Regression works partially
- κ prediction is approximate
- Decoder has non-zero error rate
- Perplexity degrades but acceptable
- **V2 hypothesis PARTIALLY CONFIRMED**

**Outcome C:** Regression fails completely
- Model cannot learn h → κ mapping
- Decoder produces garbage
- **V2 hypothesis REJECTED**

**Contingency:** If Outcome C, diagnose:
- Is D × d_model enough capacity?
- Is the Transformer learning useful representations in h?
- Is κ_target too complex/high-entropy?

---

## Expected Results

### Hypothesis 1: Baseline > Kronecker-input

**Claim:** Standard learned embeddings outperform Kronecker input.

**Counter-claim (from paper):** Kronecker input is 2.5% better.

**Test:** Compare Approach A vs Approach B.

### Hypothesis 2: Invertible output matches standard output

**Claim:** h → κ regression works as well as h → logits.

**Rationale:** 
- κ is structured, sparse, deterministic
- Transformer might find κ easier to predict than arbitrary logits
- D=8192 dimensions may be enough

**Test:** Compare Approach B vs Approach C on perplexity.

### Hypothesis 3: Parameter savings

**Claim:** Approach C uses fewer parameters than Approach B.

**Evidence (expected):**
- Output head: 4.2M (C) vs 25.6M (B) at V=10K
- Input head: same
- **Net savings: ~21M parameters**

---

## Success Criteria

### Minimum Success

- Approach C trains without diverging
- Perplexity < 2× baseline
- Decoder reconstructs >50% of tokens correctly
- Parameter count is lower

### Strong Success

- Perplexity within 10% of baseline
- Decoder accuracy >90%
- Parameter count 5-10× lower
- Inference throughput comparable or better

### Exceptional Success

- Perplexity matches or beats baseline
- Decoder accuracy >95%
- Parameter count 10-100× lower
- Scales to V=100K-1M

---

## Timeline

### Week 1: Implementation
- [ ] Day 1-2: Build tiny Transformer
- [ ] Day 2-3: Implement three output heads
- [ ] Day 3-4: Data loading, training loop
- [ ] Day 4: Initial training runs

### Week 2: Experiments
- [ ] Day 5-6: Train all three approaches
- [ ] Day 7: Ablations (loss functions, architectures)
- [ ] Day 8-9: Scaling tests (vocab 10K → 50K)
- [ ] Day 10: Analysis, plots, tables

### Week 3: Documentation
- [ ] Day 11-12: Phase 2 report
- [ ] Day 13: Blog post
- [ ] Day 14: Paper draft (if successful)

**Total:** ~3 weeks for complete Phase 2.

---

## Code Structure

```
experiments/phase2/
├── tiny_transformer.py       # Transformer implementation
├── output_heads.py            # Three output head variants
├── train.py                   # Training loop
├── evaluate.py                # Evaluation metrics
├── data_loader.py             # TinyStories loader
└── compare_approaches.py      # Side-by-side comparison

models/
├── baseline.py                # Approach A
├── kronecker_standard.py      # Approach B
└── kronecker_invertible.py    # Approach C
```

---

## Key Metrics to Track

| Metric | Baseline | Kron+Std | Kron+Inv | Why It Matters |
|--------|----------|----------|----------|----------------|
| Train loss | ✓ | ✓ | ✓ | Convergence |
| Val loss | ✓ | ✓ | ✓ | Generalization |
| Perplexity | ✓ | ✓ | ✓ | LM quality |
| Top-1 accuracy | ✓ | ✓ | ✓ | Token prediction |
| Decode accuracy | N/A | N/A | ✓ | Algebraic decoder |
| Parameters (input) | ✓ | ✓ | ✓ | Memory |
| Parameters (output) | ✓ | ✓ | ✓ | Main comparison |
| Training time | ✓ | ✓ | ✓ | Efficiency |
| Inference time | ✓ | ✓ | ✓ | Practical use |
| MSE(κ_pred, κ_target) | N/A | N/A | ✓ | Regression quality |

---

## Open Research Questions

### Q1: Does h contain enough information to predict κ?

**Background:** The Transformer learns representations h optimized for next-token prediction via softmax. Does h also contain the structured information needed to predict sparse κ?

**Test:** Measure mutual information I(h; κ) or train a probe.

### Q2: Is κ easier or harder to predict than logits?

**Hypothesis A:** Easier - κ is structured, sparse, deterministic  
**Hypothesis B:** Harder - κ is high-dimensional (8192 vs V for small vocab)

**Test:** Compare training curves.

### Q3: Can we use a smaller D?

**Current:** D = 256 × 32 = 8192  
**Alternative:** D = 256 × 16 = 4096 (reduce pos_dim)

**Tradeoff:** Smaller D = fewer parameters, but loses longer tokens.

### Q4: Can we share parameters between input and output?

**Idea:** Use the SAME projection for both:
```
Input: κ(token) → W @ κ(token) → e_in
Output: h → W^T @ h → κ_pred
```

**Benefit:** Even fewer parameters (only one D × d matrix)

**Risk:** May constrain learning

---

## Comparison to Related Work

### Adaptive Softmax (Grave et al., 2017)

**Approach:** Cluster vocabulary by frequency, use smaller softmax for rare words.

**Comparison:** Reduces output computation, not parameters. Still vocabulary-dependent.

### Hierarchical Softmax

**Approach:** Binary tree over vocabulary.

**Comparison:** O(log V) computation, but still vocabulary-dependent.

### Kronecker V2

**Approach:** Predict structured representation, decode algorithmically.

**Advantage:** Vocabulary-agnostic output head. Scales to 1M+ tokens.

---

## Risk Analysis

### Risk 1: Regression fails

**Mitigation:** 
- Start with small vocab (10K) where both approaches have similar parameter counts
- If fails, try hybrid loss (MSE + cross-entropy)
- Ablate: test if problem is capacity, optimization, or fundamental

### Risk 2: Decoder errors compound

**Mitigation:**
- Measure decode accuracy separately
- Test decoder robustness to κ_pred noise
- Consider beam search over candidate decodings

### Risk 3: Training instability

**Mitigation:**
- Careful initialization
- Gradient clipping
- Learning rate warmup
- Monitor κ_pred statistics (sparsity, magnitude)

---

## Phase 2 Deliverables

### Code
- [ ] Tiny Transformer implementation
- [ ] Three output head variants
- [ ] Training and evaluation scripts
- [ ] Comparison tools

### Results
- [ ] Training curves (loss, perplexity)
- [ ] Parameter count comparison
- [ ] Inference speed comparison
- [ ] Decoder accuracy analysis

### Documentation
- [ ] Phase 2 technical report
- [ ] Blog post: "Can Invertible Embeddings Replace the Output Head?"
- [ ] Paper draft (if successful)

---

## Success Definition

**Phase 2 succeeds if:**

We can definitively answer: "Can h → κ regression replace V × d output softmax?"

**Answer could be:**
- YES (strong success)
- PARTIALLY (interesting for large V)
- NO (but we understand why)

All three outcomes are scientifically valuable.

---

**Phase 2 Start Date:** August 14, 2026  
**Target Completion:** September 7, 2026 (3 weeks)

Let's begin.
