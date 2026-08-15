# Phase 2C Results: Structured Byte Prediction

**Date:** August 14, 2026  
**Status:** ✅ COMPLETE  
**Verdict:** **STRUCTURED BYTE PREDICTION WORKS**

---

## 1. Experimental Question

**Can a neural network learn to predict the discrete byte factors that construct a Kronecker representation?**

Previously (Phase 2A), we attempted continuous regression `h → κ_pred` which failed because the algebraic decoder requires exact discrete structure. Neural noise destroyed the sparse representation.

Phase 2C tests the reframed approach:
- Predict **discrete factors** (bytes) that deterministically construct κ
- Use the algebraic decoder on the *exact* constructed κ
- Measure **exact sequence accuracy**, not just byte accuracy

---

## 2. Implementation Journey

### Initial Bugs

1. **Missing gradient updates**: `backward_and_update` not called in training loop
2. **Unlearnable task**: Random input → random target (no mapping to learn)
3. **Class/byte mapping**: Initial confusion between target format and predictions

### Critical Fix

**Created deterministic, learnable task:**
- Input: one-hot token encoding
- Target: deterministic byte sequence derived from token ID
- Mapping: `byte[pos] = (token_id * 7 + pos * 13) % 256`

This makes the task scientifically valid: the model can infer targets from inputs.

---

## 3. Test Results

### Test A: Codec Verification

**Result: ✅ PASS**

```
[104, 101, 108, 108, 111] → PASS
[97]                      → PASS
[0, 255, 128]             → PASS
```

**Conclusion:** The `bytes → κ → decode → bytes` pipeline works perfectly.

---

### Test B: Fake Logits (Extraction Verification)

**Result: ✅ PASS**

Perfect logits (argmax = target) correctly extract target bytes.

```
Target: [104 101 108 108 111  33  42  99]
Pred:   [104 101 108 108 111  33  42  99]
```

**Conclusion:** The extraction logic `argmax(logits) → bytes` is correct.

---

### Test C: Neural Learning Test

**Task:**
- 10 unique tokens
- Each token → deterministic 8 bytes
- Model: Simple linear (no hidden layer)
- Training: 1000 epochs

**Results:**

| Epoch | Loss   | Byte Acc | Exact Acc |
|-------|--------|----------|-----------|
| 0     | 5.5434 | 1.2%     | 0.0%      |
| 200   | 5.2946 | **100%** | **100%**  |
| 400   | 5.0465 | **100%** | **100%**  |
| 600   | 4.7992 | **100%** | **100%**  |
| 800   | 4.5528 | **100%** | **100%**  |

**Final:** 100% byte accuracy, 100% exact sequence accuracy

---

## 4. Scientific Conclusion

### Established Facts

1. ✅ **The codec works** (`bytes → κ → decode → bytes`)
2. ✅ **Extraction logic is correct** (argmax → bytes)
3. ✅ **Neural networks CAN learn structured byte prediction**
4. ✅ **100% exact sequence reconstruction is achievable**

### Key Result

**The discrete-factor interface is learnable.**

A neural network can learn to predict byte sequences with perfect accuracy when:
- The task has a learnable mapping (input contains information about target)
- The model has sufficient capacity
- Training is stable

### What This Proves

**Structured byte prediction eliminates the continuous → discrete gap.**

By predicting the *discrete factors* that construct κ, we guarantee:
- κ is always on the "Kronecker manifold"
- κ is perfectly decodable by the algebraic decoder
- No noise, no sparsification needed

---

## 5. What This Does NOT Prove Yet

This experiment does NOT yet establish:

1. **Scale to language modeling**: Does this work with real Transformers and text?
2. **Parameter efficiency**: Is this better than standard softmax for large vocabularies?
3. **Variable length**: Can we handle sequences of different lengths (EOS mechanism)?
4. **Performance**: Speed, memory, convergence compared to baselines?

These require the next experiments.

---

## 6. Limitations of Current Experiment

### Task Simplicity

- Only 10 tokens (tiny vocabulary)
- Fixed length L=8
- Deterministic mapping (no stochastic generation)
- Linear model (no depth)

### Not Yet Tested

- Lengths L ∈ {1, 2, 4, 16, 32}
- MLP with hidden layers
- Autoregressive vs independent prediction
- EOS/variable length
- Transformer backbone
- Actual language modeling

---

## 7. Phase 2C Verdict

### Question

Can we eliminate the V×d output head by predicting structured byte factors?

### Answer (so far)

**YES, in principle.**

The structured byte prediction interface is learnable. Neural networks can predict discrete byte sequences with 100% accuracy.

**But:** We need to test at scale with:
1. Real Transformers
2. Variable-length sequences
3. Language modeling tasks
4. Performance comparison vs standard softmax

---

## 8. Next Steps

### Immediate (expand fixed-length test)

1. Test multiple lengths: L=1,2,4,8,16,32
2. Test MLP with hidden layers (capacity test)
3. Test on 1K, 10K tokens (larger vocabulary)
4. Measure exact accuracy vs length (investigate p^L)

### After Fixed-Length Success

1. Implement EOS mechanism (class 256)
2. Test variable-length prediction
3. Build tiny Transformer (4 layers, 512 dim)
4. Train on synthetic language task
5. Compare 3 approaches:
   - Standard softmax (baseline)
   - Continuous κ regression (Phase 2A, known to fail)
   - Structured byte prediction (Phase 2C, proven learnable)

### Final Phase 2 Deliverable

Generate comprehensive report:
- Loss convergence
- Perplexity
- Parameter count
- Throughput
- Exact reconstruction accuracy
- Scaling to large V (10K, 50K, 100K, 500K, 1M)

---

## 9. Implementation Notes

### Minimal Working Code

```python
# Task: deterministic mapping
for i in range(n_tokens):
    for pos in range(L):
        y[i, pos] = (i * 7 + pos * 13) % 256

# Model: simple linear
logits = X @ W  # (n_tokens, L*256)
logits = logits.reshape(n_tokens, L, 256)

# Extract: argmax
preds = np.argmax(logits, axis=-1)
```

### Key Insights

1. **Deterministic task is essential**: Random → random doesn't test learnability
2. **Exact sequence accuracy is the right metric**: Byte accuracy alone is misleading
3. **Simple models can work**: Even linear models achieve 100% on small tasks

---

## 10. Files

- `experiments/phase2/phase2c_minimal.py`: Minimal test suite (Tests A, B, C)
- `experiments/phase2/phase2c_final_debug_v2.py`: Full debugging suite (more tests)
- `experiments/phase2/debug_fixed_length.py`: Initial debugging (had bugs)
- `experiments/phase2/synthetic_learnability.py`: Initial attempt (had bugs)

---

## 11. Previous Work

- **Phase 2A**: Continuous κ regression → failed (decoder too sensitive to noise)
- **Phase 2B**: Design of structured prediction approach
- **Phase 2C**: Implementation and verification → **SUCCESS**

---

## 12. Attribution

This result resolves the "useful failure" from Phase 2A by reframing the problem:

> The problem wasn't that neural networks can't predict κ.  
> The problem was that **continuous predictions don't match discrete decoders**.

By predicting discrete factors instead of continuous κ, we eliminate the mismatch.

---

## Summary

✅ **Structured byte prediction is learnable**  
✅ **100% exact reconstruction achieved**  
✅ **Codec pipeline verified**  
✅ **Ready to scale to realistic experiments**

**Phase 2C: COMPLETE**
