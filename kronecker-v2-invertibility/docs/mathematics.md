# Mathematical Analysis: Kronecker Embedding Injectivity

**Research Question:** Is the Kronecker Embedding representation injective, and if so, where does information loss occur?

## 1. The Encoding Pipeline

From the paper (Shravan, 2026, arXiv:2605.29459), the Kronecker encoding consists of these stages:

### Stage 1: UTF-8 Bytes
```
token → bytes
"hello" → [104, 101, 108, 108, 111]  (5 bytes)
```

**Properties:**
- Deterministic
- Injective (for valid UTF-8)
- Truncated at pos_dim bytes (default: 32)

### Stage 2: Kronecker Representation
```
κ(b) = (1/√L) · Σ_{p=1}^L (c_{b_p} ⊗ p_p)
```

Where:
- c_v ∈ ℝ²⁵⁶: one-hot vector for byte value v (position v = 1, rest zeros)
- p_p ∈ ℝ^{pos_dim}: one-hot vector for byte position p
- ⊗: Kronecker product
- Result: κ(b) ∈ ℝ^D where D = 256 × pos_dim = 8192 (for pos_dim=32)

**Properties:**
- Sparse: at most L non-zero entries
- Each non-zero entry = 1/√L
- Linear index: idx = byte_value × pos_dim + position

### Stage 3: Z-Normalization (Optional)
```
κ̄(b) = (κ(b) - μ) / σ
```

Where μ = mean(κ(b)), σ = std(κ(b))

**Properties:**
- Subtracts mean → mean information LOST
- NOT invertible without storing μ, σ

### Stage 4: Projection (Optional)
```
e(b) = W · κ̄(b)
```

Where W ∈ ℝ^{d_model × D}

**Properties:**
- Dimensionality reduction when d_model < D
- Learned matrix (not deterministic across models)
- May introduce collisions if rank(W) < D

---

## 2. Injectivity Analysis

### Theorem 1: Raw Kronecker (Stage 2) is Injective for Distinct Byte Sequences

**Claim:** For two distinct byte sequences b₁ ≠ b₂ where len(b₁), len(b₂) ≤ pos_dim, the Kronecker representations κ(b₁) ≠ κ(b₂).

**Proof Sketch:**

Consider two distinct byte sequences:
- b₁ = (b₁,₁, b₁,₂, ..., b₁,L₁)
- b₂ = (b₂,₁, b₂,₂, ..., b₂,L₂)

The Kronecker representation places a value at index:
```
idx(byte_val, pos) = byte_val × pos_dim + pos
```

**Case 1:** L₁ ≠ L₂ (different lengths)

The length normalization factor 1/√L differs, so even if the byte patterns overlapped, the magnitudes would differ. Moreover, the number of non-zero entries differs, making the vectors distinguishable.

**Case 2:** L₁ = L₂ = L (same length)

If b₁ ≠ b₂, then ∃ position p where b₁,p ≠ b₂,p.

At this position:
- κ(b₁) has a non-zero entry at index: b₁,p × pos_dim + p
- κ(b₂) has a non-zero entry at index: b₂,p × pos_dim + p

Since b₁,p ≠ b₂,p, these indices are different.

Therefore, κ(b₁) and κ(b₂) have non-zero entries at different positions → κ(b₁) ≠ κ(b₂).

**Conclusion:** κ is injective over distinct byte sequences up to length pos_dim. ∎

### Caveat: Truncation

**Exception:** When L > pos_dim, the encoding is truncated. In this case:
```
κ(b₁...b_{pos_dim}b_{pos_dim+1}...) = κ(b₁...b_{pos_dim})
```

Two byte sequences differing only after position pos_dim will collide.

**Empirical Finding:** For GPT-2's vocabulary with pos_dim=32, the paper reports ≥99.82% of tokens fit within 32 bytes, so truncation collisions are rare in practice.

---

### Theorem 2: Z-Normalization Destroys Direct Invertibility

**Claim:** The z-normalized representation κ̄(b) = (κ(b) - μ) / σ is NOT directly invertible without storing μ and σ.

**Proof:**

Given only κ̄(b), we cannot recover κ(b) because:
```
κ(b) = σ · κ̄(b) + μ
```

Without knowledge of μ and σ, which depend on the specific byte sequence, reconstruction is impossible.

**Implication:** If z-normalization is used, we must either:
1. Store μ, σ as auxiliary information (adds ~2 scalars per token)
2. Skip z-normalization entirely
3. Derive μ, σ from the structure of κ (may be possible since κ is sparse and structured)

---

### Theorem 3: Projection May Preserve Injectivity Over Finite Vocabulary

**Claim:** Even when d_model < D, the projection W may still separate a finite vocabulary V if the Kronecker representations {κ(v) : v ∈ V} occupy a low-dimensional subspace.

**Reasoning:**

The raw Kronecker representations are sparse (at most 32 non-zero entries out of 8192). They effectively lie in a much lower-dimensional manifold.

If the effective dimensionality of {κ(v)} is less than d_model, then a projection to ℝ^{d_model} can preserve distinctness.

**Empirical Test Required:**
- Encode entire vocabulary
- Project to d_model dimensions
- Check for collisions

**Initial Finding:** On 89-token test set, no collisions found at Stage 2 (raw Kronecker).

---

## 3. Information-Theoretic Bounds

### Minimum Bits Required

For a vocabulary of size V:
```
H = log₂(V) bits
```

Examples:
- V = 50K → 15.6 bits
- V = 100K → 16.6 bits
- V = 1M → 19.9 bits

### Available Capacity

**In raw Kronecker (D = 8192, float64):**
```
C = 8192 × 64 = 524,288 bits
```

This is vastly more than needed for any practical vocabulary.

**After projection to d_model = 768:**
```
C_proj = 768 × 64 = 49,152 bits
```

Still far more than log₂(1M) ≈ 20 bits.

**Conclusion:** Information capacity is NOT the bottleneck. The question is whether the specific learned projection W preserves distinctness.

---

## 4. Decodability Without Vocabulary Enumeration

### Challenge

Given a representation e(b), can we decode b without checking all V tokens?

### Approach 1: Inverse Projection (Requires Orthogonal W)

If W is orthogonal (WᵀW = I), then:
```
W⁺ = Wᵀ   (pseudo-inverse)
κ̄(b) ≈ Wᵀ · e(b)
```

Then reverse z-norm (if we know μ, σ):
```
κ(b) = σ · κ̄(b) + μ
```

Finally, extract bytes from the sparse vector κ(b).

### Approach 2: Sparse Recovery

Since κ(b) is sparse (≤32 non-zero entries), we can potentially use compressed sensing techniques:
```
Given: e(b) = W · κ(b)
Find: κ(b) such that ||κ(b)||₀ ≤ 32  (sparsity constraint)
```

### Approach 3: Algebraic Decoder

Observe that κ(b) has a specific structure:
- Non-zero entries occur at indices: b_p × pos_dim + p for p = 1..L
- All non-zero entries have the same magnitude: 1/√L

Algorithm:
1. Given κ(b), identify non-zero positions
2. For each non-zero position idx:
   - byte_val = idx // pos_dim
   - position = idx % pos_dim
3. Reconstruct byte sequence: (byte_val₁, byte_val₂, ..., byte_valL)
4. Decode UTF-8 → token

**This is a constructive inverse!**

### Complexity

- Step 1: O(D) scan for non-zeros → O(8192)
- Step 2-3: O(L) operations → O(32)
- Step 4: O(L) UTF-8 decoding

**Total: O(D) = O(8192), independent of vocabulary size V.**

This is vastly faster than O(V) for large vocabularies.

---

## 5. Experimental Verdict

### Preliminary Results (N = 89 tokens)

| Stage | Dimension | Unique | Collisions | Min Distance |
|-------|-----------|--------|------------|--------------|
| Stage 2: Raw Kronecker | 8192 | 88 | 0 | ~0.02 |

**VERDICT B: No Collision Found (Empirical)**

No collisions detected across 89 tokens at raw Kronecker stage. This provides empirical evidence of injectivity over this finite vocabulary but is NOT a mathematical proof over all possible byte strings.

### Caveats

1. **Test size:** 89 tokens is small. Need to test on 50K+ (GPT-2) or 1M+ tokens.
2. **Numerical precision:** Used float64; need to test float32, float16.
3. **Z-norm not tested:** Current test skipped z-normalization.
4. **Projection not tested:** Current test skipped learned projection.

---

## 6. Next Steps

### Mathematical

- [ ] Formalize proof of Theorem 1 (raw Kronecker injectivity)
- [ ] Analyze length normalization: does 1/√L introduce issues?
- [ ] Determine if μ, σ can be recovered from structure

### Empirical

- [ ] Scale test to 50K tokens (GPT-2)
- [ ] Test with z-normalization enabled
- [ ] Test with projection to d_model=768
- [ ] Measure collision rate under float16/bf16
- [ ] Test UTF-8 truncation edge cases

### Decoder

- [ ] Implement algebraic decoder (extract bytes from sparse κ)
- [ ] Test decoder on unseen tokens
- [ ] Measure decoding speed vs vocabulary size
- [ ] Compare vs nearest-neighbor search

---

## 7. Theoretical Conclusion

**The raw Kronecker representation κ(b) appears to be injective over distinct byte sequences ≤ pos_dim.**

**Information loss occurs at:**
1. **Truncation:** When byte length > pos_dim
2. **Z-normalization:** Mean is subtracted and cannot be recovered
3. **Projection:** May introduce collisions if W has insufficient rank or is poorly conditioned

**An algebraic decoder IS theoretically possible** for the raw Kronecker representation, with complexity O(D) independent of vocabulary size.

**Key insight:** The parameter-efficiency win on the INPUT side (Kronecker embedding) can potentially be extended to the OUTPUT side if we:
1. Skip or reverse z-normalization
2. Use an orthogonal or high-rank projection
3. Implement the algebraic sparse-recovery decoder

This would eliminate BOTH the input embedding table AND the output classification head, achieving symmetric parameter efficiency.

---

## References

1. Shravan, R. (2026). Kronecker Embeddings: Byte-Level Structured Token Representations for Parameter-Efficient Language Models. arXiv:2605.29459

2. Candès, E. J., & Wakin, M. B. (2008). An introduction to compressive sampling. IEEE Signal Processing Magazine, 25(2), 21-30. (Sparse recovery background)

3. Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. Cognitive Computation, 1(2), 139-159. (Vector symbolic architectures)
