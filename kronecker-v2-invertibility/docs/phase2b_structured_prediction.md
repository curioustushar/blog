# Phase 2B: Structured Factor Prediction

**Date:** August 14, 2026  
**Status:** Active research direction

---

## The Reframe

**Phase 2A discovered:** Continuous κ regression fails with algebraic decoder.

**Phase 2B hypothesis:** Predict discrete Kronecker factors instead.

### Why This Is Different

**Phase 2A (FAILED):**
```
h → dense κ̂ ∈ ℝ^8192 → algebraic_decode → token
     ↑ noisy, continuous, not on Kronecker manifold
```

**Phase 2B (TESTING):**
```
h → discrete factors → exact κ ∈ ℝ^8192 → algebraic_decode → token
                        ↑ perfectly on Kronecker manifold
```

The key: **Don't approximate κ. Construct it exactly from predicted factors.**

---

## What Are the Discrete Factors?

From Phase 1 analysis, Kronecker representation is:

```
κ(token) = Kronecker product of:
  - Byte embeddings (one-hot or learned)
  - Position embeddings (one-hot)
  - Optional normalization
```

For token "the":
- Byte 0: 't' (ASCII 116)
- Byte 1: 'h' (ASCII 104)
- Byte 2: 'e' (ASCII 101)

These map to specific indices in κ:
```
index = byte_value * pos_dim + position
```

**The discrete factors are:**
- Bytes: {b₁, b₂, ..., bₗ} where bᵢ ∈ [0, 255]
- Length: L ∈ [1, 32]

**That's it.** Everything else is deterministic construction.

---

## Structured Byte Prediction Architecture

### Head Design

```python
class StructuredByteHead:
    """
    Predict discrete bytes at each position.
    Then construct exact Kronecker representation.
    """
    
    def __init__(self, d_model, max_length=32):
        self.max_length = max_length
        
        # Byte classifiers (one per position)
        self.byte_predictors = [
            Linear(d_model, 256) for _ in range(max_length)
        ]
        
        # End-of-token predictor
        self.eot_predictor = Linear(d_model, max_length + 1)
        
    def forward(self, h):
        """
        Args:
            h: (batch, d_model) hidden state
            
        Returns:
            byte_logits: (batch, max_length, 256)
            length_logits: (batch, max_length+1)
        """
        byte_logits = [pred(h) for pred in self.byte_predictors]
        byte_logits = stack(byte_logits, dim=1)
        
        length_logits = self.eot_predictor(h)
        
        return byte_logits, length_logits
    
    def predict_and_construct_kappa(self, h):
        """
        Predict bytes, construct EXACT κ, decode.
        """
        byte_logits, length_logits = self.forward(h)
        
        # Predict discrete bytes
        bytes_pred = argmax(byte_logits, dim=-1)  # (batch, max_length)
        length_pred = argmax(length_logits, dim=-1)  # (batch,)
        
        # Construct EXACT Kronecker representation
        kappas = []
        for i in range(batch_size):
            L = length_pred[i]
            bytes_i = bytes_pred[i, :L]
            
            # Exact Kronecker construction
            kappa_i = construct_kronecker_exact(bytes_i)
            kappas.append(kappa_i)
        
        kappas = stack(kappas)
        
        # Algebraic decode (guaranteed to work on exact κ)
        tokens = [algebraic_decode(k) for k in kappas]
        
        return tokens
```

### Key Properties

**Parameters:** `d_model × (256 × max_length + max_length + 1)`

For d=512, L=32:
- Byte predictors: 512 × 256 × 32 = 4,194,304
- Length predictor: 512 × 33 = 16,896
- **Total: 4,211,200**

Compare to:
- Standard softmax (V=100K): 51,200,000 (12× larger)
- Standard softmax (V=1M): 512,000,000 (122× larger)

**Crucially:** This scales with L, not V!

---

## The Three-Way Comparison

### Architecture A: Standard Softmax (Baseline)
```
h → W_out @ h → logits ∈ ℝ^V → argmax → token
```
**Parameters:** V × d

### Architecture B: Continuous κ Regression (Phase 2A - FAILED)
```
h → W_regress @ h → κ̂ ∈ ℝ^D → algebraic_decode → token
```
**Parameters:** D × d  
**Problem:** κ̂ is noisy, decoder fails

### Architecture C: Structured Byte Prediction (Phase 2B - TESTING)
```
h → byte_classifiers → discrete bytes → exact κ → algebraic_decode → token
```
**Parameters:** (256L + L + 1) × d  
**Advantage:** κ is EXACT, decoder guaranteed to work

---

## Implementation Plan

### Step 1: Understand κ Construction [CURRENT]

Inspect `KroneckerEncoderStaged` to extract exact construction:

```python
def construct_kronecker_from_bytes(
    byte_seq: bytes,
    char_dim: int = 256,
    pos_dim: int = 32,
    apply_length_norm: bool = True
) -> np.ndarray:
    """
    Exact Kronecker construction from byte sequence.
    
    This is the FORWARD direction: bytes → κ
    (We already have the inverse: κ → bytes from Phase 1)
    """
    L = len(byte_seq)
    kappa = np.zeros(char_dim * pos_dim)
    
    for p in range(L):
        byte_val = byte_seq[p]
        idx = byte_val * pos_dim + p
        kappa[idx] = 1.0  # or some other value
    
    if apply_length_norm:
        kappa /= sqrt(L)
    
    return kappa
```

Verify this matches the encoder exactly.

### Step 2: Build Oracle Pipeline

```python
# Test: bytes → exact κ → decode → bytes
test_tokens = ["the", "hello", "world"]

for token in test_tokens:
    bytes_in = token.encode('utf-8')
    
    # Construct exact κ
    kappa = construct_kronecker_from_bytes(bytes_in)
    
    # Algebraic decode
    bytes_out = algebraic_decode(kappa)
    
    assert bytes_in == bytes_out, "Oracle pipeline broken!"
```

This establishes that discrete factors → exact κ → decode WORKS.

### Step 3: Implement Structured Head

```python
class StructuredByteHead:
    # (full implementation as above)
    pass
```

### Step 4: Training Loop

```python
# Standard cross-entropy on EACH byte position
for position in range(max_length):
    byte_loss += CrossEntropy(
        byte_logits[:, position, :],
        target_bytes[:, position]
    )

# Cross-entropy on length
length_loss = CrossEntropy(length_logits, target_lengths)

total_loss = byte_loss + length_loss
```

### Step 5: Inference

```python
# Predict discrete bytes
bytes_pred = argmax(byte_logits, dim=-1)
length_pred = argmax(length_logits, dim=-1)

# Construct EXACT κ (not approximate!)
kappa_exact = construct_kronecker_from_bytes(bytes_pred[:length_pred])

# Algebraic decode (guaranteed to work)
token = algebraic_decode(kappa_exact)
```

---

## Expected Results

### If This Works

**Byte accuracy:** 95-99% (per position)  
**Exact token accuracy:** 85-95% (all bytes correct)  
**Decode success rate:** 100% (on valid κ)  
**Parameter scaling:** O(256L), independent of V

**This would validate the V2 hypothesis.**

### If This Fails

Investigate:
- Is byte prediction accurate enough?
- Does length prediction work?
- Are there byte-to-byte dependencies we're missing?
- Should we use autoregressive byte decoding?

### Comparison to Phase 2A

Phase 2A:
- MSE: 0.000062
- Decode accuracy: 0%

Phase 2B (expected):
- Byte accuracy: >90%
- Decode accuracy: >80%

**The difference:** Exact κ construction from discrete predictions.

---

## Autoregressive Extension

Once independent byte prediction works, test:

```
p(b₁ | h)
p(b₂ | h, b₁)
p(b₃ | h, b₁, b₂)
...
```

This models byte dependencies (e.g., UTF-8 sequences, common patterns).

**Implementation:**
```python
class AutoregressiveByteHead:
    def __init__(self, d_model):
        self.byte_embedding = Embedding(256, d_model)
        self.rnn = GRU(d_model, d_model)
        self.byte_classifier = Linear(d_model, 256)
    
    def forward(self, h, max_length):
        bytes_pred = []
        hidden = h
        
        for pos in range(max_length):
            logits = self.byte_classifier(hidden)
            byte_pred = argmax(logits)
            bytes_pred.append(byte_pred)
            
            # Update hidden state
            byte_emb = self.byte_embedding(byte_pred)
            hidden = self.rnn(byte_emb, hidden)
        
        return bytes_pred
```

Compare parameter counts and accuracy.

---

## Critical Experiments

### Experiment 1: Oracle Verification
**Test:** bytes → exact κ → decode → bytes  
**Expected:** 100% accuracy  
**Purpose:** Verify the pipeline

### Experiment 2: Synthetic Data
**Task:** Predict random byte sequences  
**Data:** 10K random strings, length 1-32  
**Model:** Simple MLP → byte classifiers  
**Expected:** >95% byte accuracy  
**Purpose:** Test if discrete prediction is learnable

### Experiment 3: Tiny Transformer
**Task:** Language modeling on small corpus  
**Compare:** Softmax vs Continuous-κ vs Structured-bytes  
**Metrics:** Loss, perplexity, exact token accuracy, parameters  
**Purpose:** Main comparison

### Experiment 4: Vocabulary Scaling
**Test:** V ∈ {10K, 50K, 100K, 500K, 1M}  
**Measure:** Parameter count, inference time  
**Purpose:** Show O(256L) vs O(V) scaling

### Experiment 5: Autoregressive vs Independent
**Compare:** Independent byte heads vs autoregressive decoder  
**Metrics:** Byte accuracy, exact token accuracy  
**Purpose:** Determine if dependencies matter

---

## Parameter Count Analysis

For d_model=512, max_length=32:

| Architecture | Parameters | V=10K | V=100K | V=1M |
|--------------|-----------|-------|--------|------|
| Standard softmax | V × d | 5.1M | 51M | 512M |
| Continuous κ | D × d | 4.2M | 4.2M | 4.2M |
| Structured bytes | (256L+L+1) × d | 4.2M | 4.2M | 4.2M |

**Key observation:** Continuous-κ and structured-bytes have SAME parameter count.

**The difference is NOT parameters - it's the INTERFACE.**

Continuous-κ produces noisy ℝ^D vectors.  
Structured-bytes produces exact discrete factors.

---

## The Real V2 Hypothesis

**Old hypothesis (Phase 2A):**
> "We can regress κ from h and decode it."

**New hypothesis (Phase 2B):**
> "We can predict the discrete structural factors of κ from h, construct exact κ, and decode it deterministically."

**This is fundamentally different.**

The Transformer doesn't need to learn the Kronecker algebra.  
It only needs to predict bytes.  
The algebra is handled by deterministic construction.

---

## Success Criteria

### Minimum Success
- Byte prediction accuracy >90%
- Exact token accuracy >70%
- Works on synthetic data
- Works in tiny Transformer
- Decode success rate ~100% (on predicted bytes)

### Strong Success
- Exact token accuracy >90%
- Matches softmax perplexity
- Parameter count ~5× lower than softmax at V=50K
- Demonstrated at V=1M scale
- Autoregressive variant works better

### Exceptional Success
- Exact token accuracy >95%
- Beats softmax perplexity
- Works without ANY vocabulary-dependent parameters
- Scales to arbitrary vocabulary size
- Enables truly open-vocabulary LM

---

## Next Actions

1. **[IN PROGRESS]** Analyze exact κ construction from bytes
2. Build and test oracle pipeline
3. Implement `StructuredByteHead`
4. Train on synthetic data
5. Compare three architectures on tiny Transformer
6. Measure vocabulary scaling
7. Test autoregressive variant

**Do NOT:**
- Write blog post yet
- Scale to large Transformer yet
- Pursue sparsification experiments yet

**First validate:** Discrete factor prediction → exact κ → successful decode

---

**Status:** Phase 2B initiated  
**Expected time:** 2-3 days  
**Key question:** Can discrete structural prediction solve the continuous-noise problem?
