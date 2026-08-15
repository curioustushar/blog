# Phase 3B: Tiny PyTorch Proxy Specification

**Status:** Designed, not yet implemented  
**Purpose:** Bridge Phase 2 (numpy) to Phase 3C (full benchmark)  

---

## Objective

**Reproduce Phase 2 results in PyTorch before attempting full LM training.**

This catches implementation bugs early, before introducing:
- Large datasets
- Complex Transformer architecture
- Expensive GPU training

---

## Design Principle

**Checkpoint progression:**

```
NumPy Phase 2 result
    ↓
PyTorch implementation
    ↓
Transformer integration (simple)
    ↓
Real LM (Phase 3C)
```

**Each step validates the previous before adding complexity.**

---

## Test 1: Fixed-Length Reproduction

### Goal

Reproduce Phase 2C result: 100% exact accuracy on L=8 fixed-length task.

### Specification

**Task:**
- 50 tokens, deterministic mapping
- Token → 8 bytes
- Byte value: `(token_id * 7 + pos * 13) % 256`

**Model:**
- Simple linear: `logits = input @ W`
- W: (50, 8 × 256)
- Output: (batch, 8, 256)

**Training:**
- 200 train examples, 50 test
- 1000 epochs
- Adam optimizer, lr=0.01
- Cross-entropy loss

**Success criterion:**
- Test exact accuracy ≥ 95% (ideally 100%)
- Matches Phase 2C numpy result

**Code structure:**
```python
import torch
import torch.nn as nn

class LinearByteHead(nn.Module):
    def __init__(self, n_tokens, L):
        super().__init__()
        self.linear = nn.Linear(n_tokens, L * 256)
        self.L = L
    
    def forward(self, x):
        # x: (batch, n_tokens) one-hot
        logits = self.linear(x)  # (batch, L*256)
        return logits.view(-1, self.L, 256)

def train_epoch(model, data, optimizer):
    model.train()
    for X, y in data:
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(
            logits.view(-1, 256),
            y.view(-1)
        )
        loss.backward()
        optimizer.step()
    return loss.item()

def evaluate(model, data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data:
            logits = model(X)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).all(dim=1).sum().item()
            total += X.size(0)
    return correct / total
```

**Deliverable:**
- Script: `phase3b_test1_fixed_length.py`
- Output: Exact accuracy ≥ 95%
- Runtime: < 1 minute on CPU

---

## Test 2: Variable-Length Reproduction

### Goal

Reproduce Phase 2E result: 100% exact accuracy on variable-length with EOS.

### Specification

**Task:**
- 25 tokens, lengths L=1 to 8
- Token → variable-length bytes + EOS
- EOS = class 256

**Model:**
- Linear: (25, 8 × 257)
- Output: (batch, 8, 257) with EOS

**Training:**
- 200 train, 50 test
- 1000 epochs
- Same optimizer as Test 1

**Success criterion:**
- Length accuracy ≥ 95%
- Exact sequence accuracy ≥ 95%

**Extraction:**
```python
def extract_bytes(logits):
    """Extract byte sequence, stop at EOS."""
    preds = logits.argmax(dim=-1)  # (batch, L)
    sequences = []
    for pred in preds:
        # Find first EOS (256)
        eos_mask = (pred == 256)
        if eos_mask.any():
            end = eos_mask.nonzero()[0].item()
        else:
            end = len(pred)
        # Extract bytes before EOS
        byte_seq = pred[:end].clamp(0, 255).cpu().numpy()
        sequences.append(bytes(byte_seq))
    return sequences
```

**Deliverable:**
- Script: `phase3b_test2_variable_length.py`
- Output: Exact accuracy ≥ 95%
- Runtime: < 1 minute on CPU

---

## Test 3: Length Scaling

### Goal

Reproduce Phase 2D result: 100% accuracy across L ∈ {1,2,4,8,16,32}.

### Specification

**Task:**
- Run Test 1 for each length
- Measure exact accuracy vs L

**Expected:**
- All lengths ≥ 95% (ideally 100%)
- No degradation with increasing L

**Deliverable:**
- Script: `phase3b_test3_scaling.py`
- Output: Table of accuracy vs L
- Plot: `accuracy_vs_length.png`

---

## Test 4: Codec Integration

### Goal

Verify PyTorch → κ → decode → token pipeline.

### Specification

**Components:**
```python
def construct_kappa(byte_seq):
    """bytes → κ (numpy)"""
    L = len(byte_seq)
    kappa = np.zeros(8192)
    for p in range(L):
        idx = byte_seq[p] * 32 + p
        kappa[idx] = 1.0
    if L > 0:
        kappa /= np.sqrt(L)
    return kappa

def decode_kappa(kappa):
    """κ → token_id (via algebraic decoder)"""
    from kronecker_encoder import algebraic_decode_raw_kronecker
    byte_seq = algebraic_decode_raw_kronecker(kappa, 256, 32, True)
    return bytes_to_token_id(byte_seq)

def full_pipeline(model, input):
    """input → model → bytes → κ → token_id"""
    logits = model(input)
    byte_seq = extract_bytes(logits)
    kappa = construct_kappa(byte_seq)
    token_id = decode_kappa(kappa)
    return token_id
```

**Test:**
- 100 random inputs
- Measure: % that decode successfully
- Expected: ≥ 95%

**Deliverable:**
- Script: `phase3b_test4_codec.py`
- Output: Codec success rate

---

## Test 5: Simple Transformer

### Goal

Attach byte head to minimal Transformer, before full Phase 3C.

### Specification

**Model:**
```python
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, L):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        
        # Simple Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model*4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Structured byte head
        self.byte_head = nn.Linear(d_model, L * 257)
        self.L = L
    
    def forward(self, x):
        # x: (batch, seq_len)
        seq_len = x.size(1)
        
        # Embed
        x = self.embed(x) + self.pos_embed(
            torch.arange(seq_len, device=x.device)
        )
        
        # Transform
        x = self.transformer(x.transpose(0, 1))  # (seq, batch, d)
        x = x.transpose(0, 1)  # (batch, seq, d)
        
        # Byte head (use last position)
        h_last = x[:, -1, :]
        logits = self.byte_head(h_last)
        return logits.view(-1, self.L, 257)
```

**Task:**
- Copy task: predict next token's bytes from context
- 50 vocab, seq_len=4, L=8
- 500 train examples

**Training:**
- 2000 steps
- Adam, lr=3e-4
- Gradient clipping

**Success criterion:**
- Exact accuracy ≥ 70%
  - (Lower bar due to more complex task)
- No crashes, no NaNs
- Training stable

**Deliverable:**
- Script: `phase3b_test5_transformer.py`
- Output: Final accuracy, loss curve
- Plot: `transformer_training.png`

---

## Test 6: Baseline Comparison

### Goal

Compare structured bytes vs standard softmax on Test 5 task.

### Specification

**Models:**
1. Softmax head: (d_model, vocab_size)
2. Byte head: (d_model, L × 257)

**Same:**
- Transformer backbone
- Training configuration
- Dataset

**Measure:**
- Accuracy (both methods)
- Parameters (backbone vs head)
- Training time (wall-clock)
- Memory usage

**Expected:**
- Both achieve similar accuracy (within 5%)
- Byte head has fewer parameters
- Training time comparable

**Deliverable:**
- Script: `phase3b_test6_comparison.py`
- Output: Comparison table
- **This is the mini-Phase 3C**

---

## Success Criteria for Phase 3B

### All Tests Must Pass

**Required for proceeding to Phase 3C:**

1. ✅ Test 1: Fixed-length ≥ 95% exact
2. ✅ Test 2: Variable-length ≥ 95% exact
3. ✅ Test 3: All lengths ≥ 95%
4. ✅ Test 4: Codec ≥ 95% success
5. ✅ Test 5: Transformer trains stably
6. ✅ Test 6: Byte head competitive with softmax

**If any test fails:**
- Debug before Phase 3C
- Do NOT proceed to expensive GPU training
- Fix PyTorch implementation first

---

## Implementation Checklist

### Before Writing Code

- [ ] Read Phase 2 numpy implementations
- [ ] Understand extraction logic (especially EOS)
- [ ] Review codec interface
- [ ] Plan tensor shapes carefully

### During Implementation

- [ ] Use PyTorch best practices (nn.Module, DataLoader)
- [ ] Add assertions for tensor shapes
- [ ] Log intermediate values for debugging
- [ ] Test on CPU first

### After Implementation

- [ ] All 6 tests pass
- [ ] Code is clean and documented
- [ ] Results match Phase 2 (where applicable)
- [ ] Ready for Phase 3C

---

## File Structure

```
phase3_experiments/
├── phase3b_proxy/
│   ├── test1_fixed_length.py
│   ├── test2_variable_length.py
│   ├── test3_scaling.py
│   ├── test4_codec.py
│   ├── test5_transformer.py
│   ├── test6_comparison.py
│   ├── utils.py
│   └── README.md
├── results/
│   ├── test1_results.txt
│   ├── test2_results.txt
│   ├── ...
│   └── plots/
└── phase3c_benchmark/
    └── (empty until Phase 3B passes)
```

---

## Timeline

**Estimated time:** 3-5 days

- Day 1: Tests 1-3 (reproduce Phase 2 core results)
- Day 2: Test 4 (codec integration)
- Day 3-4: Test 5 (simple Transformer)
- Day 5: Test 6 (comparison)

---

## Why This Matters

### Catches Issues Early

**Problems found in Phase 3B:**
- Tensor shape bugs
- Loss calculation errors
- Extraction logic bugs
- PyTorch vs numpy differences

**Cost:** Minutes to hours on CPU

---

**Problems found in Phase 3C:**
- Same issues, but discovered after:
  - Days of GPU training
  - Expensive dataset processing
  - Complex Transformer debugging

**Cost:** Days to weeks of wasted time

---

### Validates the Bridge

**Phase 3B proves:**
- PyTorch implementation correct
- Results reproduce Phase 2
- Ready for scale-up

**Without Phase 3B:**
- Jump directly numpy → full LM
- No validation checkpoint
- High risk of silent bugs

---

## Deliverable

**Phase 3B Report:**
```markdown
# Phase 3B Results

## Test 1: Fixed-Length
- Exact accuracy: 100%
- Status: ✅ PASS

## Test 2: Variable-Length
- Exact accuracy: 100%
- Status: ✅ PASS

## Test 3: Scaling
- L=1: 100%
- L=2: 100%
- ...
- Status: ✅ PASS

## Test 4: Codec
- Success rate: 100%
- Status: ✅ PASS

## Test 5: Transformer
- Exact accuracy: 78%
- Training stable: Yes
- Status: ✅ PASS

## Test 6: Comparison
- Softmax: 82%
- Byte head: 78%
- Status: ✅ PASS (within 5%)

## Verdict
✅ All tests pass. Ready for Phase 3C.
```

---

## Status

**Phase 3B:** Specification complete, not yet implemented  
**Next:** Implement Phase 3B before Phase 3C  
**Boundary:** Do not proceed to expensive GPU training until proxy passes
