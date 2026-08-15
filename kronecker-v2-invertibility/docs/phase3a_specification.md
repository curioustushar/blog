# Phase 3A: Experimental Specification

**Status:** Protocol designed, not yet executed  
**Date:** August 14, 2026  

---

## Design Principle

**Phase 3 is a reproducible benchmark, not an exploratory phase.**

All experimental choices are specified **before** touching the implementation.

---

## Experimental Question

**Does structured byte prediction achieve comparable or better LM quality while producing a meaningful reduction in output-head cost and/or actual end-to-end resource usage?**

**Critical:** Success is NOT just "fewer parameters." It requires:
1. Comparable perplexity
2. Meaningful resource reduction (memory, speed, or both)

---

## Dataset

### Initial Benchmark: WikiText-103

**Why WikiText-103:**
- Standard LM benchmark
- Well-established baselines
- Manageable size (~100M tokens)
- Clean preprocessing available

**Specification:**
- Use official train/valid/test splits
- No additional preprocessing beyond standard tokenization
- Report perplexity on both validation and test sets

**Alternative (if WikiText-103 unavailable):**
- WikiText-2 (smaller, faster iteration)
- Penn Treebank (classic benchmark)
- Documented reason for substitution

---

## Tokenizer & Vocabulary

### Fixed Across All Methods

**Critical:** Same tokenizer for fair comparison

**Initial vocabulary size:** V = 10,000
- Byte-Pair Encoding (BPE) or WordPiece
- Train on WikiText-103 training set
- No special handling of rare tokens
- Same V for softmax and structured-byte methods

**Vocabulary scaling experiments:**
- After initial benchmark, test V ∈ {10K, 50K, 100K}
- Use larger corpus if needed for meaningful rare tokens
- Keep backbone architecture constant

**Encoding details:**
- Record maximum token length in bytes (L_max)
- Set fixed L for structured bytes (e.g., L=8 with padding)
- Document UTF-8 encoding assumptions

---

## Transformer Backbone

### Single Configuration Shared by All Methods

**Architecture:**
```
Layers: 6
Hidden dim (d_model): 512
FFN dim: 2048
Attention heads: 8
Dropout: 0.1
Max sequence length: 512
```

**Why this size:**
- Large enough to be meaningful
- Small enough for rapid iteration
- Matches common small-LM benchmarks

**What's shared:**
- Embedding layer: V × d_model (same for all)
- All Transformer layers (identical)
- Layer norms, positional encodings
- Only the OUTPUT HEAD differs

---

## Output Heads (Methods Under Test)

### Method 1: Standard Softmax (Baseline)

**Architecture:**
```python
h: (batch, seq_len, d_model)
logits = h @ W_out + b_out  # W_out: (d_model, V)
probs = softmax(logits, dim=-1)
```

**Parameters:** V × d_model = 10,000 × 512 = 5.12M

**Loss:** Standard cross-entropy

---

### Method 2: Structured Byte Head

**Architecture:**
```python
h: (batch, seq_len, d_model)
byte_logits = h @ W_bytes  # W_bytes: (d_model, L × 257)
# Reshape: (batch, seq_len, L, 257)
# 257 classes: 0-255 (bytes) + 256 (EOS)
```

**Parameters:** d_model × L × 257
- Example (L=8): 512 × 8 × 257 = 1.05M

**Loss:** Cross-entropy on byte predictions
- All positions (including EOS)
- Optional: weighted by actual token length

**Decoding:**
```python
predicted_bytes = argmax(byte_logits, dim=-1)
# Stop at first EOS
# Construct κ: bytes → κ
# Decode: κ → token_id (via algebraic decoder)
```

---

### Method 3: Continuous κ Head (Optional Control)

**Architecture:**
```python
h: (batch, seq_len, d_model)
kappa_pred = h @ W_kappa  # W_kappa: (d_model, D)
# D = 8192 (full κ dimension)
```

**Parameters:** d_model × D = 512 × 8192 = 4.19M

**Loss:** MSE on κ representation

**Decoding:**
```python
# Apply sparsification (top-k or threshold)
kappa_sparse = sparsify(kappa_pred)
# Algebraic decode
token_id = decode(kappa_sparse)
```

**Note:** Phase 2A suggests this will fail. Include only if computationally practical, to confirm the negative result.

---

## Training Configuration

### Fixed Across All Methods

**Optimizer:** AdamW
- Learning rate: 3e-4
- Weight decay: 0.1
- Betas: (0.9, 0.95)
- Gradient clipping: 1.0

**Schedule:** Cosine decay with warmup
- Warmup steps: 2000
- Total steps: 100,000 (or until convergence)

**Batch size:** 128 sequences
- Effective batch size (with gradient accumulation if needed)

**Sequence length:** 512 tokens

**Training budget:** IDENTICAL for all methods
- Same number of tokens seen
- Same number of gradient steps
- Same compute per forward pass (measure)

---

## Parameter Accounting

### Report Separately

**Backbone parameters:**
- Embedding: V × d_model
- Transformer layers: ~d_model² × (6 layers × 4 matrices)
- Layer norms, positional embeddings

**Output head parameters:**
- Softmax: V × d_model
- Structured bytes: d_model × L × 257
- Continuous κ: d_model × D

**Total model size:**
- Report backbone + head separately
- Enables fair comparison across V

---

## Evaluation Metrics

### Primary Metric: Perplexity

**Validation perplexity:**
- Evaluated every 1000 steps
- Early stopping if no improvement for 5 evaluations

**Test perplexity:**
- Evaluated only once at end (best checkpoint)
- Report with confidence intervals (multiple seeds)

**Token-level accuracy (optional):**
- Exact match rate
- Top-5 accuracy

---

### Secondary Metrics: Resource Usage

**Training throughput:**
- Tokens per second (wall-clock)
- Measured on same hardware

**Peak memory:**
- Training: activations + gradients
- Inference: activations only
- Measure on same GPU

**Inference latency:**
- Time to generate 1 token (batched)
- Time to generate sequence (autoregressive)
- Include decoding overhead for structured bytes

**Convergence speed:**
- Steps to reach target perplexity
- Wall-clock time to convergence

---

### Tertiary Metrics: Decoding Analysis

**For structured bytes only:**

**Byte prediction accuracy:**
- Per-position accuracy
- Exact sequence accuracy (all bytes correct)

**Length prediction accuracy:**
- Does model predict correct byte length?
- EOS placement accuracy

**Codec success rate:**
- % of predictions that decode successfully
- % that produce valid token IDs

**Error modes:**
- What happens when bytes are wrong?
- Graceful degradation or catastrophic failure?

---

## Vocabulary Scaling Experiments

### After Initial Benchmark (V=10K)

**Test progressively larger vocabularies:**

| Vocabulary | Corpus | Rare Tokens |
|------------|--------|-------------|
| 10K        | WikiText-103 | Baseline |
| 50K        | Larger corpus | More coverage |
| 100K       | Even larger | Full coverage |

**For each V:**
- Train all methods (softmax + structured bytes)
- Measure perplexity, memory, speed
- Calculate parameter savings
- Plot: perplexity vs V, memory vs V, speed vs V

**Goal:** Find crossover point
- Where does structured bytes start winning?
- Is there a V threshold for viability?

---

## Reproducibility Requirements

### Fixed Seeds

**All experiments use:**
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
```

**Multiple seeds for final results:**
- Report mean ± std over 3 seeds
- Use seeds: 42, 43, 44

---

### Configuration Files

**All hyperparameters in YAML:**
```yaml
model:
  layers: 6
  d_model: 512
  ffn_dim: 2048
  heads: 8
  dropout: 0.1

training:
  optimizer: adamw
  lr: 3e-4
  weight_decay: 0.1
  batch_size: 128
  max_steps: 100000

data:
  dataset: wikitext-103
  vocab_size: 10000
  max_seq_len: 512

output_head:
  type: [softmax | structured_bytes | continuous_kappa]
  L: 8  # for structured bytes
```

**Version control:**
- Git commit hash recorded
- All dependencies pinned (requirements.txt)

---

### Checkpointing

**Save checkpoints:**
- Every 5000 steps
- Best validation checkpoint
- Final checkpoint

**Checkpoint contents:**
- Model weights
- Optimizer state
- Training step
- Random state
- Configuration

---

### Logged Runs

**Use experiment tracking:**
- WandB, TensorBoard, or similar
- Log every 100 steps

**Tracked metrics:**
- Train loss
- Validation perplexity
- Learning rate
- Gradient norm
- Memory usage
- Throughput (tokens/sec)

**Logged artifacts:**
- Config file
- Code snapshot
- Final results table
- Plots (perplexity curves, resource usage)

---

## Success Criteria

### Strong Success

**Both conditions must hold:**

1. **Quality:** Perplexity ≤ 102% of softmax baseline
   - Structured bytes competitive or better
   
2. **Efficiency:** Meaningful resource reduction
   - Memory: ≤ 90% of baseline
   - OR: Speed: ≥ 90% of baseline
   - OR: Parameters: ≤ 50% of baseline with no quality/speed loss

### Partial Success

**Quality acceptable, efficiency unclear:**

1. **Quality:** Perplexity ≤ 105% of softmax
2. **Efficiency:** Mixed results
   - Parameter savings realized
   - But speed/memory tradeoffs present
   - Use-case specific advantages

### Failure

**Any of:**
- Perplexity > 105% of baseline
- Memory usage > baseline (despite fewer params)
- Speed < 70% of baseline
- Doesn't scale to large V

---

## Statistical Significance

**All comparisons require:**
- Multiple seeds (n=3 minimum)
- Report mean ± standard error
- Statistical test (paired t-test, p < 0.05)

**Don't claim victory from:**
- Single run
- Cherry-picked seed
- Unreported variance

---

## Ablation Studies

### After Main Results

**If structured bytes succeeds, test:**

1. **Byte length (L):**
   - L ∈ {4, 6, 8, 10, 12}
   - Find optimal tradeoff

2. **Independent vs autoregressive:**
   - Current: independent (parallel)
   - Alternative: left-to-right (sequential)

3. **Loss weighting:**
   - Uniform (all positions equal)
   - Length-weighted
   - Position-weighted

4. **Architectural variants:**
   - MLP vs linear byte head
   - Shared vs separate position predictors

---

## Experiment Log Structure

```
phase3_experiments/
├── configs/
│   ├── softmax_v10k.yaml
│   ├── structured_bytes_v10k.yaml
│   └── continuous_kappa_v10k.yaml
├── runs/
│   ├── softmax_v10k_seed42/
│   ├── structured_bytes_v10k_seed42/
│   └── ...
├── checkpoints/
├── results/
│   ├── tables/
│   ├── plots/
│   └── analysis/
└── logs/
```

---

## Timeline & Resources

### Estimated Timeline

**Phase 3B (Tiny proxy):** 3-5 days
- PyTorch implementation
- Reproduce Phase 2 results
- Validate training loop

**Phase 3C (Initial benchmark):** 2-3 weeks
- V=10K on WikiText-103
- All three methods
- Multiple seeds

**Phase 3D (Scaling):** 2-3 weeks
- V ∈ {50K, 100K}
- Analysis & plots

**Total:** 5-8 weeks

### Required Resources

**Hardware:**
- GPU: 1× A100 (40GB) or equivalent
- Storage: 100GB for datasets/checkpoints
- RAM: 32GB+

**Software:**
- PyTorch 2.0+
- transformers library
- datasets library
- wandb or tensorboard

---

## What This Specification Enables

### Before Execution

✅ Complete experimental design  
✅ All decisions documented  
✅ Reproducibility guaranteed  
✅ Success criteria pre-specified  

### During Execution

✅ No exploratory drift  
✅ Fair comparison enforced  
✅ Statistical rigor maintained  
✅ Clear stopping conditions  

### After Execution

✅ Results are defensible  
✅ Comparison is fair  
✅ Claims are precise  
✅ Replication is possible  

---

## Boundary with Phase 2

**Phase 2:** Feasibility validated (numpy, synthetic tasks)  
**Phase 3:** Utility validated (PyTorch, real LM tasks)

**Phase 3 does NOT re-validate feasibility.**
- Assumes structured bytes work (Phase 2 proved this)
- Focuses on: does it offer practical advantages?

---

## Next: Phase 3B Specification

The tiny PyTorch proxy that bridges Phase 2 (numpy) to Phase 3C (full LM).

**Status:** Phase 3A specification complete, not yet implemented.
