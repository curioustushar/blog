# Kronecker Embedding V2: Invertibility Investigation

**Research Question:** Can Kronecker Embeddings be made mathematically invertible, eliminating the need for a vocabulary-sized output classification head?

## Abstract

This repository investigates whether the Kronecker Embedding representation (Shravan, 2026) can be extended to support efficient decoding without requiring a conventional V × d_model output classifier. We analyze the mathematical properties of the encoding pipeline, identify sources of information loss, design experiments to test injectivity, and evaluate whether an explicit inverse transformation exists that can scale to 1M+ token vocabularies.

**Status:** 🚧 Active Research — In Progress

## Problem Selection

**Selected:** Problem 5 — Invertible / Reverse Kronecker Representation

From the original Kronecker Embedding paper (arXiv:2605.29459), we know:
- Input embedding reduced from 91-94% of parameters
- BUT: Output head still requires V × d_model parameters
- Weight tying is "architecturally inapplicable" because codec dimension D ≠ d_model

**Core hypothesis:** If we can construct an inverse mapping `embedding → token` that doesn't require evaluating all V vocabulary items, we could eliminate the output head entirely and achieve symmetric parameter efficiency on both input and output sides.

## Research Objectives

### Primary

1. **Determine injectivity:** Is the current Kronecker encoding injective over the vocabulary?
2. **Identify information loss:** Where exactly does the representation lose information?
3. **Design invertible variant:** What minimal modifications make the encoding bijective?
4. **Construct explicit inverse:** Can we decode without a learned V-way classifier?
5. **Validate at scale:** Does it work at 32K, 100K, 500K, and 1M tokens?

### Secondary

6. Measure collision rates under floating-point precision
7. Test generalization to unseen tokens
8. Integrate into a small Transformer and measure end-to-end performance
9. Compare parameter counts and inference costs vs. standard approaches

## Mathematical Framework

### Current Kronecker Encoding

For a token with UTF-8 byte sequence **b** = (b₁, ..., b_L):

```
κ(b) = (1/√L) · Σₚ₌₁ᴸ (cᵦₚ ⊗ pₚ)
```

Where:
- c_v ∈ R²⁵⁶: one-hot vector for byte value v
- p_p ∈ R^{pos_dim}: one-hot vector for position p
- ⊗: Kronecker product
- Result: κ(b) ∈ R^D where D = 256 × pos_dim

Then:
1. **Z-normalization:** κ̄(b) = (κ(b) - μ) / σ
2. **Projection:** e(b) = W · κ̄(b), where W ∈ R^{D×d_model}

### Information Flow Analysis

| Stage | Transform | Input Dim | Output Dim | Learnable? | Potential Info Loss? |
|-------|-----------|-----------|------------|------------|---------------------|
| 1. Bytes | UTF-8 decode | variable | ≤256 values | No | None (deterministic) |
| 2. Kronecker | c ⊗ p | bytes | D = 256·pos_dim | No | **Position overflow** (if L > pos_dim) |
| 3. Length norm | ÷√L | D | D | No | **Length info preserved** (in norm) |
| 4. Z-norm | (x-μ)/σ | D | D | No | **Mean removed** (not recoverable) |
| 5. Projection | W·x | D | d_model | Yes | **Dimension reduction** (D→d_model) |

### Levels of Invertibility

We define five increasingly strong notions:

**Level 1 — Empirical Reconstruction**
- Decoder achieves >X% accuracy on test set
- ❌ Does NOT prove mathematical invertibility
- ❌ May be memorization

**Level 2 — Injectivity**
- ∀ x≠y in vocabulary: f(x) ≠ f(y)
- Finite vocabulary property
- Still requires checking all V candidates

**Level 3 — Constructive Inverse**
- Explicit algorithm: f⁻¹(f(x)) = x
- No vocabulary enumeration needed
- Complexity: o(V)

**Level 4 — Scalable Inverse**
- Works at 32K, 100K, 500K, 1M+ tokens
- Practical time/memory bounds
- Real-world deployment feasible

**Level 5 — Continuous Invertibility**
- Works for arbitrary byte strings
- Not just the training vocabulary
- Generalizes to unseen tokens

## Research Plan

### Phase 1: Mathematical Analysis [Week 1]

**1.1 Encoder Decomposition**
- [ ] Document exact transformation at each stage
- [ ] Identify reversible vs irreversible operations
- [ ] Compute theoretical information capacity

**1.2 Injectivity Proof/Disproof**
- [ ] **Before z-norm:** Is κ(b) injective?
- [ ] **After z-norm:** Does normalization introduce collisions?
- [ ] **After projection:** When D > d_model, is W·κ̄ injective?

**1.3 Information-Theoretic Bounds**
- [ ] Minimum bits required: log₂(V)
- [ ] Available capacity in D dimensions
- [ ] Available capacity in d_model dimensions
- [ ] Floating-point precision effects

**Deliverable:** `docs/mathematics.md` with theorems, proofs, or explicit counterexamples

### Phase 2: Collision Detection [Week 1]

**2.1 Brute-Force Search**
- [ ] Encode all tokens in vocabulary
- [ ] Check pairwise distances
- [ ] Report exact collisions (if any)
- [ ] Test at κ level, κ̄ level, and e level

**2.2 Adversarial Search**
- [ ] Generate byte strings designed to collide
- [ ] Test permutations, repeated characters
- [ ] Test maximum-length tokens
- [ ] Test Unicode edge cases

**2.3 Numerical Precision**
- [ ] Test FP32, FP16, BF16
- [ ] Measure effective distinguishability
- [ ] Report collision rate vs tolerance

**Deliverable:** `results/collisions/` with collision pairs, distances, statistics

### Phase 3: Decoder Design [Week 2]

**3.1 Baseline: Vocabulary Classifier**
```python
class VocabClassifier:
    def decode(self, e: Tensor) -> int:
        logits = self.W_out @ e  # (V, d_model) @ (d_model,)
        return logits.argmax()
```
- Standard approach
- Parameters: V × d_model
- Complexity: O(V)

**3.2 Nearest Neighbor Decoder**
```python
class NearestNeighborDecoder:
    def decode(self, e: Tensor) -> int:
        # Precompute embeddings for all V tokens
        distances = torch.cdist(e, self.vocab_embeddings)
        return distances.argmin()
```
- Tests representation quality
- No learned parameters (beyond encoder)
- Complexity: O(V)

**3.3 Algebraic Decoder (Primary Target)**
```python
class AlgebraicDecoder:
    def decode(self, e: Tensor) -> int:
        # Inverse projection (if possible)
        kappa_bar = self.W_pseudo_inv @ e
        
        # Reverse z-norm (requires mean/std recovery)
        kappa = kappa_bar * std + mean
        
        # Extract byte-position pairs from sparse vector
        bytes = self.extract_bytes(kappa)
        
        # Map bytes → token_id
        return self.bytes_to_token(bytes)
```
- **Goal:** Avoid enumerating V tokens
- **Challenge:** Reverse z-norm requires μ, σ
- **Challenge:** Projection may not be invertible

**3.4 Hybrid Decoder**
- Algebraic decoding to narrow candidates
- Small classifier over reduced set
- Complexity: O(√V) or O(log V)

**Deliverable:** `src/decoders/` with all decoder implementations

### Phase 4: Encoder Modifications [Week 2]

**4.1 Remove Z-Normalization**
- Eliminates irreversible mean-centering
- May hurt downstream performance
- Test impact on LM loss

**4.2 Orthogonal Projection**
- Replace learned W with orthogonal matrix
- Guarantees invertibility when d_model ≥ D
- Test: random orthogonal, learned orthogonal (Cayley)

**4.3 Expand D**
- Increase pos_dim to raise codec dimension
- D > d_model → append identity block
- Preserve injectivity through dimensionality

**4.4 Embed Token Length**
- Add explicit length encoding to κ
- Helps reverse length normalization
- Cost: +log₂(pos_dim) bits

**4.5 Store Decoding Hints**
- Auxiliary small vector with μ, σ, L
- Concatenate to main embedding
- Cost: +3 dimensions

**Deliverable:** `src/encoders/` with modified Kronecker variants

### Phase 5: Injectivity Experiments [Week 3]

**5.1 Vocabulary Sweep**
```python
# For vocabularies: 1K, 10K, 32K, 100K, 500K, 1M
for V in vocab_sizes:
    encoder = ModifiedKronecker(V, pos_dim=P)
    embeddings = {token: encoder(token) for token in vocab}
    
    # Check uniqueness
    unique = len(set(embeddings.values()))
    collisions = V - unique
    
    report(V, collisions, collision_rate)
```

**5.2 Generalization Test**
- Train on vocab_train (50% of tokens)
- Test decoding on vocab_test (unseen 50%)
- Success = decoder works on unseen tokens

**5.3 Scaling Analysis**
```
Plot:
X-axis: Vocabulary size (log scale)
Y-axis: Decoding accuracy, collision rate, time per token
Lines: Each decoder type
```

**Deliverable:** `results/injectivity/` with plots, tables, and analysis

### Phase 6: Transformer Integration [Week 3]

**6.1 Tiny Transformer**
- 2-6 layers, 256-512 hidden dim
- Small dataset (e.g., TinyStories)
- Controlled comparison

**6.2 Three Arms**
1. **Baseline:** Standard embedding + standard output head
2. **Kronecker-Standard:** Kronecker input + standard output head (current paper)
3. **Kronecker-Invertible:** Kronecker input + algebraic decoder

**6.3 Metrics**
- Training loss, validation loss, perplexity
- Top-1 accuracy, top-5 accuracy
- Parameter count (input side, output side, total)
- Tokens/sec throughput
- Memory footprint

**6.4 Ablations**
- Remove z-norm
- Orthogonal projection
- Different decoder types
- Various pos_dim values

**Deliverable:** `experiments/transformer/` with training logs, checkpoints, plots

### Phase 7: Attack the Method [Week 4]

**7.1 Collision Attack**
- Explicitly search for colliding byte sequences
- Genetic algorithms, gradient-based search
- Report any found collisions

**7.2 Precision Attack**
- Test under FP16, BF16, INT8 quantization
- Measure degradation in decoder accuracy

**7.3 Adversarial Strings**
- Pathological inputs: all zeros, all 0xFF
- Very long tokens (truncated)
- Repeated sequences

**7.4 Distribution Shift**
- Train on English, test on code
- Train on ASCII, test on Unicode
- Out-of-distribution robustness

**7.5 Noise Injection**
- Add Gaussian noise to embeddings
- Measure decoder degradation
- Compare robustness vs standard embeddings

**Deliverable:** `experiments/ablations/` with attack results

## Success Criteria

### Minimum Success

- [x] Documented exact Kronecker encoding
- [ ] Mathematical analysis of injectivity
- [ ] Collision experiments on 100K+ vocab
- [ ] ≥3 decoder implementations
- [ ] Clear identification of information bottlenecks
- [ ] Honest assessment of feasibility

### Strong Success

Additionally:
- [ ] Proof or disproof of injectivity
- [ ] Explicit inverse with complexity o(V)
- [ ] Decoder works at 1M tokens
- [ ] Integration into Transformer without loss degradation
- [ ] Parameter savings vs standard approach quantified

### Exceptional Success

Additionally:
- [ ] Generalization to unseen tokens demonstrated
- [ ] Formal theorem with proof
- [ ] Published-quality results
- [ ] Open research questions identified for future work

## Preliminary Findings

### Encoding Analysis

**Injectivity Before Projection:**

The base Kronecker codec κ(b) is **likely injective** for distinct byte sequences up to length `pos_dim`, because:
- Each (byte, position) pair maps to a unique index in the D-dimensional vector
- Two different byte sequences → different sparse patterns
- **Exception:** Truncation when L > pos_dim may cause collisions

**Z-Normalization Issue:**

Z-norm removes the mean, which contains information about the token. This is **not invertible** without storing μ.

**Projection Bottleneck:**

When D > d_model (typical: D=4096 or 8192, d_model=768 or 1024), the projection W ∈ R^{D×d_model} is a dimensionality reduction. Even with a learned pseudo-inverse, this may not be fully invertible over the finite vocabulary.

**Hypothesized Solution:**

1. **Remove z-norm** or **store normalization parameters**
2. **Use orthogonal projection** to preserve as much information as possible
3. **Increase d_model** or **append auxiliary dimensions** for decoding hints
4. **Algebraic decoder:** Compute pseudo-inverse W⁺, then extract bytes from sparse vector

## Repository Structure

```
kronecker-v2-invertibility/
├── README.md                          # This file
├── pyproject.toml                     # Dependencies
├── LICENSE                            # MIT
│
├── src/
│   ├── kronecker/                     # Base Kronecker implementation
│   │   ├── __init__.py
│   │   ├── encoder.py                 # Kronecker codec
│   │   └── utils.py                   # Byte handling, truncation
│   ├── encoders/                      # Modified encoders
│   │   ├── __init__.py
│   │   ├── no_znorm.py               # Variant without z-norm
│   │   ├── orthogonal.py              # Orthogonal projection
│   │   ├── expanded.py                # Expanded dimensionality
│   │   └── hint_embedding.py          # With decoding hints
│   ├── decoders/                      # Decoder implementations
│   │   ├── __init__.py
│   │   ├── vocab_classifier.py        # Baseline: full softmax
│   │   ├── nearest_neighbor.py        # NN search
│   │   ├── algebraic.py               # Algebraic inverse
│   │   └── hybrid.py                  # Hybrid approach
│   └── models/                        # Transformer integration
│       ├── __init__.py
│       ├── tiny_transformer.py        # Small test model
│       └── training.py                # Training loop
│
├── experiments/
│   ├── collisions/                    # Collision detection
│   │   ├── brute_force.py
│   │   ├── adversarial.py
│   │   └── precision_test.py
│   ├── injectivity/                   # Injectivity proofs/tests
│   │   ├── vocab_sweep.py
│   │   └── generalization.py
│   ├── scaling/                       # Scaling experiments
│   │   ├── vocab_1k.py
│   │   ├── vocab_10k.py
│   │   ├── vocab_100k.py
│   │   └── vocab_1m.py
│   ├── precision/                     # Numerical precision
│   │   ├── fp32_vs_fp16.py
│   │   └── quantization.py
│   ├── transformer/                   # End-to-end training
│   │   ├── train_baseline.py
│   │   ├── train_kronecker.py
│   │   └── train_invertible.py
│   └── ablations/                     # Attack experiments
│       ├── collision_attack.py
│       ├── noise_injection.py
│       └── distribution_shift.py
│
├── tests/
│   ├── test_encoder.py
│   ├── test_decoders.py
│   ├── test_injectivity.py
│   └── test_integration.py
│
├── results/                           # Experimental results
│   ├── collisions/
│   ├── injectivity/
│   ├── scaling/
│   └── transformer/
│
├── figures/                           # Plots and visualizations
│   ├── collision_rates.png
│   ├── decoder_comparison.png
│   ├── scaling_curves.png
│   └── transformer_loss.png
│
└── docs/
    ├── mathematics.md                 # Mathematical analysis
    ├── experiments.md                 # Experiment details
    └── related_work.md                # Literature review
```

## Related Work

### Vector Symbolic Architectures
- Plate (1995): Holographic Reduced Representations
- Kanerva (2009): Hyperdimensional Computing
- Gayler (2003): Vector Symbolic Architectures

**Relevance:** These systems use high-dimensional vectors with algebraic operations (binding, bundling) that ARE invertible. Could Kronecker adopt similar principles?

### Invertible Neural Networks
- Dinh et al. (2017): Normalizing Flows (Real NVP)
- Kingma & Dhariwal (2018): Glow
- Behrmann et al. (2019): Invertible Residual Networks

**Relevance:** Techniques for constructing invertible transformations in neural networks. Could we wrap Kronecker in an invertible architecture?

### Output Layer Compression
- Grave et al. (2017): Adaptive Softmax
- Shazeer et al. (2017): Hierarchical Softmax
- Jean et al. (2015): Sampled Softmax

**Relevance:** Existing methods to reduce output head cost. How does invertible Kronecker compare?

### Learned Embeddings Geometry
- Lopardo et al. (2026): "Tied embeddings are biased toward output prediction space"
- Mu & Viswanath (2018): Word embeddings anisotropy

**Relevance:** Understanding structure of learned embeddings may inform invertibility constraints.

## Reproduction Instructions

### Setup

```bash
# Clone repository
git clone <repo-url>
cd kronecker-v2-invertibility

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Install reference Kronecker (for comparison)
pip install kronecker-embeddings

# Run tests
pytest tests/ -v
```

### Run Collision Detection

```bash
# Brute-force collision search on 32K vocab
python experiments/collisions/brute_force.py --vocab-size 32768 --pos-dim 16

# Adversarial search
python experiments/collisions/adversarial.py --num-trials 10000
```

### Run Injectivity Tests

```bash
# Vocabulary sweep
python experiments/injectivity/vocab_sweep.py --vocab-sizes 1000 10000 32000 100000

# Generalization test
python experiments/injectivity/generalization.py --split 0.5
```

### Train Tiny Transformer

```bash
# Baseline
python experiments/transformer/train_baseline.py --layers 4 --hidden 256 --data tinystories

# Kronecker-invertible
python experiments/transformer/train_invertible.py --layers 4 --hidden 256 --decoder algebraic
```

### Generate All Figures

```bash
python scripts/generate_figures.py --results results/ --output figures/
```

## Current Status

**Week 1: Mathematical Analysis**
- [x] Reviewed reference Kronecker implementation
- [x] Documented encoding pipeline
- [x] Identified information loss points
- [ ] Formal injectivity proof/disproof — IN PROGRESS
- [ ] Collision experiments — NEXT

**Week 2-4: Implementation & Experiments**
- [ ] NOT STARTED

## Research Questions

1. **Is current Kronecker κ(b) injective before z-norm?**
   - Hypothesis: YES, for distinct byte sequences ≤ pos_dim
   - Test: Explicit collision search

2. **Does z-normalization introduce collisions?**
   - Hypothesis: NO, but removes invertibility (mean is lost)
   - Solution: Store μ, σ or skip z-norm

3. **Can projection W be inverted over finite vocabulary?**
   - Hypothesis: PARTIALLY — pseudo-inverse recovers approximate κ̄
   - Test: Measure reconstruction error

4. **Does algebraic decoder scale to 1M tokens?**
   - Hypothesis: YES, if we solve (2) and (3)
   - Test: Implement and benchmark

5. **Does invertible variant hurt LM performance?**
   - Hypothesis: NO, if properly designed
   - Test: Train small Transformer, compare perplexity

## Conclusion

This research investigates a fundamental question: **Can token embeddings be made truly invertible?**

If successful, we could:
- Eliminate the output vocabulary head
- Achieve symmetric parameter efficiency (input + output)
- Enable vocabulary-free or vocabulary-agnostic architectures

If unsuccessful, we will:
- Identify the exact mathematical obstacles
- Quantify information bottlenecks
- Propose the closest achievable approximation

Either outcome advances our understanding of token representations in language models.

## Citation

If you use this research, please cite:

```bibtex
@misc{gupta2026kronecker_invertibility,
  title={Kronecker Embedding V2: Investigating Invertible Token Representations},
  author={Gupta, Tushar},
  year={2026},
  howpublished={\url{https://curioustushar.github.io/blog/}},
  note={Research investigation}
}
```

And cite the original Kronecker Embedding paper:

```bibtex
@article{shravan2026kronecker,
  title={Kronecker Embeddings: Byte-Level Structured Token Representations
         for Parameter-Efficient Language Models},
  author={Shravan, Rohan},
  journal={arXiv preprint arXiv:2605.29459},
  year={2026},
  url={https://github.com/theschoolofai/kronecker-embeddings}
}
```

## License

MIT License - see LICENSE file

## Contact

- Author: Tushar Gupta
- Blog: https://curioustushar.github.io/blog/
- Issues: [GitHub Issues](https://github.com/curioustushar/blog/issues)

---

**This is active research.** Findings are preliminary and subject to revision as experiments progress.
