# Kronecker V2: Invertible Token Representations

> Can we eliminate vocabulary-sized output heads in language models?

[![Status](https://img.shields.io/badge/Phase_0-Complete-success)]() [![Status](https://img.shields.io/badge/Phase_1-Complete-success)]() [![Status](https://img.shields.io/badge/Phase_2-Complete-success)]() [![Status](https://img.shields.io/badge/Phase_3-Designed-blue)]() [![Python](https://img.shields.io/badge/python-3.8+-blue)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

**TL;DR:** We demonstrate that structured byte prediction can eliminate V×d output parameters in transformer language models, achieving 100% exact reconstruction on synthetic tasks with up to 512× parameter savings. Practical utility for real language modeling remains an open empirical question.

---

## Overview

Standard language models use a vocabulary-sized output projection: `h → softmax(V×d) → token`. For large vocabularies (V > 100K), this becomes a massive parameter bottleneck.

**This project explores:** Can we replace it with structured byte prediction?
```
h → bytes(d×L×256) → κ → algebraic_decode → token
```

### Key Results

| Phase | Question | Result |
|-------|----------|--------|
| **Phase 0** | Is κ injective? | ✅ **YES** - 0/89 collisions; algebraic decoder viable |
| **Phase 1** | Does it scale? | ✅ **YES** - 0 collisions at 1K–50K; 98.9% unseen decode |
| **Phase 2** | Is it learnable? | ✅ **YES** - 100% reconstruction (synthetic) |
| **Phase 3** | Is it practical? | ⏳ **TBD** - Designed, not executed |

---

## Phase 0: Mathematical Feasibility

**Question:** Can Kronecker token embeddings be inverted algebraically, eliminating the V×d output head?

**Approach:** 30-minute falsification experiment — test for collisions, identify information-loss stages, sketch an algebraic decoder.

### Findings

| Stage | Reversible? | Notes |
|-------|-------------|-------|
| UTF-8 bytes | ✅ Yes | Deterministic |
| Raw Kronecker κ(b) | ✅ Yes | 0/89 collisions; sparse O(D) decode |
| Length norm (1/√L) | ⚠️ Partial | Recoverable from magnitude |
| Z-normalization | ❌ Initially | Mean μ subtracted and lost |
| Projection W·κ̄ | ⚠️ Untested | Needed large-scale validation |

**Key result:** Raw Kronecker appears injective over distinct byte sequences (len ≤ 32). An algebraic sparse-recovery decoder runs in O(D), independent of vocabulary size V.

**Parameter implication (V=1M, d=768):** 768M softmax params → ~6.3M projection params (~122× reduction).

**Limitations:** Small test set (89 tokens); z-norm and projection not yet validated at scale.

📄 **Full write-up:** [Phase 0 blog post](https://curioustushar.github.io/blog/posts/kronecker-v2-invertibility-phase0/)

---

## Phase 1: Scale Validation

**Question:** Does injectivity hold at real vocabulary scales, through the full pipeline (z-norm + projection)?

**Approach:** Collision tests at 1K–50K tokens; algebraic decoder on training and unseen tokens; decoder scaling benchmarks.

### Findings

| Aspect | Result |
|--------|--------|
| Raw Kronecker injectivity | ✅ 0 collisions (1K, 10K, 50K) |
| Z-norm preserves uniqueness | ✅ 0 collisions (revised Phase 0 hypothesis) |
| Projection preserves uniqueness | ✅ 0 collisions at d=768 |
| Algebraic decoder (training) | ✅ 100% exact reconstruction |
| Algebraic decoder (unseen) | ✅ 98.9% exact reconstruction |
| Decoder scaling | ✅ O(D) — ~35μs, ~23,440× faster than O(V) search |

**Key insight:** Z-norm is not globally invertible over ℝ^D, but preserves uniqueness over the restricted Kronecker vocabulary subset (sparse, structured index patterns).

**Verdict:** GREEN — strong preliminary evidence that algebraic decoding is viable at tested scales.

📄 **Full write-up:** [Phase 1 blog post](https://curioustushar.github.io/blog/posts/kronecker-v2-phase1-scale-validation/) · [Technical report](docs/phase1_report.md)

---

## What We Proved (Phase 2)

📄 **Full write-up:** [Phase 2 blog post](https://curioustushar.github.io/blog/posts/kronecker-v2-phase2-complete/)

### ✅ Feasibility Validated

**Structured byte prediction works:**
- 100% exact accuracy across all lengths (L=1 to 32)
- Zero p^L degradation
- Variable-length functional (EOS mechanism)
- Codec 100% reliable
- Transformer-compatible

### Parameter Comparison

| Vocabulary | Softmax | Structured Bytes | Savings |
|------------|---------|------------------|---------|
| 10K        | 5.1M    | ~1M              | **5×**  |
| 50K        | 25.6M   | ~1M              | **26×** |
| 100K       | 51.2M   | ~1M              | **51×** |
| 1M         | 512M    | ~1M              | **512×** |

**Complexity:** O(L) instead of O(V) where L = max token byte length.

---

## Quick Start

### Reproduce Phase 2 Results

```bash
# Clone repository
cd kronecker-v2-invertibility/experiments/phase2

# Run validation tests (< 5 minutes total)
python phase2c_minimal.py      # Interface validation (30s)
python phase2d_ultra_fast.py   # Length scaling (80s)
python phase2e_fixed.py        # Variable-length (30s)
```

**Requirements:** Python 3.8+, numpy

**Expected output:** 100% exact accuracy on all tests.

---

## How It Works

### The Problem: Continuous κ Fails

**Phase 2A tried:** `h → κ_pred` (continuous regression)

❌ **Result:** 0% reconstruction - neural noise destroyed the sparse structure needed by the algebraic decoder.

### The Solution: Discrete Factors Work

**Phase 2C approach:** `h → bytes` (discrete classification)

✅ **Result:** 100% reconstruction - predict the discrete bytes that *construct* κ, guaranteeing valid structure.

### Pipeline

```python
# 1. Predict bytes (discrete)
byte_logits = model(h)  # (batch, L, 256)
bytes = argmax(byte_logits)

# 2. Construct κ (exact)
κ = construct_kappa(bytes)  # Deterministic, no noise

# 3. Decode token (algebraic)
token = algebraic_decode(κ)  # Works perfectly
```

---

## Project Structure

```
kronecker-v2-invertibility/
├── src/
│   └── kronecker_encoder.py          # Algebraic decoder
├── experiments/
│   └── phase2/
│       ├── phase2c_minimal.py         # Interface validation
│       ├── phase2d_ultra_fast.py      # Length scaling
│       ├── phase2e_fixed.py           # Variable-length
│       └── phase2f_transformer.py     # Transformer integration
├── docs/
│   ├── phase1_report.md               # Scale validation
│   ├── phase2_design.md                 # Phase 2 initial design
│   ├── phase2_findings.md               # Continuous κ failure
│   ├── phase2b_structured_prediction.md # Discrete factor design
│   ├── phase2_verdict.md                # Scientific verdict
│   ├── phase2_final_summary.md          # Technical report
│   ├── phase3a_specification.md         # Phase 3 protocol
│   └── ...
├── PROJECT_STATUS.md                  # Master document
└── README.md                          # This file
```

---

## Phase 2 Results

### Fixed-Length (Phase 2C)

**Task:** Predict 8 bytes for each token  
**Model:** Simple linear  
**Result:** 100% exact accuracy

### Length Scaling (Phase 2D)

**Task:** Test L ∈ {1, 2, 4, 8, 16, 32}  
**Result:** 100% exact for ALL lengths

| L  | Exact Acc | Convergence |
|----|-----------|-------------|
| 1  | 100%      | Fast        |
| 2  | 100%      | Fast        |
| 4  | 100%      | Fast        |
| 8  | 100%      | Moderate    |
| 16 | 100%      | Slow        |
| 32 | 100%      | Slow        |

**Key finding:** No p^L degradation observed.

### Variable-Length (Phase 2E)

**Task:** Mixed lengths (L=1-8) with EOS token  
**Result:** 100% length and content prediction

### Transformer Integration (Phase 2F)

**Task:** Attach structured byte head to a 2-layer Transformer backbone  
**Model:** Mini Transformer + byte head (numpy)  
**Result:** Architecturally compatible; codec 100% functional on all outputs

| Check | Result |
|-------|--------|
| Byte head attaches to Transformer hidden states | ✅ Pass |
| `bytes → κ → decode → bytes` on model outputs | ✅ 100% |
| Full end-to-end Transformer training | ⚠️ Not demonstrated |

**Note:** Full backprop through the Transformer was not practical in numpy (head-only training did not learn). Architectural compatibility is proven; end-to-end LM training awaits Phase 3 (PyTorch).

📄 **Script:** [experiments/phase2/phase2f_transformer.py](experiments/phase2/phase2f_transformer.py)

---

## What We Didn't Prove

Phase 2 established **feasibility**, not **utility**.

**Still unknown:**
- ❓ Perplexity on real LM tasks
- ❓ Speed vs softmax baseline
- ❓ Memory efficiency in practice
- ❓ Generation quality
- ❓ Scaling to V > 100K

**These require Phase 3** (PyTorch + GPU + WikiText-103).

---

## Phase 3: Designed, Not Executed

### Three-Stage Protocol

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

### Timeline (if continuing)

- Phase 3B: 3-5 days
- Phase 3C: 5-8 weeks
- **Total: ~2 months**

### Requirements

- PyTorch 2.0+
- GPU (A100 or equivalent)
- WikiText-103 dataset

---

## Key Insights

### 1. Discrete > Continuous

Predicting discrete factors (bytes) that construct κ eliminates the noise problem of continuous regression.

### 2. Parameter Scaling

Structured bytes scale with **max token length** (L), not **vocabulary size** (V).

- Good for: Large vocabularies (V > 100K)
- Neutral for: Medium vocabularies
- Overkill for: Small vocabularies (V < 10K)

### 3. No p^L Degradation

Byte errors are independent, not compounding. 100% byte accuracy → 100% exact accuracy.

---

## Methodological Highlights

### Clear Scope

**Phase 2 claim:** Feasibility validated (synthetic)  
**Phase 3 claim:** (none yet - awaiting experiments)

**We don't claim:**
- Better than softmax (unproven)
- Production-ready (not tested)
- Practical utility (requires Phase 3)

### Pre-Registration

Phase 3 protocol defined **before** experiments:
- Prevents p-hacking
- Ensures fair comparison
- Enables replication

### Validation Checkpoints

```
numpy (Phase 2) → PyTorch proxy (3B) → Full LM (3C)
```

Each step validates before adding complexity.

---

## Documentation

### Core Documents

- **[Project Status](PROJECT_STATUS.md)** - Master document
- **[Phase 1 Report](docs/phase1_report.md)** - Scale validation & algebraic decoder
- **[Phase 2 Verdict](docs/phase2_verdict.md)** - Scientific verdict (what was proven)
- **[Phase 2 Summary](docs/phase2_final_summary.md)** - Technical details
- **[Phase 3 Roadmap](docs/phase3_roadmap.md)** - What comes next

### Phase-Specific

- [Phase 0 Blog Post](https://curioustushar.github.io/blog/posts/kronecker-v2-invertibility-phase0/) - Mathematical feasibility
- [Phase 1 Blog Post](https://curioustushar.github.io/blog/posts/kronecker-v2-phase1-scale-validation/) - Scale validation & algebraic decoder
- [Phase 2 Blog Post](https://curioustushar.github.io/blog/posts/kronecker-v2-phase2-complete/) - Structured byte prediction results
- [Phase 2 Design](docs/phase2_design.md) - Initial Transformer integration plan
- [Phase 2 Findings](docs/phase2_findings.md) - Continuous κ regression failure
- [Phase 2B Design](docs/phase2b_structured_prediction.md) - Discrete factor approach
- [Phase 2C Results](docs/phase2c_results.md) - Debugging & validation
- [Phase 2D Results](docs/phase2d_results.md) - Length scaling
- [Phase 3A Specification](docs/phase3a_specification.md) - Experiment protocol
- [Phase 3B Proxy Spec](docs/phase3b_proxy_spec.md) - PyTorch tests
- [Phase 3C Benchmark Spec](docs/phase3c_benchmark_spec.md) - Full benchmark

---

## Citation

If you use this work, please cite:

```bibtex
@misc{kronecker-v2-invertibility-2026,
  title={Structured Byte Prediction for Parameter-Efficient Language Models},
  author={[Your Name]},
  year={2026},
  note={Phase 2: Feasibility validated under synthetic conditions}
}
```

---

## Success Criteria

### Phase 0 (Met) ✅

- [x] Zero collisions on raw Kronecker (89 diverse tokens)
- [x] Algebraic decoder algorithm sketched (O(D) complexity)
- [x] Information-loss stages identified

### Phase 1 (Met) ✅

- [x] Zero collisions at 1K–50K vocabulary scales
- [x] Z-norm and projection preserve uniqueness
- [x] Algebraic decoder: 100% training, 98.9% unseen
- [x] Decoder ~23,000× faster than O(V) search

### Phase 2 (Met) ✅

- [x] Exact reconstruction on synthetic tasks: 100%
- [x] Codec reliability: 100%
- [x] Scaling across lengths: All pass
- [x] Architectural compatibility: Verified

### Phase 3 (Pending)

**Strong success:**
- [ ] Test perplexity ≤ 102% of softmax
- [ ] Memory ≤ 90% OR Throughput ≥ 90%

**Partial success:**
- [ ] Test perplexity ≤ 105%
- [ ] Parameter savings realized, mixed efficiency

---

## Limitations

### Phase 2 Limitations

- ✅ Synthetic tasks only (not real LM)
- ✅ Small scale (V=50, L≤32)
- ✅ Simple models (linear, 2-layer MLP)
- ✅ numpy implementation (no GPU)

### Requires Phase 3

- Real language modeling benchmarks
- Perplexity comparison vs softmax
- Computational efficiency measurement
- Large vocabulary scaling (V > 100K)

---

## Related Work

### Kronecker Embeddings

This project builds on Kronecker product-based token representations:
- **Phase 0:** Mathematical feasibility — injectivity and algebraic decoder
- **Phase 1:** Scale validation — z-norm, projection, decoder benchmarks
- **Phase 2:** Output head replacement — structured byte prediction

### Parameter-Efficient LMs

- Adaptive softmax
- Mixture of softmaxes
- Character-level models
- Byte-level models

**Our contribution:** Structured prediction with algebraic decoding.

---

## Roadmap

### Completed ✅

- [x] Phase 0: Mathematical feasibility (89-token collision test)
- [x] Phase 1: Scale validation (1K–50K, algebraic decoder)
- [x] Phase 2A: Continuous κ regression (failed — informed Phase 2C)
- [x] Phase 2B: Structured byte prediction design
- [x] Phase 2C: Interface validation
- [x] Phase 2D: Length scaling
- [x] Phase 2E: Variable-length
- [x] Phase 2F: Transformer compatibility
- [x] Phase 2 documentation
- [x] Phase 3 specification

### Next Steps ⏳

- [ ] Set up PyTorch environment
- [ ] Implement Phase 3B tests
- [ ] Validate proxy passes
- [ ] Execute Phase 3C benchmark
- [ ] Analyze results
- [ ] Write paper (if successful)

---

## Contributing

This is currently a research project. Contributions welcome after Phase 3 execution.

**Areas for contribution:**
- PyTorch implementation (Phase 3B)
- Benchmark execution (Phase 3C)
- Optimization improvements
- Additional experiments

---

## FAQ

**Q: Is this production-ready?**  
A: No. Phase 2 only proves feasibility. Production readiness requires Phase 3+ validation.

**Q: Should I use this instead of softmax?**  
A: Not yet. Wait for Phase 3 results comparing perplexity and efficiency.

**Q: Does it work for large vocabularies?**  
A: Theoretically yes (parameter savings scale with V). Empirically untested beyond V=50.

**Q: What about generation quality?**  
A: Unknown. Phase 2 tested reconstruction accuracy, not generation.

**Q: Can I reproduce the results?**  
A: Yes! See [Quick Start](#quick-start). All Phase 2 code runs in < 5 minutes.

---

## License

MIT License - see LICENSE file for details.

---

## Contact

For questions about this research:
- Open an issue
- See [Project Status](PROJECT_STATUS.md) for detailed documentation

---

## Acknowledgments

- Problem selected from Kronecker Embedding V2 research agenda
- Built on prior encoder validation work (Phase 1)

---

**Status:** Phases 0–2 complete, Phase 3 designed (not executed)  
**Last updated:** August 14, 2026  
**Next milestone:** Phase 3B PyTorch proxy implementation
