# Phase 3 Roadmap: Empirical LM Validation

**Status:** Future work (not started)  
**Prerequisite:** Phase 2 complete (feasibility validated)  

---

## One-Line Summary

**Phase 2 demonstrated feasibility of structured byte prediction as a V×d-free output interface under controlled synthetic experiments; practical language-modeling utility remains unvalidated.**

---

## Phase 3 Objective

**Validate whether structured byte prediction offers practical advantages for real language modeling.**

---

## Experimental Design

### 1. Implementation

**PyTorch implementation** of:
- Full Transformer (4 layers, 512 dim, or comparable)
- Three output heads:
  - **Standard softmax** (V×d baseline)
  - **Structured byte head** (d×(L×256) with EOS)
  - **Continuous κ head** (d×D control, if computationally practical)

### 2. Benchmark

**WikiText-103** (or comparable):
- Standard LM dataset
- Well-established baselines
- Meaningful perplexity comparison

### 3. Matched Baseline

**Critical:** Same Transformer backbone for all heads
- Identical: layers, hidden dim, attention, training hyperparameters
- Different: only the output head
- Ensures fair comparison

### 4. Metrics

**Do NOT assume parameter savings = computational savings**

Measure **actual behavior**:
- **Perplexity** (validation + test)
- **Parameter count** (total model size)
- **Memory usage** (peak training, peak inference)
- **Throughput** (tokens/sec training, tokens/sec inference)
- **Training cost** (wall-clock time to convergence)
- **Decoding latency** (time per generated token)

### 5. Scaling Test

**Test toward large vocabularies:**
- V ∈ {10K, 50K, 100K}
- Measure parameter savings vs performance cost
- Identify crossover point (if any)

---

## Key Methodological Point

### Parameter Savings ≠ Computational Savings

**Why:**
- Structured bytes: more classes per position (256 vs typical <100)
- May trade smaller output head for additional decoding computation
- Codec overhead not yet profiled
- GPU optimization patterns unclear

**Therefore:**
- Measure **actual wall-clock** performance
- Don't extrapolate from parameter count alone
- Profile memory carefully (activations, gradients, KV cache)

---

## Success Criteria

### Strong Success

- Perplexity ≈ softmax baseline (within 1-2%)
- Parameter savings realized (5-50x for large V)
- Training/inference speed ≥ 0.8x baseline
- Memory usage ≤ baseline

### Partial Success

- Perplexity slightly worse (<5% degradation)
- Parameter savings substantial (>10x for V>100K)
- Speed/memory tradeoffs acceptable for specific use cases

### Failure

- Perplexity significantly worse (>5%)
- OR: Speed/memory costs outweigh parameter savings
- OR: Doesn't scale to large V

---

## Hypotheses to Test

1. **H1:** Structured bytes achieve comparable perplexity to softmax
2. **H2:** Parameter savings translate to memory savings
3. **H3:** Computational overhead is acceptable
4. **H4:** Advantages grow with vocabulary size

---

## Experiment Timeline (Estimated)

**Assuming PyTorch + GPU access:**

1. Implementation: 1-2 weeks
2. Initial WikiText runs: 1 week
3. Hyperparameter tuning: 1-2 weeks
4. Scaling experiments: 1-2 weeks
5. Analysis + writeup: 1 week

**Total: 5-8 weeks**

---

## Risks & Mitigations

### Risk 1: Continuous κ head infeasible

**Mitigation:** Drop it if too slow, focus on softmax vs structured bytes

### Risk 2: Decoding overhead dominates

**Mitigation:** Profile early, optimize codec if needed

### Risk 3: Hyperparameters don't transfer

**Mitigation:** Grid search, treat as separate architecture

---

## Open Design Questions

### 1. Autoregressive vs Independent Bytes

**Phase 2 used:** Independent (parallel) prediction  
**Phase 3 should test:** Both strategies
- Independent: Predict all bytes in parallel
- Autoregressive: Predict left-to-right with context

### 2. Byte Representation Length

**Phase 2 tested:** L ≤ 32  
**Phase 3 should determine:** Optimal L for real tokens
- UTF-8: Most tokens ≤ 4 bytes
- Option: Fixed L=8 with padding
- Option: Variable L with EOS

### 3. Loss Weighting

**Question:** Should byte positions have equal loss weight?
- Option 1: Uniform (all positions weighted equally)
- Option 2: Weighted (longer tokens get more weight)
- Option 3: Length-normalized

---

## Deliverables

1. **PyTorch implementation** (open source)
2. **Benchmark results** (perplexity, speed, memory)
3. **Scaling plots** (performance vs vocabulary size)
4. **Technical report** (Phase 3 findings)
5. **Paper draft** (if results warrant publication)

---

## Non-Goals

**Do NOT attempt in Phase 3:**
- Novel architectures beyond output head
- Novel training procedures
- Novel datasets
- Production optimization

**Focus:** Clean, controlled comparison on standard benchmark.

---

## Connection to Phase 2

### What Phase 2 Established

✅ Feasibility (100% exact reconstruction, synthetic)  
✅ Architectural compatibility  
✅ No fundamental barriers  

### What Phase 3 Must Establish

❓ Practical utility (real LM performance)  
❓ Computational efficiency  
❓ Scalability  

---

## Decision Point After Phase 3

### If successful:
→ Phase 4: Production engineering, optimization, deployment

### If marginal:
→ Revisit design (autoregressive, loss weighting, architecture)

### If unsuccessful:
→ Document negative result, move to other research directions

---

## Phase 3 Summary

**Objective:** Validate practical utility of structured byte prediction  
**Method:** PyTorch implementation + WikiText-103 benchmark  
**Timeline:** 5-8 weeks  
**Success metric:** Comparable perplexity with realized parameter savings  

---

**Phase 2 is frozen. Phase 3 is well-defined. Clean handoff complete.** ✅
