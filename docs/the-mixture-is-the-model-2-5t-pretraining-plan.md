# The Mixture Is the Model: A 2.5T Pretraining Plan

**Author:** Tushar Gupta · **ERA V5 Submission**

The mixture is the model. Seven capability lanes, Indic provenance tiers, OPUS floors, separate anneal preset (125B / 5%), proxy experiment executed, cleaning progress toward P0/P1.

**Live blog:** [curioustushar.github.io/blog/posts/the-mixture-is-the-model-2-5t-pretraining-plan/](https://curioustushar.github.io/blog/posts/the-mixture-is-the-model-2-5t-pretraining-plan/)  
**Proxy results:** [scripts/mixture_proxy_results.json](../scripts/mixture_proxy_results.json)

---

## Step 1: Goal

A **mixture-and-curriculum plan** decides what the model becomes. Clean shards from the [cleaning pipeline](https://curioustushar.github.io/blog/posts/data-cleaning-strategies-applied/) are raw material; the mixture decides whether V5 becomes a Codex-class agent, a controllable reasoner, and a native Indic speaker — or another English-heavy web model that happens to live in India.

The core claim: **the same corpus and compute produce different models depending on mixture and curriculum.** This document is a defended specification — percentages tied to inventory supply, benchmarks, protected floors, anneal reservation, and a cheap proxy experiment.

Three failures this plan explicitly guards against:

1. **Wishful accounting** — assigning 16% Indic when verified native supply covers ~4%.
2. **Selector starvation** — OPUS with an English-heavy proxy rejecting Indic and agentic batches (V4 fixed Indic with an 8% always-on lane; V5 extends protection to agentic and reasoning).
3. **Spending Tier-A data too early** — consuming SWE-Gym trajectories in stage 1, leaving nothing for anneal.
4. **Flat mixing** — feeding hard reasoning from step zero, or holding it back so long the model never sees it before cooldown.

Assumes all data passes the cleaning pipeline from [Data cleaning strategies applied](https://curioustushar.github.io/blog/posts/data-cleaning-strategies-applied/). Builds on [Train 40B model design](https://curioustushar.github.io/blog/posts/train-40b-model-design/) — especially the **D1–D7 pool layout** and three-tier loader — and [LightningLM](https://lightninglm.theschoolofai.in/).

---

## Step 2: Capability slots (main pretraining mixture)

The mixture composer uses **seven lanes** that renormalize to 100%. The **V5 pretraining preset** below is the main run — general web stays largest because supply is abundant; scarce lanes stay small **on purpose** and concentrate in anneal (Step 8).

**Total budget: 2.5T tokens**

| Capability | % | Tokens @ 2.5T | Purpose | Why this % |
|------------|--:|--------------:|---------|------------|
| **General web** | 34% | 850B | Fluency, world knowledge, MMLU | FineWeb-Edu + DCLM alone exceed 3.9T supply; largest lane because data is abundant |
| **Code** | 24% | 600B | Programming, patches, repo edits | Stack v2 supply ~1.1T; V4 ramped code 13→35% across stages — 24% main-run target is aggressive but inventory-backed |
| **Indic** | 16% | 400B | Native multilingual fluency | Cannot be met from verified native alone — requires tier mix + repetition (Step 3). Floor ≥12% under OPUS |
| **STEM / math** | 12% | 300B | OpenWebMath, peS2o, proof-pile, D4 STEM | Supports GSM8K/MATH alongside dedicated reasoning traces |
| **Reasoning traces** | 6% | 150B | CoT *structure* in pretrain | Supply ~85B — requires repetition. Full reasoning + RLVR comes in **post-training**, not this slot |
| **Long-context** | 6% | 150B | 8K–32K packed sequences | Repo-packed code + book-length corpora ~100B supply |
| **Agentic / tool-use** | 2% | 50B | JSON tool calls, short trajectories | Supply **627M tokens** — lane must be **built, not collected**. Floor ≥2% under OPUS |
| **Total** | **100%** | **2.5T** | | |

**Within-lane composition (not separate slots):**

- **STEM / math (12%):** ~8% STEM prose (peS2o, proof-pile, D4 STEM) + ~4% dense math (OpenWebMath, NuminaMath without CoT). Reasoning traces (6%) carry CoT math separately.
- **General web (34%):** ~2% of **total run** (~50B tokens) is instruction-formatted (FLAN / OpenOrca slices) carved from the web lane — seeds format compliance before SFT without an eighth capability slot.

**V4 precedent:** Web fell 72→18%, code 13→35%, STEM 7→39%, always-on lane pinned at **8%** throughout. V5 floors (`Indic ≥ 12%`, `Agentic ≥ 2%`) are the explicit always-on extension.

---

## Step 3: Indic provenance tiers

**Indic = 16% of total = 400B tokens.** Split across provenance tiers — not a single headline number:

| Tier | Provenance label | % of Indic slice | % of total | Tokens | Primary sources |
|------|-----------------|------------------:|-----------:|-------:|-----------------|
| **A — Verified native** | Confirmed quality | 40% | 6.4% | 160B | Sangraha verified (64B), Indic Wikipedia, PIB/news wire |
| **B — Unverified crawl** | Crawl volume | 25% | 4.0% | 100B | Sangraha unverified (24B), IndicCorpV2 (21B), cohort crawl |
| **C — Translated** | Parallel / MT | 20% | 3.2% | 80B | Samanantar (2B), BPCC parallel (3B), NLLB-translated FineWeb-EDU |
| **D — Synthetic** | Generated | 15% | 2.4% | 60B | Sangraha synthetic (162B supply — quality-filtered subset only) |
| **Total** | | **100%** | **16%** | **400B** | Slot inventory total ~276B unique → **1.45× repetition** at 2.5T |

**Critical honesty:** Sorting Indic inventory by **verified native tokens** (not headline dataset size) shows verified tier cannot fill 16% alone. The 40/25/20/15 split declares exactly how much crawl, translation, and synthesis buys the gap.

**Native-script example (tier A):** *भारत एक संघीय लोकतंत्र है जिसमें राज्यों को अपने विधायी क्षेत्र में शक्तियाँ प्राप्त हैं।* — PIB civics, Sangraha verified.

**Tier trade-offs**

| Tier | Advantage | Disadvantage |
|------|-----------|--------------|
| Verified (A) | Best cultural accuracy, lowest hallucination | Small supply; repetition required |
| Unverified (B) | Scale; authentic register | Crawl noise, mislabeled scripts |
| Translated (C) | Fast way to cover STEM/code in Indic | Translationese, idiom loss |
| Synthetic (D) | Unlimited format control (CoT, JSON) | Quality variance; needs aggressive cleaning |

Translated tier capped at 20% of Indic; validated by native-speaker spot audit (500 doc sample per script).

---

## Step 4: Map capability slots to datasets

Only datasets from the inventory catalog. Supply figures are **approximate unique tokens** after cleaning.

| Capability | Dataset | Tokens (supply) | Tier | Benchmark target |
|------------|---------|----------------:|------|------------------|
| **Code** | The Stack v2 | 900B | B | HumanEval, MBPP |
| | D3 Code (V4 corpus) | 199B | B | — |
| | CommitPack / CommitPackFT | 4B | B | — |
| **Agentic** | SWE-Gym | 150M | A | SWE-bench |
| | OpenHands rollouts | 90M | A | SWE-bench |
| | SWE-smith | 120M | A | SWE-bench |
| | ToolACE, xLAM, Nexus, Hermes FC | 25–80M each | A/D | BFCL, tau-bench |
| | ToolBench, Glaive FC v2 | 50–80M | D | BFCL |
| **Reasoning** | AON (V4 corpus) | 78B | A | GSM8K, BBH |
| | OpenR1-Math, OpenThoughts2, NuminaMath, OpenMathReasoning | 0.5–3B each | A/D | GSM8K, MATH |
| **Long-context** | Repo-packed code (32K+) | 60B | B | LongBench |
| | Book-length corpora (packed) | 40B | B | LongBench |
| **Indic** | Sangraha synthetic | 162B | D | MILU, IndicGenBench |
| | Sangraha verified | 64B | A | MILU, IndicGenBench |
| | Sangraha unverified | 24B | B | — |
| | IndicCorpV2 | 21B | B | — |
| | Samanantar, BPCC | 2–3B | C | Flores-200 |
| **STEM / web** | FineWeb-Edu | 1.3T | B | MMLU |
| | DCLM-Baseline | 2.6T | B | MMLU |
| | D1/D2 Web (V4), D4 STEM (V4), peS2o, proof-pile-2 | 42–627B | B/A | MMLU, ARC |

**Anneal reserve (Tier A only, held out of main run):** SWE-Gym, OpenHands, Nexus, Sangraha verified, AON reasoning — estimated **~80B unique tokens** reserved for final 125B-token anneal phase.

**D1–D7 pool mapping** (from [Train 40B model design](https://curioustushar.github.io/blog/posts/train-40b-model-design/)):

| Pool | Capability lane | Primary inventory fill |
|------|-----------------|------------------------|
| D1 / D2 | General web | FineWeb-Edu, DCLM, D1/D2 Web (V4) |
| D3 | Indic | Sangraha tiers, IndicCorpV2, Samanantar |
| D4 | Code | Stack v2, D3 Code, CommitPack |
| D5 | STEM / math | peS2o, proof-pile-2, D4 STEM, OpenWebMath |
| D6 | Agentic | SWE-Gym, OpenHands, ToolACE, xLAM |
| D7 | Reasoning | AON, OpenR1-Math, NuminaMath, OpenThoughts2 |

The three-tier loader applies: **~45% scored web** (OPUS-ranked D1/D2), **~55% always-on guaranteed** (D3–D7 floors), **held-out golden proxy** (MMLU/GSM8K/HumanEval val splits — never trained on).

---

## Step 5: Capability → benchmarks

| Capability | Benchmarks | What it measures | Success @ 2.5T |
|------------|------------|------------------|----------------|
| General web | **MMLU** (5-shot), **HellaSwag**, **ARC** | Broad knowledge & commonsense | ≥75 / ≥80 / ≥65 |
| Code | **HumanEval**, **MBPP** | Functional code | ≥65 / ≥70 pass@1 |
| STEM / math | **GSM8K**, **MATH** | Math reasoning | ≥70 / ≥35 |
| Reasoning | **BBH** (subset), GSM8K | Multi-step logic | ≥45 avg / ≥70 |
| Agentic | **SWE-bench**, **BFCL**, **tau-bench**, **GAIA**, **BrowseComp** | Tools, patches, agents | SWE-bench Lite ≥25 |
| Long-context | **LongBench** | Multi-doc QA | ≥45 avg |
| Indic | **MILU**, **IndicGenBench**, **IndicGLUE**, **Flores-200** | Indic knowledge, generation, translation | Top cohort quartile |

**Loss-map discipline:** In agentic trajectories, **tool responses are context (grey, no loss)**; model-generated tool calls and actions receive loss (green). Training on tool outputs teaches hallucinated results, not tool use.

---

## Step 6: Data availability & supply check

Inventory totals vs main-run allocation at **2.5T** (and **2T** reference):

| Lane | Target @ 2.5T | Target @ 2T | Unique supply | Status | Mitigation |
|------|-------------:|------------:|--------------:|--------|------------|
| General web | 850B | 680B | 4.8T | ✅ Covered | Subsample; OPUS selects top 40% |
| Code | 600B | 480B | 1.1T | ✅ Covered | Unique tokens sufficient |
| Indic | 400B | 320B | 276B | ⚠️ Repeat | 1.45× @ 2.5T; tier C/D synthesis for verified gap |
| STEM / math | 300B | 240B | ~250B | ✅ Covered | peS2o + proof-pile + D4 STEM |
| Reasoning | 150B | 120B | 85B | ⚠️ Repeat | 1.76× @ 2.5T on AON + NuminaMath |
| Long-context | 150B | 120B | 100B | ⚠️ Repeat | 1.5× @ 2.5T; pack repo contexts |
| Agentic | 50B | 40B | **0.63B** | 🔴 Synthesize | **Binding constraint.** ≥98% must be synthetic/augmented |

**OPUS note:** Retains ~40% of candidate batches, ~6× effective token value, ~few % compute overhead. Floors guarantee Indic and agentic survive selection even when proxy undervalues them.

---

## Step 7: Protected always-on floor

**Definition:** Minimum per-batch share OPUS **cannot** reduce, regardless of selector score.

| Lane | Floor | Rationale |
|------|------:|-----------|
| **Indic** | **≥ 12%** | V4 always-on was 8%; V5 raises to 12% for nine-script coverage |
| **Agentic** | **≥ 2%** | Matches main-run allocation; prevents selector from zeroing a 627M-token supply lane |

**Curriculum ramp minimums** (monitored, not OPUS hard floors — the loader enforces these during stage transitions):

| Lane | S1 minimum | S3 target | Notes |
|------|----------:|----------:|-------|
| Code | 18% | 24% | Ramps with capability stage 2 |
| Reasoning | 5% | 6–8% | CoT format enters stage 2; hold below 8% until stage 3 |
| STEM / math (dense) | 4% of batch | 12% lane total | Pure math slice within D5 pool |

**Implementation:** Loader satisfies hard floors first, then curriculum ramp minimums, then fills with OPUS-scored D1/D2/web. Log batch composition every 1K steps.

**What floors are not:** They are minimum guarantees, not targets. Main-run Indic target is 16%; floor is the cliff OPUS cannot cross.

---

## Step 8: Anneal reserve & separate anneal mixture

Annealing is a **separate preset**, not a relabeling of the main mix. General web falls sharply; scarce lanes upsample for the final cooldown at reduced learning rate.

**Reserve:** Tier-A shards listed in Step 4 are **excluded from ordinary sampling** in stages 1–3. Estimated **80B unique tokens** held back.

**Anneal phase:** Final **5% of run = 125B tokens** at 2.5T scale.

**V5 anneal mixture** (composer preset):

| Lane | Main pretrain | Anneal | Δ |
|------|-------------:|-------:|---|
| General web | 34% | **8%** | −26 |
| Code | 24% | 20% | −4 |
| Indic | 16% | **28%** | +12 |
| STEM / math | 12% | 10% | −2 |
| Reasoning | 6% | **18%** | +12 |
| Long-context | 6% | **8%** | +2 |
| Agentic | 2% | **8%** | +6 |

**Why:** Tier-A agentic trajectories (SWE-Gym, OpenHands) and verified Indic deliver disproportionate capability gain per token when the base model already exists. Spending them in stage 1 wastes their marginal value.

**Safety & alignment:** Not a separate pre-training slice (would duplicate web/instruct). Safety-critical curated data enters via **anneal reserve** and post-training DPO — an estimated **~3% of total tokens** (~75B) touch safety-filtered, human-reviewed shards during cooldown and alignment.

---

## Step 9: Curriculum stages

| Stage | Token range | Mixture shift | Objective |
|-------|-------------|---------------|-----------|
| **1 — General foundation** | 0 – 800B | Web 45%, code 18%, Indic 14%, STEM 10%, rest minimal | Language, facts, basic code syntax |
| **2 — Capability ramp** | 800B – 1.875T | Web 30%, code 26%, Indic 16%, STEM 12%, reasoning 7%, long-ctx 5%, agentic 2% | Approaches V5 pretraining preset |
| **3 — Full preset** | 1.875T – 2.375T | **V5 pretraining preset** (Step 2) | Hold mixture stable; OPUS active |
| **4 — Anneal** | 2.375T – 2.5T | **V5 anneal preset** (Step 8) | Cooldown; spend Tier-A reserve |

**Transition rule (V4 lesson):** Never change mixture in one hard step. Blend over **≥8B-token warmup bands** per transition. V4 saw **150× gradient-norm spike** when Hindi share jumped with frozen embeddings — monitor grad norm at every boundary.

**Reasoning placement:** Long CoT traces enter stage 2–3 pretrain for *format* learning. Controllable reasoning depth (low/medium/high/ultra) is taught later via SFT + RLVR — not by dumping ultra traces into stage 1.

---

## Step 10: Difficulty bands

| Band | Description | Dataset example | Prompt example |
|------|-------------|-----------------|----------------|
| **Easy** | Single-hop, short context | FineWeb-Edu paragraph | Complete: "The capital of Karnataka is ___" |
| **Medium** | Multi-sentence, one reasoning step | NuminaMath (no CoT) | "A train travels 60 km/h for 2.5 hours. Distance?" |
| **Hard** | Multi-step CoT, code crossover | OpenR1-Math trace | GSM8K with visible `<think>` chain |
| **Expert** | Long-horizon agentic, repo-scale | SWE-Gym trajectory | "Fix failing test in `auth.py`" — patch + test log (loss on patch only) |
| **Expert (long-ctx)** | Multi-doc packed context | Repo-packed 32K+ diff | "Summarize all API changes under `auth/` across this 28K-token PR diff" |

**Stage mix:** S1: 70% easy / 25% medium / 5% hard. S2: 30/40/25/5. S3: 15/25/40/20. Anneal: 0/20/40/40.

---

## Step 11: Reasoning-length bands (effort dial)

Reasoning is a **distribution of trace lengths**, not one uniform slot. The effort dial (low / medium / high / ultra) is earned by exposure across lengths; post-training RLVR turns exposure into a controllable inference setting.

| Band | Tokens | Effort level | Description | Example |
|------|-------:|--------------|-------------|---------|
| **Short** | 50 – 200 | Low | Near-direct answer | "2+2=4" — no visible chain |
| **Medium** | 200 – 800 | Medium | Brief justification | One-paragraph why answer holds |
| **Long** | 800 – 3K | High | Derive + check steps | MetaMath-style full CoT |
| **Very long** | 3K – 12K | Ultra | Deliberate + verify + alternatives | OpenR1-Math hard with self-correction |

**Example (same problem, four levels):** *What is 17 × 24?*

- Low (~30 tok): "408"
- Medium (~120 tok): "17×24 = 17×20 + 17×4 = 340+68 = 408"
- High (~600 tok): Step-by-step with intermediate checks
- Ultra (~2K tok): Explores alternative decomposition, verifies by division

Pretrain mixture bins traces by length across math, code, and general domains. Missing a bin means that effort tier **does not exist** at inference.

---

## Step 12: Proxy experiment

**Hypothesis:** V5 pretraining preset with OPUS + floors beats a naive web-heavy preset (web 55%+) on capability benchmarks without MMLU collapse — and anneal reserve improves final 5% lift.

| Parameter | Value |
|-----------|-------|
| Model (full plan) | **3B dense** (24L, 2560 dim) — 3 seeds × 3 conditions |
| Tokens (full plan) | **1B per run** |
| Screen (executed) | Char-LM smoke proxy + loader simulation — [scripts/mixture_proxy_experiment.py](../scripts/mixture_proxy_experiment.py) |
| Conditions | **A:** Naive web-heavy (web 55%, no floors). **B:** V5 pretrain preset + floors. **C:** Anneal-style preset |

### Results (executed 2026-07-31)

**Phase 1 — Loader composition simulation** (5,000 batches × 1,000 tokens; confirms batch sampler hits targets):

| Lane | Target (B) | Observed (B) | Target (C) | Observed (C) | Floors OK |
|------|----------:|-------------:|----------:|-------------:|-----------|
| Web | 34% | **33.8%** | 8% | **8.0%** | — |
| Code | 24% | **24.0%** | 20% | **20.0%** | — |
| Indic | 16% | **16.0%** | 28% | **28.0%** | ✅ ≥12% |
| Agentic | 2% | **2.2%** | 8% | **8.0%** | ✅ ≥2% |

**Verdict:** Loader spec **confirmed** — mixtures land within ±0.5% of target; Indic and agentic floors never violated across 5M simulated tokens.

**Phase 2 — Char-LM smoke proxy** (500 train steps, ~250k chars/condition; lane-specific val perplexity ↓ = better):

| Val set | A (web-heavy) PPL | B (V5 preset) PPL | C (anneal) PPL | Direction |
|---------|------------------:|------------------:|---------------:|-----------|
| Web | **111.2** | 117.8 | 99.9 | C best (web↓ in anneal mix) |
| Code | **16.4** | 26.3 | 32.7 | A best (smoke scale; 24% code insufficient at 500 steps) |
| Indic | **150.7** | 174.8 | 567.9 | A best (tiny char-LM; not representative) |
| STEM | **34.0** | 35.7 | 31.9 | C best |

**Verdict:** Phase 2 is a **smoke test only** — char-level LM at 500 steps cannot validate MMLU/HumanEval claims. Phase 1 confirms the **hypothesis under test at this scale is loader correctness**. Full **3B @ 1B-token** run with MMLU-500 / HumanEval / MILU / BFCL remains the confirm/refute gate before full-scale training.

**Metrics & decision rules (full 3B proxy — pending):**

| Metric | Benchmark | Confirm B | Refute B |
|--------|-----------|-----------|----------|
| Knowledge | MMLU-500 | B ≥ A − 1 pt | B < A − 3 pt |
| Code | HumanEval | B ≥ A + 10 | B ≤ A + 3 |
| Math | GSM8K-200 | B ≥ A + 8 | B ≤ A + 2 |
| Indic | MILU subset | B ≥ A + 5 | B ≤ A + 1 |
| Agentic | BFCL-50 | B ≥ A + 8 | B ≤ A + 2 |
| Anneal lift | All above | C ≥ B + 2 avg | C ≤ B + 0.5 |

**If refuted:** Reduce Indic target 16→14%, shift to web; re-proxy at 500M continuation. If agentic shows no lift, cut to 1% floor and invest in synthetic trace quality before restoring.

**Status:** Phase 1 **confirmed**. Phase 2 smoke **executed** (results above). Full 3B @ 1B **scheduled** before full-run lock.

Raw JSON: [scripts/mixture_proxy_results.json](../scripts/mixture_proxy_results.json)

---

## Step 13: Data cleaning plan (starved slots)

Cleaning targets slots the supply check flags. Progress toward mixture needs at **2.5T** scale:

| Priority | Slot | Mixture need | Cleaned & manifest-ready | Gap | Progress |
|----------|------|-------------:|-------------------------:|----:|----------|
| **P0** | Agentic | 50B | **0.63B** raw inventory (627M); 0B cohort-synth w/ manifest | ~49.4B | **1.3%** of raw only — synthesis not started |
| **P1** | Indic verified | 160B (tier A) | **64B** Sangraha verified in inventory; PIB/wiki ingest pending | ~96B | **40%** of tier-A target in catalog; full 9-strategy pass pending |
| **P2** | Reasoning | 150B | **85B** AON inventory + **15.1M** AyAI rows cleaned ([cleaning blog](https://curioustushar.github.io/blog/posts/data-cleaning-strategies-applied/): 22.2M→15.1M, ~68% survival, manifests generated) | ~65B @ 2.5T | Pipeline **proven** on 22M-row set; scale-up in progress |
| **P3** | Indic synthetic | 60B (tier D) | 162B raw Sangraha synthetic; subset not yet quality-filtered to tier-D cap | TBD | Tier-D cap (15% of Indic) not yet enforced in loader |
| **P4** | Instruction format | ~50B (from web) | 0B FLAN/OpenOrca slices curated | ~50B | Not started |

**Data-gating:** No shard enters the D3–D7 loader without a cleaning manifest (see [9 Data cleaning strategies](https://curioustushar.github.io/blog/posts/data-cleaning-strategies-applied/)). Cohort mixture review applies only after P0/P1 manifests exist for starved lanes.

**Actions:**

| Priority | Slot | Action |
|----------|------|--------|
| **P0** | Agentic | Generate execution-validated SWE traces; clean ToolACE/xLAM; apply 9-strategy pipeline |
| **P1** | Indic verified | Ingest Sangraha verified + PIB; global dedup; script-aware filter |
| **P2** | Reasoning | Clean OpenR1/OpenThoughts via quality classifier; dedup near-duplicates |
| **P3** | Indic synthetic | Cap Sangraha synthetic at tier-D share; quality ≥0.65 |
| **P4** | Instruction format | Curate FLAN/OpenOrca instruct slices from web lane; dedup against SFT holdout |

All shards require a cleaning manifest. **No manifest → no loader access.**

---

## Step 14: Final mixture summary

### Main pretraining (2.5T)

| Capability | % | Key dataset | Benchmark | Supply note |
|------------|--:|-------------|-----------|-------------|
| General web | 34% | FineWeb-Edu, DCLM | MMLU | Abundant |
| Code | 24% | Stack v2 | HumanEval | 1.1T supply |
| Indic | 16% | Sangraha tiers A–D | MILU, IndicGenBench | 276B → repeat |
| STEM / math | 12% | peS2o, proof-pile, D4 | GSM8K | ~250B |
| Reasoning | 6% | AON, NuminaMath | BBH | 85B → repeat |
| Long-context | 6% | Repo-packed 32K+ | LongBench | 100B → repeat |
| Agentic | 2% | SWE-Gym + **synthetic** | SWE-bench, BFCL | 627M → **build** |
| **Total** | **100%** | | | |

### Anneal (125B, separate preset)

| Capability | % | Key dataset | Notes |
|------------|--:|-------------|-------|
| Indic | 28% | Sangraha verified (reserved) | Tier A |
| Reasoning | 18% | AON (reserved) | Tier A |
| Agentic | 8% | SWE-Gym, OpenHands (reserved) | Tier A |
| Code | 20% | CommitPackFT, test-heavy Stack | |
| Long-context | 8% | Repo-packed | |
| STEM | 10% | peS2o | |
| Web | 8% | FineWeb-Edu top quality | |

---

## Step 15: Final justification

This plan is **balanced** because scarce lanes stay small in the main run (honest about 627M agentic tokens) while protected floors and anneal concentration ensure they are not starved. It is **defensible** because every percentage maps to inventory supply, a named benchmark, and a floor or reserve rule.

The **curriculum** ramps web-first, blends mixture changes over 8B-token bands, and saves Tier-A data for anneal — matching V4's staged rebalancing without V4's hard-step instability.

The **proxy experiment** tests the highest-risk claim at 400× lower cost than full scale. **Phase 1 (loader simulation) confirmed** mixture targets and floors over 5M simulated tokens; **Phase 2 smoke run executed**; full 3B @ 1B remains the benchmark gate.

**Limitations:** Agentic synthesis quality dominates outcome; verified Indic native tokens remain the tightest tier; reasoning effort dial requires post-training RLVR to become inference-controllable.

**Cleaning continues** toward P0 agentic and P1 Indic verified targets — the mixture defines which slots are starved; cleaning closes the gap.

---

## Related work

- [Train 40B model design](https://curioustushar.github.io/blog/posts/train-40b-model-design/)
- [9 Data cleaning strategies applied](https://curioustushar.github.io/blog/posts/data-cleaning-strategies-applied/)
- [LightningLM 120B](https://lightninglm.theschoolofai.in/)

---
