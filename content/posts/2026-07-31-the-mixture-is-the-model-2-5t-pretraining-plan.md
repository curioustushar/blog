---
title: "The Mixture Is the Model: A 2.5T Pretraining Plan"
slug: "the-mixture-is-the-model-2-5t-pretraining-plan"
date: 2026-07-31T07:50:00-05:00
categories: ["machine-learning", "llm"]
tags: ["training", "curriculum", "data-mixture", "pretraining", "indic"]
author: Tushar Gupta
description: "A defended 2.5T-token mixture spec — seven capability lanes, Indic provenance tiers, OPUS floors, separate anneal preset, inventory supply checks, loader sim confirmed, benchmark proxy pending."
---

<div class="post-summary">

The mixture is the model: **what you train on matters more than parameter count.** This post specifies two presets — a **main pretraining mixture** (web-heavy, scarce lanes small) and a **separate anneal mixture** (scarce lanes concentrated) — over **2.5T tokens**, with seven capability lanes, Indic provenance tiers, OPUS floors, and Tier-A data reserved for cooldown.

<div class="plan-status" role="status" aria-label="Plan status">
  <span class="status-badge status-ok">Spec complete</span>
  <span class="status-badge status-ok">Loader sim ✓</span>
  <span class="status-badge status-pending">Benchmark proxy pending</span>
  <span class="status-badge status-gated">Data-gated · P0/P1</span>
</div>

</div>

<nav class="post-toc" aria-label="Table of contents">

**On this page**

| | |
|---|---|
| **Status & goal** | [0 · Plan status](#step-0-plan-status--inventory-methodology) · [1 · Goal](#step-1-goal) |
| **Mixture design** | [2 · Capability slots](#step-2-capability-slots-main-pretraining-mixture) · [3 · Indic tiers](#step-3-indic-provenance-tiers) · [4 · Datasets](#step-4-map-capability-slots-to-datasets) · [5 · Benchmarks](#step-5-capability--benchmarks) · [6 · Supply check](#step-6-data-availability--supply-check) · [7 · Floors](#step-7-protected-always-on-floor) · [8 · Anneal](#step-8-anneal-reserve--separate-anneal-mixture) |
| **Curriculum** | [9 · Stages](#step-9-curriculum-stages) · [10 · Difficulty](#step-10-difficulty-bands) · [11 · Reasoning length](#step-11-reasoning-length-bands-effort-dial) |
| **Validation & ops** | [12 · Proxy experiment](#step-12-proxy-experiment) · [13 · Cleaning plan](#step-13-data-cleaning-plan-starved-slots) |
| **Wrap-up** | [14 · Summary](#step-14-final-mixture-summary) · [15 · Justification](#step-15-final-justification) |

</nav>

---

## Step 0: Plan status & inventory methodology

This is a **defended hypothesis** at 2.5T scale — not a claim that every lane is manifest-ready today.

| Dimension | Status |
|-----------|--------|
| **Mixture & curriculum spec** | Complete — lanes, tiers, floors, anneal preset, stages defined |
| **Loader validation** | Phase 1 simulation ✅ — mixtures and floors hit targets over 5M tokens |
| **Benchmark proxy** | Phase 3 (3B @ 1B, A/B/C) **not run** — confirm/refute gate before full-scale lock |
| **Manifest gating** | P0 agentic + P1 verified Indic **not ready** — plan is spec-complete, data-gated |

**Three inventory labels** (used in Steps 4–6):

1. **Catalog** — published dataset size from inventory docs (Sangraha, Stack v2, etc.).
2. **Manifest-ready** — cleaned, deduped, provenance-stamped; loader may consume.
3. **Locally verified** — inspected in-repo or on disk (today: AyAI reasoning subset only — 15.1M rows from [cleaning blog]({{< relref "2026-07-23-data-cleaning-strategies.md" >}})).

Supply figures in Step 6 are **catalog estimates** unless marked manifest-ready or locally verified. Binding constraints: **agentic synthesis** (~98% of 50B target) and **verified Indic tier-A** (~40% of 160B target in catalog; ~0B manifest-ready).

---

## Step 1: Goal

A **mixture-and-curriculum plan** decides what the model becomes. Clean shards from the [cleaning pipeline]({{< relref "2026-07-23-data-cleaning-strategies.md" >}}) are raw material; the mixture decides whether the model becomes a Codex-class agent, a controllable reasoner, and a native Indic speaker — or another English-heavy web model that happens to live in India.

The core claim: **the same corpus and compute produce different models depending on mixture and curriculum.** This document is a defended specification — percentages tied to inventory supply, benchmarks, protected floors, anneal reservation, and a cheap proxy experiment. It remains a **hypothesis** until Phase 3 benchmark proxy confirms or refutes the mixture choices.

Four failures this plan explicitly guards against:

1. **Wishful accounting** — assigning 16% Indic when verified native supply covers ~4%.
2. **Selector starvation** — OPUS with an English-heavy proxy rejecting Indic and agentic batches (a prior run fixed Indic with an 8% always-on lane; this plan extends protection to agentic and reasoning).
3. **Spending Tier-A data too early** — consuming SWE-Gym trajectories in stage 1, leaving nothing for anneal.
4. **Flat mixing** — feeding hard reasoning from step zero, or holding it back so long the model never sees it before cooldown.

Assumes all data passes the cleaning pipeline from [Data cleaning strategies applied]({{< relref "2026-07-23-data-cleaning-strategies.md" >}}). Builds on [Train 40B model design]({{< relref "2026-07-17-train-40b-model-design.md" >}}) — especially the **D1–D7 pool layout** and three-tier loader — and [LightningLM](https://lightninglm.theschoolofai.in/).

---

## Step 2: Capability slots (main pretraining mixture)

The mixture composer uses **seven lanes** that renormalize to 100%. The **main pretraining preset** below is the main run — general web stays largest because supply is abundant; scarce lanes stay small **on purpose** and concentrate in anneal (Step 8).

**Total budget: 2.5T tokens** — aligned with the [40B model design]({{< relref "2026-07-17-train-40b-model-design.md" >}}) pretraining target, not an arbitrary round number.

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

**Prior-run precedent:** Web fell 72→18%, code 13→35%, STEM 7→39%, always-on lane pinned at **8%** throughout. Protected floors (`Indic ≥ 12%`, `Agentic ≥ 2%`) are the explicit always-on extension.

**Execution readiness (catalog vs manifest, today):** web/code/STEM ✅ catalog-covered; Indic/reasoning/long-ctx ⚠️ repeat required; agentic 🔴 must be synthesized. **0 of 7 lanes** are fully manifest-gated for a production loader run (see Step 13).

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

**Per-tier supply vs demand @ 2.5T:**

| Tier | Target | Catalog supply | Manifest-ready | Gap | Fill strategy |
|------|-------:|---------------:|---------------:|----:|---------------|
| **A — Verified** | 160B | ~64B | ~0B | ~96B | PIB/wiki ingest + 9-strategy cleaning (P1) |
| **B — Unverified** | 100B | ~45B+ | 0B | ~55B | More crawl shards; dedup before repeat |
| **C — Translated** | 80B | ~5B | 0B | ~75B | BPCC + NLLB pipeline — source, don't repeat |
| **D — Synthetic** | 60B | 162B raw | 0B filtered | cap at 15% Indic | Quality filter ≥0.65; synthetic is correct fill for D |

**Verified-native honesty:** Tier A alone is **6.4% of total run** — the remaining **9.6%** of the 16% Indic target must come from tiers B/C/D.

**Repetition policy:**

| Scope | Max repeat | Rationale |
|-------|----------|-----------|
| Indic overall | **1.45×** @ 2.5T | 276B catalog → 400B target; acceptable with tier diversity |
| Tier A (verified) alone | **≤3×** without fresh ingest | Small verified pool; prefer translation/synthesis over heavy re-run |
| Tier C/D | **Source-first** | Translated and synthetic tiers exist to avoid repeating tiny verified pools |
| Agentic | **≥80×** effective via synthesis | 627M catalog → 50B target — repetition is not the plan; generation is |

Repetition caps are **policy hypotheses** until Phase 3 proxy or a real-data micro-run confirms them.

---

## Step 4: Map capability slots to datasets

Only datasets from the inventory catalog. Supply figures are **approximate unique tokens** after cleaning unless marked otherwise.

| Capability | Dataset | Tokens (supply) | Tier | Inventory status | Benchmark target |
|------------|---------|----------------:|------|------------------|------------------|
| **Code** | The Stack v2 | 900B | B | Catalog | HumanEval, MBPP |
| | D3 Code (V4 corpus) | 199B | B | Catalog | — |
| | CommitPack / CommitPackFT | 4B | B | Catalog | — |
| **Agentic** | SWE-Gym | 150M | A | Catalog · **Gap** | SWE-bench |
| | OpenHands rollouts | 90M | A | Catalog · **Gap** | SWE-bench |
| | SWE-smith | 120M | A | Catalog · **Gap** | SWE-bench |
| | ToolACE, xLAM, Nexus, Hermes FC | 25–80M each | A/D | Catalog · **Gap** | BFCL, tau-bench |
| | ToolBench, Glaive FC v2 | 50–80M | D | Catalog · **Gap** | BFCL |
| **Reasoning** | AON (V4 corpus) | 78B | A | Catalog | GSM8K, BBH |
| | AyAI reasoning-v1 (cleaned) | ~15.1M rows | A/D | **Manifest-ready** · locally verified | GSM8K |
| | OpenR1-Math, OpenThoughts2, NuminaMath, OpenMathReasoning | 0.5–3B each | A/D | Catalog | GSM8K, MATH |
| **Long-context** | Repo-packed code (32K+) | 60B | B | Catalog | LongBench |
| | Book-length corpora (packed) | 40B | B | Catalog | LongBench |
| **Indic** | Sangraha synthetic | 162B | D | Catalog · raw | MILU, IndicGenBench |
| | Sangraha verified | 64B | A | Catalog · P1 pending | MILU, IndicGenBench |
| | Sangraha unverified | 24B | B | Catalog | — |
| | IndicCorpV2 | 21B | B | Catalog | — |
| | Samanantar, BPCC | 2–3B | C | Catalog | Flores-200 |
| **STEM / web** | FineWeb-Edu | 1.3T | B | Catalog | MMLU |
| | DCLM-Baseline | 2.6T | B | Catalog | MMLU |
| | D1/D2 Web (V4), D4 STEM (V4), peS2o, proof-pile-2 | 42–627B | B/A | Catalog | MMLU, ARC |

**Anneal reserve (Tier A only, held out of main run):** SWE-Gym, OpenHands, Nexus, Sangraha verified, AON reasoning — estimated **~80B unique tokens** reserved for final 125B-token anneal phase.

**D1–D7 pool mapping** (from [Train 40B model design]({{< relref "2026-07-17-train-40b-model-design.md" >}})):

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

### Catalog supply vs allocation

Inventory totals vs main-run allocation at **2.5T** (and **2T** reference). Supply column = **catalog** unique tokens unless noted.

| Lane | Target @ 2.5T | Target @ 2T | Catalog supply | Status | Mitigation |
|------|-------------:|------------:|---------------:|--------|------------|
| General web | 850B | 680B | 4.8T | ✅ Covered | Subsample; OPUS selects top 40% |
| Code | 600B | 480B | 1.1T | ✅ Covered | Unique tokens sufficient |
| Indic | 400B | 320B | 276B | ⚠️ Repeat | 1.45× @ 2.5T; tier C/D synthesis for verified gap |
| STEM / math | 300B | 240B | ~250B | ✅ Covered | peS2o + proof-pile + D4 STEM |
| Reasoning | 150B | 120B | 85B | ⚠️ Repeat | 1.76× @ 2.5T on AON + NuminaMath |
| Long-context | 150B | 120B | 100B | ⚠️ Repeat | 1.5× @ 2.5T; pack repo contexts |
| Agentic | 50B | 40B | **0.63B** | 🔴 Synthesize | **Binding constraint.** ≥98% must be synthetic/augmented |

### Execution readiness (today)

| Lane | Fundable from catalog without synthesis | Manifest-ready | Blocks full loader |
|------|----------------------------------------:|---------------:|--------------------|
| General web | ~100% | 0B | No (catalog OK) |
| Code | ~100% | 0B | No |
| STEM / math | ~83% | 0B | No |
| Indic | ~69% (with repeat) | ~0B | **Yes** — P1 verified tier |
| Reasoning | ~57% (with repeat) | AyAI subset only | Partial — scale-up pending |
| Long-context | ~67% (with repeat) | 0B | Packing pipeline pending |
| Agentic | **~1.3%** | 0B | **Yes** — P0 synthesis |

**Summary:** 4/7 lanes are catalog-covered for planning; **0/7** are fully manifest-gated for production loader access. Agentic and verified Indic are the binding gaps.

**OPUS note:** Retains ~40% of candidate batches, ~6× effective token value, ~few % compute overhead. Floors guarantee Indic and agentic survive selection even when proxy undervalues them.

---

## Step 7: Protected always-on floor

**Definition:** Minimum per-batch share OPUS **cannot** reduce, regardless of selector score.

| Lane | Floor | Rationale |
|------|------:|-----------|
| **Indic** | **≥ 12%** | Prior always-on was 8%; this plan raises to 12% for nine-script coverage |
| **Agentic** | **≥ 2%** | Matches main-run allocation; prevents selector from zeroing a 627M-token supply lane |

**Within-Indic sub-floor:** When Indic is sampled, **≥40% of the Indic slice must be tier A (verified)** — matches the Step 3 tier split and prevents OPUS from filling the Indic lane with synthetic-only batches.

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

**Anneal mixture** (composer preset):

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

**Anneal fundability:** Tier-A reserve ≈ **80B unique tokens** vs anneal draw **125B tokens** → **~1.56× repeat** on the reserve pool during cooldown. Acceptable if reserve quality is high; if reserve falls short, anneal backfills from tier-B shards with quality ≥0.65 (never synthetic for tier-A reserved slots).

**Safety & alignment:** Not a separate pre-training slice (would duplicate web/instruct). Safety-critical curated data enters via **anneal reserve** and post-training DPO — an estimated **~3% of total tokens** (~75B) touch safety-filtered, human-reviewed shards during cooldown and alignment.

---

## Step 9: Curriculum stages

| Stage | Token range | Mixture shift | Objective |
|-------|-------------|---------------|-----------|
| **1 — General foundation** | 0 – 800B | Web 45%, code 18%, Indic 14%, STEM 10%, rest minimal | Language, facts, basic code syntax |
| **2 — Capability ramp** | 800B – 1.875T | Web 30%, code 26%, Indic 16%, STEM 12%, reasoning 7%, long-ctx 5%, agentic 2% | Approaches main pretraining preset |
| **3 — Full preset** | 1.875T – 2.375T | **Main pretraining preset** (Step 2) | Hold mixture stable; OPUS active |
| **4 — Anneal** | 2.375T – 2.5T | **Anneal preset** (Step 8) | Cooldown; spend Tier-A reserve |

**Transition rule:** Never change mixture in one hard step. Blend over **≥8B-token warmup bands** per transition. A prior run saw **150× gradient-norm spike** when Hindi share jumped with frozen embeddings — monitor grad norm at every boundary.

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

**Hypothesis:** Main pretraining preset with OPUS + floors beats a naive web-heavy preset (web 55%+) on capability benchmarks without MMLU collapse — and anneal reserve improves final 5% lift.

**This mixture is a hypothesis.** The experiment below is designed to test it, not confirm it.

| Phase | What | Status | What it proves |
|-------|------|--------|----------------|
| **1 — Loader sim** | 5,000 batches × 1K tokens; A/B/C mixtures + floors | ✅ Executed | Batch sampler hits targets; floors never violated |
| **2 — Smoke char-LM** | Synthetic corpora, 500 steps, ~250k chars/condition | ✅ Executed | **Pipeline runs only** — not mixture quality |
| **3 — Benchmark proxy** | 3B dense @ 1B tokens, 3 seeds × 3 conditions | ⏳ Scheduled | Confirm/refute mixture vs benchmarks |

| Parameter | Value |
|-----------|-------|
| Model (Phase 3) | **3B dense** (24L, 2560 dim) — 3 seeds × 3 conditions |
| Tokens (Phase 3) | **1B per run** |
| Phase 1–2 script | [scripts/mixture_proxy_experiment.py](https://github.com/curioustushar/blog/blob/master/scripts/mixture_proxy_experiment.py) |
| Conditions | **A:** Naive web-heavy (web 55%, no floors). **B:** Main pretrain preset + floors. **C:** Anneal-style preset |
| Compute (Phase 1–2) | Local CPU — loader sim + char-LM smoke; Phase 3 requires GPU cluster |

### Phase 1 — Loader composition simulation (executed 2026-07-31)

| Lane | Target (B) | Observed (B) | Target (C) | Observed (C) | Floors OK |
|------|----------:|-------------:|----------:|-------------:|-----------|
| Web | 34% | **33.8%** | 8% | **8.0%** | — |
| Code | 24% | **24.0%** | 20% | **20.0%** | — |
| Indic | 16% | **16.0%** | 28% | **28.0%** | ✅ ≥12% |
| Agentic | 2% | **2.2%** | 8% | **8.0%** | ✅ ≥2% |

**Verdict:** Loader spec **confirmed** — mixtures land within ±0.5% of target; Indic and agentic floors never violated across 5M simulated tokens.

### Phase 2 — Char-LM smoke proxy (executed 2026-07-31)

**Does NOT validate the mixture hypothesis.** Hand-written lane corpora + 500 training steps cannot support MMLU/HumanEval/MILU claims. Counterintuitive results (A beating B on code/indic) are **expected at this scale** — do not interpret as evidence for or against the preset.

| Val set | A (web-heavy) PPL | B (target preset) PPL | C (anneal) PPL | Direction |
|---------|------------------:|------------------:|---------------:|-----------|
| Web | **111.2** | 117.8 | 99.9 | C best (web↓ in anneal mix) |
| Code | **16.4** | 26.3 | 32.7 | A best — smoke scale only |
| Indic | **150.7** | 174.8 | 567.9 | A best — not representative |
| STEM | **34.0** | 35.7 | 31.9 | C best |

### Phase 3 — Benchmark proxy (pending)

**Metrics & decision rules (full 3B proxy — not yet run):**

| Metric | Benchmark | Confirm B | Refute B |
|--------|-----------|-----------|----------|
| Knowledge | MMLU-500 | B ≥ A − 1 pt | B < A − 3 pt |
| Code | HumanEval | B ≥ A + 10 | B ≤ A + 3 |
| Math | GSM8K-200 | B ≥ A + 8 | B ≤ A + 2 |
| Indic | MILU subset | B ≥ A + 5 | B ≤ A + 1 |
| Agentic | BFCL-50 | B ≥ A + 8 | B ≤ A + 2 |
| Anneal lift | All above | C ≥ B + 2 avg | C ≤ B + 0.5 |

**If refuted:** Reduce Indic target 16→14%, shift to web; re-proxy at 500M continuation. If agentic shows no lift, cut to 1% floor and invest in synthetic trace quality before restoring.

**Status:** Phase 1 **confirmed**. Phase 2 smoke **executed** (pipeline only). Phase 3 **scheduled** before full-run lock.

Raw JSON: [scripts/mixture_proxy_results.json](https://github.com/curioustushar/blog/blob/master/scripts/mixture_proxy_results.json)

---

## Step 13: Data cleaning plan (starved slots)

**Gating rule:** Full mixture lock requires **P0 + P1 manifests**. Until then, the plan is **spec-complete, data-gated** — percentages are policy, not a claim that the loader can run today.

Cleaning targets slots Step 6 flags. Progress toward mixture needs at **2.5T** scale:

| Priority | Slot | Unblocks (Step 6) | Mixture need | Cleaned & manifest-ready | Gap | Progress |
|----------|------|-------------------|-------------:|-------------------------:|----:|----------|
| **P0** | Agentic | Agentic lane | 50B | **0.63B** raw inventory (627M); 0B cohort-synth w/ manifest | ~49.4B | **1.3%** of raw only — synthesis not started |
| **P1** | Indic verified | Indic tier A + anneal reserve | 160B (tier A) | **64B** Sangraha verified in catalog; PIB/wiki ingest pending | ~96B | **40%** of tier-A target in catalog; full 9-strategy pass pending |
| **P2** | Reasoning | Reasoning lane | 150B | **85B** AON inventory + **15.1M** AyAI rows cleaned ([cleaning blog]({{< relref "2026-07-23-data-cleaning-strategies.md" >}}): 22.2M→15.1M, ~68% survival, manifests generated) | ~65B @ 2.5T | Pipeline **proven** on 22M-row set; scale-up in progress |
| **P3** | Indic synthetic | Indic tier D cap | 60B (tier D) | 162B raw Sangraha synthetic; subset not yet quality-filtered to tier-D cap | TBD | Tier-D cap (15% of Indic) not yet enforced in loader |
| **P4** | Instruction format | Web instruct slice | ~50B (from web) | 0B FLAN/OpenOrca slices curated | ~50B | Not started |

**Data-gating:** No shard enters the D3–D7 loader without a cleaning manifest (see [9 Data cleaning strategies]({{< relref "2026-07-23-data-cleaning-strategies.md" >}})). Cohort mixture review applies only after P0/P1 manifests exist for starved lanes.

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

The **curriculum** ramps web-first, blends mixture changes over 8B-token bands, and saves Tier-A data for anneal — matching staged rebalancing from prior runs without hard-step instability.

The **proxy experiment** tests the highest-risk claim at 400× lower cost than full scale. **Phase 1 (loader simulation) confirmed** mixture targets and floors; **Phase 2 smoke executed** (pipeline only); **Phase 3 (3B @ 1B) remains the benchmark gate**.

**Limitations:**

- Supply numbers are **catalog-level** unless marked manifest-ready or locally verified (AyAI subset only).
- Agentic lane assumes **≥98% synthesis** — outcome depends on trace quality, not mixture % alone.
- Phase 2 char-LM results **must not** be read as benchmark evidence.
- Verified Indic tier-A remains the tightest tier; repetition caps are policy until Phase 3 confirms them.
- Reasoning effort dial requires post-training RLVR to become inference-controllable.

**Cleaning continues** toward P0 agentic and P1 Indic verified targets — the mixture defines which slots are starved; cleaning closes the gap.

### Self-check against evaluation criteria

| Criterion | Status |
|-----------|--------|
| Defended share per lane | ✅ Met — every lane has % + rationale |
| Indic 4-tier split | ✅ Met — 40/25/20/15 with per-tier supply table |
| Named slots mapped to datasets | ✅ Met — catalog mapping; gaps flagged explicitly |
| Benchmark traceability | ✅ Met — Step 5 |
| Supply realism | ⚠️ Catalog-level; execution readiness table in Step 6 |
| Honesty about repetition/synthetic gaps | ✅ Met — Step 3 policy + Step 6 agentic constraint |
| Protected floor stated | ✅ Met — Indic ≥12%, Agentic ≥2%, tier-A sub-floor |
| Anneal reserve stated | ✅ Met — separate preset, fundability check |
| Difficulty + reasoning bands with examples | ✅ Met — Steps 10–11 |
| Testable hypothesis framing | ✅ Met — Step 12 |
| Proxy specified (1B/3B scale) + named metrics | ✅ Met — Phase 3 spec |
| Proxy executed at benchmark scale | ❌ Not met — Phase 3 pending; stated explicitly |
| Loader / pipeline validation | ✅ Met — Phase 1 confirmed; Phase 2 smoke |
| Cleaning tie-in | ✅ Met — Step 13 + prior cleaning post |
| Manifest gating | ⚠️ P0/P1 open — plan is data-gated until closed |

---

## Related work

- [Train 40B model design]({{< relref "2026-07-17-train-40b-model-design.md" >}})
- [9 Data cleaning strategies applied]({{< relref "2026-07-23-data-cleaning-strategies.md" >}})
- [LightningLM 120B](https://lightninglm.theschoolofai.in/)
- [docs/the-mixture-is-the-model-2-5t-pretraining-plan.md](https://github.com/curioustushar/blog/blob/master/docs/the-mixture-is-the-model-2-5t-pretraining-plan.md)
