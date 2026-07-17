---
title: "Train 40B model design"
slug: "train-40b-model-design"
date: 2026-07-17T08:54:00-05:00
categories: ["machine-learning", "llm"]
tags: ["training", "india", "multilingual", "coding", "agentic"]
author: Tushar Gupta
description: "Design for a 40B sparse MoE model targeting Gemma-class quality with India-first perspective, strong coding, agentic tool use, and fair Indic tokenization — data, cleaning, alignment, and evaluation."
---

<div class="post-summary">

Training a **40B-parameter model** that treats India as the center rather than an afterthought means rethinking tokenization, corpus composition, and evaluation from the first commit. This post lays out a concrete design: **~12B active parameters** (sparse MoE), **128K vocabulary** with Indic fertility ≤0.85× English, **2.5T pre-training tokens** with per-batch guarantees for Indic (12%), code (20%), and agentic traces (8%), plus post-training and held-out evaluation that prove each capability was learned by construction — not by luck.

</div>

---

## Executive summary

Most multilingual models bolt on non-English support after the fact. English-heavy web corpora dominate merge statistics; Indic scripts pay a 2–3× tokenization penalty; coding and tool-use formats appear as rare tail events. The result is a model that *can* answer in Hindi but *prefers* not to, writes mediocre code, and hallucinates API calls instead of using tools.

This design inverts that order. **India-first perspective** is embedded from the tokenizer through alignment: native-script numerals, Indian English conventions, and civics/history framed from an Indian lens — not translated US-centric explanations.

| Dimension | Target |
|-----------|--------|
| Architecture | 40B total, ~12B active (MoE, top-k routing) |
| English quality | Gemma 4 class — MMLU 75+, GSM8K 70+ |
| Coding | HumanEval 65+, MBPP 70+ |
| Agentic | SWE-bench Lite 25+, WebArena 40+ |
| Indic | Per-script held-out loss <2.5; IndicGLUE 65+ |
| Tokenizer | 128K vocab; Indic fertility ≤0.85 vs English |

**Pre-training:** 2.5T tokens across seven pools, with a three-tier loader: scored web (45%), always-on guaranteed capabilities (55%), and a held-out golden proxy for scoring reference only.

**Post-training:** 100K SFT examples (30% Indic, 25% code, 20% agentic) + 50K DPO preference pairs.

**Evaluation:** Per-domain held-out loss every 10B tokens; full benchmark suite every 250B tokens; custom 500-question India-context eval.

The numbers below are design targets, not achieved results — the point is a reproducible recipe another team could execute.

---

## Tokenizer architecture

### Why tokenization is the first bottleneck

Neural networks consume integer sequences, not strings. Every downstream cost — FLOPs, context window, batch size — scales with tokens per character. Typical byte-level or English-centric BPE tokenizers compress Latin text at ~3.5 characters per token but leave Devanagari, Tamil, and Telugu at 1.2–1.5× worse fertility than English. A Hindi paragraph that should be 400 tokens becomes 600; context shrinks; quality drops before the first forward pass.

Example: *भारत की जनसंख्या* (India's population) might consume 8 tokens in a GPT-class tokenizer versus 4–5 in one trained for Brahmic scripts.

### Vocabulary size: 128K

We work backward from fertility targets:

| Domain | Chars/token target | Fertility vs English |
|--------|-------------------|----------------------|
| English (baseline) | 3.5 | 1.00 |
| Hindi / Bengali / Marathi | 4.0–4.2 | 0.83–0.87 |
| Tamil | 4.3–4.5 | 0.78–0.81 |
| Telugu | 4.1–4.3 | 0.81–0.85 |
| Python code | 3.0–3.2 | 1.09–1.17 |
| Math / LaTeX | 2.8–3.0 | 1.17–1.25 |
| Agentic JSON/XML | 3.2–3.4 | 1.03–1.09 |

**64K** leaves Indic fertility above 1.0 — unacceptable. **256K** improves compression <2% at 2× embedding cost. **128K** hits all targets with reasonable overhead. LightningLM's [BrahmicTokenizer-131K](https://lightninglm.theschoolofai.in/index.html) validates this class; we adopt the same order of magnitude.

### Training recipe

- **Algorithm:** Byte-level BPE with Metaspace pre-tokenizer/decoder (HuggingFace `tokenizers`)
- **Normalizer:** NFKC
- **Faithful roundtrip gate:** `decode(encode(text))` must preserve visible non-whitespace characters — required for code, URLs, and Markdown corpora
- **Training sample:** 50B tokens weighted toward objectives:
  - English web 30%, Indic 24%, code 30%, math/science 10%, agentic 6%
  - Oversampling: Indic ×3, code ×2, math ×2
- **Expected spread** (max fertility − min fertility): 0.12–0.15 vs ~0.25–0.35 in typical multilingual tokenizers

```text
fertility(language) = token_count / faithful_units

faithful_unit = one contiguous Unicode letter/mark/number run
             OR one visible non-whitespace punctuation/symbol
```

**Why BPE over SentencePiece/Unigram:** BPE is transparent for debugging fertility imbalance; byte-level eliminates UNK on mixed-script input; Metaspace preserves whitespace and formatting for code.

---

## Pre-training data strategy

### Three-tier architecture

Inspired by [LightningLM's guaranteed-composition pipeline](https://lightninglm.theschoolofai.in/index.html):

| Tier | Share | Role |
|------|-------|------|
| **Scored & selected** | ~45% | D1/D2 web — OPUS-style dynamic selector ranks batches by alignment to capability targets |
| **Always-ON guaranteed** | ~55% | Indic, code, STEM, agentic, reasoning — fixed per-batch floor, bypasses scorer |
| **Golden proxy** | held out | MMLU/GSM8K/HumanEval validation splits — scorer reference only, never trained |

**Key principle:** A scorer aligned to English benchmarks systematically undervalues Indic text, assembly code, and specialized reasoning traces. Guarantee them per-batch; do not hope reweighting rescues them.

### Corpus composition (2.5T tokens)

| Pool | Role | Tokens | % | Sources |
|------|------|-------:|--:|---------|
| D1 Clean Web | Early curriculum backbone | 625B | 25% | FineWeb, C4, RedPajama-v2 |
| D2 Diverse Web | Long-tail knowledge | 500B | 20% | CommonCrawl 2023–2025, deduplicated |
| D3 Indic | Guaranteed multilingual | 300B | 12% | AI4Bharat, IndicCorpV2, Sangraha, Wikipedia, news |
| D4 Code | Guaranteed coding | 500B | 20% | The Stack v2 (≥10 GitHub stars) |
| D5 Scientific | STEM | 200B | 8% | arXiv, PubMed, Proof-pile-2, OpenWebMath |
| D6 Agentic | Tool use, function calling | 200B | 8% | WebArena, SWE-bench, Gorilla, synthetic traces |
| D7 Reasoning | CoT, math | 175B | 7% | MetaMath, FLAN-CoT, OpenOrca reasoning subset |

**Per-batch invariants** (enforced by loader, verified with source tags):

- ≥12% Indic
- ≥20% code
- ≥8% agentic

These hold at every stage — not as corpus averages that a selector can violate on any single step.

### Why these proportions

- **Indic 12%:** Higher than LightningLM's 6% because we cover nine scripts, not four. High enough to prevent forgetting between exposures; low enough not to collapse English/code quality.
- **Code 20%:** Matches Codex/StarCoder ratios; above Gemma (~15%) because coding is an explicit objective.
- **Agentic 8%:** Most base models see <1% structured tool-use format. Mix real traces with GPT-4/Claude synthetic data, validated ≥4/5 quality.
- **STEM 8%:** Dense signal; oversampled vs natural web frequency (~2–3%). Proof-heavy papers (theorem density ≥15%) get 1.5× weight.

### Difficulty curriculum

| Stage | Token range | D1 web | Code | Indic | Agentic | STEM |
|-------|-------------|-------:|-----:|------:|--------:|-----:|
| 1 Foundation | 0–800B | 60% | 15% | 12% | 8% | 5% |
| 2 Intermediate | 800B–1.8T | 30% | 25% | 12% | 10% | 10% |
| 3 Advanced | 1.8T–2.5T | 15% | 30% | 12% | 12% | 15% |

Training loss will **rise** at stage transitions as harder data enters. **Held-out loss must keep falling** — that is the signal the curriculum is working, not failing.

---

## Post-training and alignment

### Supervised fine-tuning (100K pairs)

| Bucket | % | Content |
|--------|--:|---------|
| Indic instruction | 30% | Native Hindi/Tamil/Telugu + translated ShareGPT |
| Code | 25% | HumanEval-style + real GitHub issues from Indian OSS |
| Agentic | 20% | Function calling, API interactions, web navigation |
| India-context QA | 15% | History, civics, geography, current affairs |
| General reasoning | 10% | Math, science, logic |

**India-first framing examples:**

- *Explain the Green Revolution* → Punjab/Haryana, Indian agricultural policy — not a generic global survey
- *What is federalism?* → Indian constitutional structure (Centre–States), not US states' rights
- *Convert currency* → INR as default base, not USD

### DPO alignment (50K preference pairs)

- 40% GPT-4 quality judgments
- 30% human annotations (Indian annotators for cultural accuracy)
- 30% rule-based: executable code > broken; native script > transliteration; API call > hallucinated fact

**Why DPO over RLHF:** No separate reward model; more stable; ~40% cheaper GPU-hours; sufficient for task-completion objectives (Zephyr/Tulu results).

Alignment objectives: helpfulness, cultural appropriateness, safety adapted to Indian legal context, preference for structured tool use over free-text guessing.

---

## Data cleaning pipeline

### Universal filters (all pools)

- MinHash LSH deduplication (Jaccard ≥0.8)
- FastText language ID — keep target languages only
- Length: 50 < tokens < 100K
- Toxicity: Perspective API + Indian profanity list
- PII: emails, phones, Aadhaar (12-digit), PAN patterns

### Pool-specific cleaning

**Web (D1/D2):** KenLM perplexity filter (>10K → drop); n-gram repetition >30% → drop; Trafilatura main-content extraction; ad/boilerplate heuristics.

**Indic (D3):** ≥80% target script per document; Arabic→native numeral substitution adjacent to Indic script; language-specific perplexity, keep top 70% per script.

**Code (D4):** Drop GPL/AGPL; remove minified/compiled artifacts; ≥10 GitHub stars; preserve comments/docstrings; 2× upsample test files (`*test*.py`, `*_spec.rb`); add tab-indented kernel/CUDA sources so tab tokens are live.

**Scientific (D5):** Preserve LaTeX equations; drop figure/table OCR; upsample proof-heavy papers.

**Agentic (D6):** Valid JSON/XML only; successful executions only; max 100 examples per API endpoint.

### Dead-token audit

After tokenizer training, scan the full 131K vocabulary for zero-frequency tokens. Expected dead zones: native Indic numerals (if corpus uses Arabic digits), tab tokens, CRLF tokens. Fix in the **corpus**, not the model — targeted numeral substitution (~4.7M swaps across Indic shards), add Windows-line-ending codebases, add tab-indented C/CUDA. A vocabulary and corpus are one design.

---

## Evaluation framework

### Per-domain held-out loss (primary signal)

0.5% of each pool, stratified, fixed at training start. Evaluated every 10B tokens.

| Domain | Start (approx) | Target at 2.5T |
|--------|---------------:|---------------:|
| Code | 2.5 | <1.8 |
| Indic (per script) | 2.0–3.0 | <2.5 |
| STEM | 3.0 | <2.0 |
| Web | 4.5 | 3.8–4.0 |

LightningLM 9B reached code held-out 1.54; our 40B target is comparable or better. Web loss staying high is expected — it is the hardest, noisiest bucket.

### Benchmark suite (every 250B tokens)

| Capability | Benchmarks | Target |
|------------|------------|-------:|
| General knowledge | MMLU | 75+ |
| Reasoning | GSM8K, MATH | 70+, 35+ |
| Coding | HumanEval, MBPP, MultiPL-E | 65+, 70+, 55+ |
| Agentic | SWE-bench Lite, WebArena | 25+, 40+ |
| Indic | IndicGLUE, Flores-200 | 65+, 30 BLEU |
| India context | Custom 500-MCQ eval | 80+ |

Greedy decoding for code (pass@1); 5-shot for MMLU/GSM8K; zero-shot for agentic tasks.

### India-context custom eval

500 four-option MCQs: Indian history (30%), constitution & governance (25%), geography (15%), current affairs 2023–2026 (15%), culture (15%). Baseline against GPT-4, Claude 3.5, Gemma 2 27B. Expected edge: +15–20 points over Gemma on framing; +5–10 over frontier models that know facts but answer US-centrically.

### Monitoring cadence

- Held-out loss: every 10B tokens
- Per-script Indic loss: every 25B tokens
- Code pass@1 (200-problem HumanEval subset): every 100B tokens
- Full benchmark suite: every 250B tokens (~10 checkpoints)

---

## Appendix: compute and formulas

**Estimated compute budget**

| Stage | GPU-days (H100) | Cost (cloud) |
|-------|----------------:|-------------:|
| Pre-training 2.5T | ~12,000 | ~$3–4M |
| SFT | ~50 | ~$15K |
| DPO | ~30 | ~$10K |
| Evaluation | ~200 | ~$60K |

```text
spread = max(fertility) − min(fertility)
score  = 1000 / spread    # tokenizer quality metric; target >5000
```

---

## What I would try next

1. **Script-adaptive oversampling** — if Tamil fertility drifts above 0.90 mid-training, bump Indic shard weight for that script only.
2. **Guaranteed tier accounting** — log actual per-batch composition from the loader (LightningLM's Always-ON bug pushed effective share to 57% for 23K steps; measure, don't assume).
3. **Agentic curriculum** — start with JSON schema completion before full multi-step WebArena traces.
4. **India-context eval as training signal** — use the 500-MCQ set as a held-out proxy direction for OPUS scoring in late curriculum.

Tokenization, data guarantees, and per-domain held-out loss are where multilingual bias shows up — long before anyone runs MMLU. Design those three correctly and the benchmarks follow.
