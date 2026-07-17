---
title: "Multilingual tokenizer - BPE"
date: 2026-07-08T11:30:00-05:00
categories: ["machine-learning"]
tags: ["tokenization", "bpe", "nlp", "multilingual"]
author: Tushar Gupta
interactive: true
bpe: true
description: "HuggingFace BPE + Metaspace over faithful Wikipedia Markdown for India in English, Hindi, Telugu, and Tamil — fertility score and verifiable roundtrip widget."
---

<div class="post-summary">

Before a language model reads text, something has to **cut it into pieces**. That step — tokenization — quietly controls vocabulary size, sequence length, and how fairly a model treats each language. Here I train a **10,000-token shared BPE** tokenizer (HuggingFace `tokenizers`, Metaspace pre-tokenizer/decoder) on **faithful Wikipedia Markdown** for the [India](https://en.wikipedia.org/wiki/India) article in **English, Hindi, Telugu, and Tamil**, then score how evenly it compresses each script while preserving visible text under `decode(encode(text))`.

</div>

---

## Why tokenization matters

Neural networks do not consume raw strings. They consume **integer sequences** indexed into a finite vocabulary. Every design choice in that mapping has downstream effects:

- **Sequence length** — more tokens per sentence means more compute and shallower context within a fixed window.
- **Faithful representation** — if `decode(encode(text))` drops apostrophes, commas in numbers, or URL characters, the tokenizer is not representing the same input even if token counts look low.
- **Cross-lingual fairness** — the same encyclopedic content in English vs. an Indic script can tokenize to very different lengths if the vocabulary was trained mostly on Latin text.

Subword methods sit in a useful middle ground: frequent words stay whole, rare words decompose into reusable pieces.

## BPE with Metaspace (faithful roundtrip)

This tokenizer uses HuggingFace **BPE** with:

- **Normalizer:** NFKC
- **Pre-tokenizer / decoder:** Metaspace (`▁` marks word boundaries; spaces round-trip)
- **Vocab size:** 10,000 (one shared table for all four languages)

Metaspace preserves punctuation, brackets, URL characters, apostrophes, and number separators — required for **faithful Markdown** evaluation. A byte-level tokenizer that skips whitespace or lacks a `decode` method fails the roundtrip gate even if fertility looks good.

Training weights (corpus oversampling): English ×3, Hindi ×4, Telugu ×4, Tamil ×2.

## The corpora: faithful Wikipedia Markdown

Plain `explaintext` extracts clip links and tables. This assignment uses **wiki-faithful Markdown**: Wikipedia REST HTML converted to Markdown while keeping links, URLs, tables, references, and categories where the converter emits them.

| Code | Language | Wikipedia title |
|------|----------|-----------------|
| en | English | [India](https://en.wikipedia.org/wiki/India) |
| hi | Hindi | [भारत](https://hi.wikipedia.org/wiki/India) |
| te | Telugu | [భారతదేశం](https://te.wikipedia.org/wiki/India) |
| ta | Tamil | [இந்தியா](https://ta.wikipedia.org/wiki/இந்தியா) |

Rebuild:

```bash
python scripts/build_wiki_faithful_markdown.py
python scripts/train_bpe_tokenizer.py
```

## Fertility metric and score

A **faithful unit** is one contiguous Unicode letter/mark/number run, **or** one visible non-space punctuation/symbol character.

For each language *i*:

**X<sub>i</sub>** = token count when encoding the full faithful corpus ÷ faithful unit count

**Score** = 1000 / (X<sub>max</sub> − X<sub>min</sub>)

All four ratios must stay **≤ 1.2** under this denominator (see below). The tokenizer must also round-trip visible text — e.g. `India's population is 1,428,627,663.` decodes unchanged.

## Results

Measured on the faithful Markdown corpora (July 2026 snapshot):

| Language | Tokens | Faithful units | Fertility X<sub>i</sub> | ≤ 1.2? |
|----------|-------:|---------------:|------------------------:|:------:|
| English (X1) | 116,038 | 186,426 | **0.6224** | pass |
| Hindi (X2) | 55,156 | 88,359 | **0.6242** | pass |
| Telugu (X3) | 26,141 | 36,293 | **0.7203** | pass |
| Tamil (X4) | 129,376 | 185,869 | **0.6961** | pass |

**Spread** (max − min): 0.7203 − 0.6224 = **0.0978**

**Raw score:** 1000 / 0.0978 = **10,220.6**

**Hindi penalty:** exp(max(0, X<sub>hi</sub> / 1.2 − 1)) = exp(0) = **1.0** (no penalty when Hindi ≤ 1.2)

**Adjusted score:** 10,220.6 / 1.0 = **10,220.6**

Roundtrip check: `decode(encode("India's population is 1,428,627,663."))` → identical string.

## How the 1.2 threshold works

Each language gets a **fertility ratio** X<sub>i</sub> = tokens / faithful units. The assignment requires **every** language to satisfy:

```text
X_i ≤ 1.2
```

So 1.2 is not an average — it is a **per-language ceiling**. If any single language exceeds 1.2, that submission fails the threshold gate even if others look good.

**Examples:**

| Language | Calculation | Pass? |
|----------|-------------|:-----:|
| English | 116,038 / 186,426 = 0.6224 ≤ 1.2 | yes |
| Hindi | 55,156 / 88,359 = 0.6242 ≤ 1.2 | yes |
| Telugu | 26,141 / 36,293 = 0.7203 ≤ 1.2 | yes |
| Tamil | 129,376 / 185,869 = 0.6961 ≤ 1.2 | yes |

**Hindi penalty** (score adjustment only, not a pass/fail gate):

```text
hindi_penalty = exp(max(0, X_hi / 1.2 - 1))
adjusted_score = raw_score / hindi_penalty
```

When X<sub>hi</sub> = 0.6242, the exponent is 0, so the penalty factor is 1.0. If Hindi were 1.32, then 1.32/1.2 − 1 = 0.1 and penalty = e<sup>0.1</sup> ≈ 1.105, reducing the adjusted score.

<div id="bpe-summary" class="nn-demo bpe-summary-demo"></div>

## Tokenization playground

Type mixed-script text (including punctuation and numbers) and watch BPE split it in real time. The playground checks that `decode(encode(text))` preserves visible characters.

<div id="bpe-playground"></div>

## Full widget: verification & downloads

The extended widget adds corpus-level statistics, a **Run verification** button (re-encodes bundled faithful Markdown in-browser), roundtrip checks, and downloads:

<p class="bpe-download-cta"><strong>Direct download:</strong> <a id="bpe-tokenizer-download-link" href="#">tokenizer.json</a> (HuggingFace format with encode + decode)</p>

<p class="bpe-widget-cta"><strong>Full widget:</strong> <a id="bpe-full-widget-link" href="#">open full widget</a></p>

## What I would try next

1. **Corpus weight tuning** — adjust oversampling if one language dominates merge statistics.
2. **Unigram LM tokenization** (SentencePiece-style) — optimize a probabilistic objective rather than greedy pair counts.
3. **NFKC normalization experiments** — consistent punctuation often shaves fertility without breaking faithfulness.
4. **Live Wikipedia drift** — re-fetch corpora periodically; faithful unit counts change as articles edit.

Tokenization looks like plumbing. It is actually the first place multilingual bias shows up — long before the model sees a loss function.
