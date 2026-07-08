---
title: "Multilingual tokenizer - BPE"
date: 2026-07-08T11:30:00-05:00
categories: ["machine-learning"]
tags: ["tokenization", "bpe", "nlp", "multilingual"]
author: Tushar Gupta
interactive: true
bpe: true
description: "Byte-level BPE over India's Wikipedia article in English, Hindi, Telugu, and Tamil — theory, ratios, score, and a verifiable tokenizer widget."
---

<div class="post-summary">

Before a language model reads text, something has to **cut it into pieces**. That step — tokenization — quietly controls vocabulary size, sequence length, and how fairly a model treats each language. Here I build a **10,000-token byte-level BPE** tokenizer on the [India](https://en.wikipedia.org/wiki/India) Wikipedia article in **English, Hindi, Telugu, and Tamil**, then measure how evenly it compresses each script.

</div>

---

## Why tokenization matters

Neural networks do not consume raw strings. They consume **integer sequences** indexed into a finite vocabulary. Every design choice in that mapping has downstream effects:

- **Sequence length** — more tokens per sentence means more compute and shallower context within a fixed window.
- **Out-of-vocabulary handling** — a tokenizer that never sees a character or byte can represent can still encode *any* Unicode string; one that only knows whole words cannot.
- **Cross-lingual fairness** — the same semantic content in English vs. an Indic script can tokenize to very different lengths if the vocabulary was trained mostly on Latin text.

Subword methods sit in a useful middle ground: frequent words stay whole, rare words decompose into reusable pieces.

## Byte Pair Encoding (BPE)

BPE starts from a small alphabet — here, the **256 UTF-8 bytes** — and iteratively **merges** the most frequent adjacent pair into a new symbol.

1. Pretokenize text into words (whitespace-delimited chunks).
2. Represent each word as a sequence of bytes.
3. Count adjacent byte pairs across the corpus.
4. Merge the most common pair; repeat until the merge budget is exhausted.

The result is a vocabulary of bytes plus learned merges. Encoding replays those merges in priority order: repeatedly combine the **lowest-rank** applicable pair until no more merges apply.

This is the same family of algorithm used in GPT-2 and many modern LLMs, adapted here for a **fixed multilingual budget** rather than English-only training.

### Why byte-level for four scripts?

English Wikipedia is mostly ASCII-friendly. Hindi (Devanagari), Telugu, and Tamil each use distinct Unicode blocks. A **byte-level** tokenizer does not need script-specific rules: any character is already a sequence of UTF-8 bytes. The cost is longer initial sequences before merges kick in — which is why **merge budget allocation** across languages becomes a design problem, not an implementation detail.

## The corpora: one topic, four languages

Using the **same article** (India) in four languages controls for topic. We are not comparing "news vs. poetry"; we are asking how the **same encyclopedic content** compresses under one shared vocabulary.

| Code | Language | Wikipedia title | Words (approx.) |
|------|----------|-----------------|-----------------|
| en | English | [India](https://en.wikipedia.org/wiki/India) | 10,121 |
| hi | Hindi | [भारत](https://hi.wikipedia.org/wiki/India) | 8,078 |
| te | Telugu | [భారతదేశం](https://te.wikipedia.org/wiki/India) | 2,511 |
| ta | Tamil | [இந்தியா](https://ta.wikipedia.org/wiki/இந்தியா) | 10,297 |

Telugu's article is shorter in this snapshot; Tamil and English are similarly long. Raw word count alone does not predict token count — script complexity and merge coverage matter more.

## Designing a 10,000-token shared vocabulary

The vocabulary constraint: **10,000 total symbols**, with roughly **5,000 allocated to English** and the remainder split across the other three languages. In practice that means:

| Language | Vocab allocation | Merge budget |
|----------|------------------|--------------|
| English | 5,000 | 3,200 |
| Hindi | 1,200 | 900 |
| Telugu | 1,000 | 700 |
| Tamil | 2,800 | 2,800 |

**English** receives the largest slice because Latin-heavy text benefits from a bigger merge table early in training.**Tamil** receives extra merge budget — Indic scripts with combining characters often need more byte-pair steps before common morphemes emerge. Hindi and Telugu sit between those extremes.

All languages share **one merge table**. A merge learned from Hindi text can help encode Tamil bytes if the underlying pair is common enough. That is the point of a multilingual vocab: shared structure, not four isolated dictionaries.

## The ratio metric and score

For each language *i*, define:

**X<sub>i</sub>** = tokens produced when encoding the full Wikipedia article ÷ vocab slots allocated to language *i*

Sort X1…X4 (English, Hindi, Telugu, Tamil). The **spread** is max − min. The **score** rewards balance:

**Score** = 1000 / (X<sub>max</sub> − X<sub>min</sub>)

A tokenizer that compresses all four languages equally yields a small spread and a high score. One that tokenizes English cheaply but explodes on Tamil yields a large spread and a low score — even if the total vocab is still 10,000.

In the playground below, **fertility** (encode tokens ÷ pretoken words) is a complementary per-input measure: it tells you how many subword pieces each whitespace-delimited word became for *your* text, not the full corpus.

<div id="bpe-summary" class="nn-demo bpe-summary-demo"></div>

## Tokenization playground

Type mixed-script text and watch BPE split it in real time. Hover a highlighted span to see its **vocab id** and literal value (spaces render as `<0x20>`, newlines as `<0x0A>`).

<div id="bpe-playground"></div>

## Full widget: verification & downloads

The extended widget adds corpus-level statistics, a **Run verification** button (re-encodes bundled Wikipedia text in-browser so you can audit the claimed counts), and downloads:

- `tokenizer.json` — merges, vocab, and precomputed stats
- `vocab.txt` — id → token mapping
- `merges.csv` — merge rank and language of origin

<p class="bpe-widget-cta"><strong>Full widget:</strong> <a id="bpe-full-widget-link" href="#">open full widget</a></p>

## What I would try next

A few directions that fall out of this exercise:

1. **Joint training** — interleave corpora during merge selection instead of training per-language tables and deduplicating.
2. **Unigram LM tokenization** (SentencePiece-style) — optimize a probabilistic objective rather than greedy pair counts.
3. **Normalize Indic text** — NFC/NFKC and consistent punctuation often shave fertility without growing vocab.
4. **Dynamic vocab budgets** — allocate merge slots proportional to corpus bytes, then re-score.

Tokenization looks like plumbing. It is actually the first place multilingual bias shows up — long before the model sees a loss function.
