---
title: "Attention, Evolved"
slug: "attention-evolved"
date: 2026-08-21T09:00:00-05:00
categories: ["machine-learning"]
tags:
  ["attention", "transformers", "RoPE", "MQA", "GQA", "timeline", "interactive"]
author: Tushar Gupta
interactive: true
description: "A chronological timeline of 18 attention mechanisms from scaled dot-product attention (2017) to DroPE (2025) — each framed as a trade-off against compute, memory, context length, or bandwidth."
---

<div class="post-summary">

Vanilla attention was not wrong. It was **expensive**. Every technique that followed is somebody looking at that bill and trying to pay less of it.

This post accompanies an interactive app — [**Attention, Evolved**](../../attention-evolved/) — that puts **18 mechanisms in verified chronological order** (not grouped by family, not in teaching order) and explains each one as: what problem existed → what changed → what it buys → what it costs → when you would actually pick it.

<div class="plan-status" role="status" aria-label="Project status">
  <span class="status-badge status-ok">18 mechanisms covered</span>
  <span class="status-badge status-ok">Dates verified against primary sources</span>
  <span class="status-badge status-ok">Interactive timeline + compare mode</span>
</div>

</div>

---

## 1: Links

|                 | URL                                                                                                                                                                  |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Live app**    | [Attention, Evolved](../../attention-evolved/) |
| **GitHub repo** | [github.com/curioustushar/curioustushar.github.io/tree/main/attention-evolved](https://github.com/curioustushar/curioustushar.github.io/tree/main/attention-evolved) |

The README includes a full **date-source table** — every displayed date traced to arXiv, conference proceedings, or technical reports.

---

## The app

<figure class="figure-card attention-evolved-embed">
  <iframe
    src="../../attention-evolved/"
    title="Attention, Evolved — interactive timeline"
    loading="lazy"
    referrerpolicy="no-referrer-when-downgrade"
  ></iframe>
  <figcaption><strong>Embedded preview.</strong> Scroll inside the frame, or <a href="../../attention-evolved/" target="_blank" rel="noopener">open the full app</a> for timeline navigation, compare mode, and the visual lab.</figcaption>
</figure>

Three views:

1. **Timeline** — chronological cards with problem / idea / pros / cons / use cases / math
2. **Visual Lab** — attention matrix explorer (full, causal, sliding, sparse, sinks) and MHA/MQA/GQA head diagrams
3. **What Changed?** — synthesis of how bottlenecks shifted over time, including pre-2017 context and the four-problem framework

---

## Before 2017: why the timeline starts at the Transformer

The interactive timeline begins at **2017** because Session 2's baseline is scaled dot-product attention with softmax — nothing in the assignment list makes sense without that foundation. But attention itself is older.

### 2014 — Bahdanau: "look where you need"

Encoder-decoder RNNs compressed an entire sentence into one fixed vector. For long inputs, information was lost.

[Bahdanau, Cho & Bengio (2014)](https://arxiv.org/abs/1409.0473) let the decoder compute alignment weights over **all** encoder hidden states instead of one summary vector:

\[
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}, \quad c_t = \sum_i \alpha_{t,i} h_i
\]

**Problem solved:** selective retrieval — the model learns *where* to look, not just *what* to remember.

### 2015 — Luong: dot-product scoring

[Luong et al. (2015)](https://arxiv.org/abs/1508.04025) simplified alignment to direct vector similarity: score(q, k) = q·k. This is the direct precursor to the Transformer's Q, K, V abstraction and the equation Attention(Q, K, V) = softmax(QKᵀ / √d) V.

**Problem solved:** faster, cleaner scoring — the same idea the Transformer would scale to self-attention.

### 2017 — Vaswani: self-attention + multi-head

The Transformer removed the RNN entirely. Q, K, and V all come from the **same sequence** — every token can attend to every other token in one layer. Multiple heads let different subspaces specialize (syntax, entities, long-range links).

**Problem solved:** parallel all-to-all interaction. **Problem created:** O(n²) compute and memory.

The app bundles scaled dot-product attention and multi-head attention into the first timeline card because that is where Session 2 starts. The chronological entries after 2017 are all responses to costs this foundation introduced.

---

## Four problems, one evolution

Nearly every post-2017 mechanism attacks one of four bottlenecks:

| # | Problem | Examples on the timeline |
|---|---------|--------------------------|
| 1 | **Selective retrieval** | (solved pre-2017; assumed by 2017 baseline) |
| 2 | **All-to-all interaction efficiently** | Sparse Transformer, Longformer, linear attention |
| 3 | **Very long sequences** | RoPE, ALiBi, YaRN, NTK, DroPE, attention sinks |
| 4 | **Inference economics** | MQA, GQA, MLA, FlashAttention (systems, not on timeline) |

---

## One equation, four levers

Standard attention:

\[
O = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
\]

Almost everything after 2017 changes one of four things:

| Lever | What changes | Timeline examples |
|-------|--------------|-------------------|
| **A** | What interacts — QKᵀ | Sparse Transformer, Longformer, DroPE |
| **B** | How many K/V exist | MQA, GQA, MLA |
| **C** | How position enters Q, K | RoPE, ALiBi, YaRN, NTK-aware scaling |
| **D** | The mechanism itself | Linear attention, delta rule, DeltaNet, Gated DeltaNet |

---

## 2: What the timeline actually shows

A list of mechanisms tells you **what exists**. A timeline tells you **what the field was worried about, when, and in what order it changed its mind**.

### The pattern I could not see from a list

1. **2017 — Exactness first.** Standard attention + sinusoidal positions. The problem was RNN sequential bottlenecks, not cost. Sequences were short (512 tokens). O(n²) was acceptable.

2. **2018–2019 — Compute, then memory.** BERT's learned positions traded extrapolation for expressiveness. Sparse Transformer asked whether full attention was necessary. MQA (2019) attacked KV cache size — but nobody cared yet because inference at scale wasn't the crisis.

3. **2020 — Length becomes the headline.** Longformer made long documents practical with sliding windows. Linear attention showed attention could be O(n) — at a quality cost. The field split: some wanted longer context, others wanted cheaper attention.

4. **2021 — Position encoding revolution.** RoPE and ALiBi arrived within a day of each other. Same problem (positional extrapolation), opposite philosophies. RoPE won adoption; ALiBi's extrapolation story was cleaner but less expressive.

5. **2022 — Systems, not math.** FlashAttention kept exact softmax attention but tiled GPU computation to cut memory traffic. O(n²) became affordable enough that the next fight was KV cache bandwidth, not raw FLOPs.

6. **2023 — The inference crisis.** ChatGPT made inference cost matter. GQA was the MHA/MQA compromise. NTK-aware scaling and YaRN came from the open-source community. Attention sinks revealed that models learn unexpected behaviors — and how to work around them.

7. **2024–2025 — Compression and recurrence return.** MLA compresses what each KV head stores, not just how many heads exist. DeltaNet and Gated DeltaNet made recurrent-style updates parallelizable. DroPE asks whether you need precise positions everywhere. The progression looks like: MHA → MQA → GQA → latent/compressed KV.

8. **2024–2026 — Do we need attention everywhere?** State-space models (Mamba and hybrids) are alternatives, not timeline entries — but they explain why recurrence is back. Modern systems mix branches rather than picking one winner.

**The bottleneck keeps moving:** compute → memory → context length → positional extrapolation → memory bandwidth → compressed representations. Each "solution" did not eliminate the problem — it moved it somewhere else.

### The generations, in one glance

| Era | Main idea | Problem addressed |
|-----|-----------|-------------------|
| 2014 | Bahdanau alignment | RNN information bottleneck |
| 2015 | Dot-product scoring | Simpler, faster alignment |
| 2017 | Self-attention + multi-head | Direct token-to-token communication |
| 2019–20 | Sparse / local / linear | Quadratic scaling |
| 2019 | MQA | KV-cache size |
| 2022 | FlashAttention | GPU memory/IO (exact math, smarter execution) |
| 2023 | GQA | MQA quality vs. efficiency trade-off |
| 2023 | YaRN / NTK | RoPE extrapolation without full retrain |
| 2024+ | MLA, DeltaNet | Further KV reduction; parallelizable recurrence |

That is why the field keeps changing its mind. And that is what a timeline makes visible — not a list grouped by family.

---

## Mechanisms covered

All required mechanisms from the assignment, plus the chronological spine:

| Date       | Mechanism                         | Category         |
| ---------- | --------------------------------- | ---------------- |
| 2017-06-12 | Scaled Dot-Product Attention      | Core             |
| 2017-06-12 | Sinusoidal Positional Encoding    | Position         |
| 2018-10-11 | Absolute Learned Positions (BERT) | Position         |
| 2019-04-23 | Sparse Transformer                | Sparse           |
| 2019-09-06 | Multi-Query Attention (MQA)       | KV cache         |
| 2020-04-10 | Sliding Window (Longformer)       | Sparse           |
| 2020-06-09 | Linear Attention                  | Linear           |
| 2021-02-18 | Delta Rule                        | Recurrent        |
| 2021-04-20 | RoPE                              | Position         |
| 2021-04-21 | ALiBi                             | Position         |
| 2023-05-22 | Grouped Query Attention (GQA)     | KV cache         |
| 2023-06-28 | NTK-aware Scaled RoPE             | Position scaling |
| 2023-08-31 | YaRN                              | Position scaling |
| 2023-09-29 | Attention Sinks (StreamingLLM)    | Core             |
| 2024-05-06 | Multi-head Latent Attention (MLA) | Compression      |
| 2024-10-02 | DeltaNet                          | Recurrent        |
| 2024-10-02 | Gated DeltaNet                    | Recurrent        |
| 2025-12-13 | DroPE                             | Position scaling |

Every entry has **both** advantages and disadvantages. If a mechanism only had pros, the explanation was incomplete.

---

## Bonus: mechanisms not on the timeline

These belong in the story but are intentionally excluded from interactive cards — different category, or outside assignment scope.

**Flash Attention** (Dao et al., [arXiv:2205.14135](https://arxiv.org/abs/2205.14135), **2022-05-26**) — IO-aware tiling that keeps **exact** softmax attention but cuts GPU memory traffic. A systems breakthrough, not a new mechanism. It made O(n²) affordable enough that the 2023 bottleneck shifted to KV cache bandwidth rather than raw FLOPs.

**BigBird** (Zaheer et al., [arXiv:2007.14062](https://arxiv.org/abs/2007.14062), **2020-07-28**) — sparse attention with random global and block patterns, alongside Longformer in the long-context family. Covered conceptually via Sparse Transformer and Longformer; not a separate card to avoid sparse-attention duplication.

**Mamba / state-space hybrids (2023–2026)** — alternatives to full attention, not modifications of it. Explain why DeltaNet-style recurrence is resurgent, but belong in an epilogue rather than the chronological spine.

---

## How it was built

- **Research first:** every date verified against primary sources before any UI code
- **Data-driven:** `src/data/attentionMechanisms.ts` — the UI reads from structured data, not hard-coded components
- **Stack:** Next.js, TypeScript, Tailwind, KaTeX — static export into the Hugo site at `/blog/attention-evolved/`

To rebuild locally:

```bash
cd attention-evolved
npm install
npm run dev          # http://localhost:3000/blog/attention-evolved
./deploy.sh          # copies build to static/attention-evolved/
```

---

## What I would send a friend

> "Before 2017, attention solved *where to look*. In 2017, it became *every token looking at every token*. After that, every paper is somebody trying to pay a smaller bill — compute, memory, context, bandwidth. Start with standard attention, scroll the timeline in date order, and watch the field change its mind. Every card tells you honestly what you gain and what you give up."

<p class="bpe-widget-cta"><strong>Full app:</strong> <a href="../../attention-evolved/" target="_blank" rel="noopener">open Attention, Evolved</a></p>
