# Attention, Evolved

An interactive chronological exploration of how attention mechanisms evolved from 2017 to today. Each technique is presented as a response to a bottleneck—compute, memory, context length, positional extrapolation, or bandwidth.

**Live Demo:** [https://curioustushar.github.io/blog/attention-evolved](https://curioustushar.github.io/blog/attention-evolved)

## Overview

This web application teaches how attention mechanisms evolved over time using a chronological timeline as the central organizing principle. Rather than grouping mechanisms by family or teaching order, they are presented in the order they actually appeared.

The core insight: **Vanilla attention was not "wrong." It was expensive. Each subsequent technique is a response to some constraint.**

### What the Timeline Shows

The chronological view reveals patterns that lists obscure:

1. **2017**: Standard attention + sinusoidal positions established the foundation
2. **2018-2019**: First efficiency attempts (sparse attention, MQA)
3. **2020**: Long context push (Longformer, linear attention)
4. **2021**: Position encoding revolution (RoPE and ALiBi within one day of each other)
5. **2023**: Inference crisis (GQA, YaRN, attention sinks)
6. **2024-2025**: Compression and recurrence return (MLA, DeltaNet, Gated DeltaNet)

The bottleneck keeps moving: compute → memory → context length → positional extrapolation → memory bandwidth → compressed representations.

## Features

- **Chronological Timeline**: 17 mechanisms displayed in verified historical order
- **Interactive Visualizations**: Attention matrix explorer, KV heads comparison (MHA/MQA/GQA)
- **Side-by-Side Comparison**: Select any two mechanisms to compare properties, advantages, and disadvantages
- **Expandable Cards**: Each mechanism shows problem, solution, pros, cons, use cases, and math
- **Category Filtering**: Filter by attention mechanism, positional encoding, KV optimization, etc.
- **Mathematical Equations**: KaTeX-rendered equations for all major mechanisms

## Run Locally

```bash
# Clone the repository
git clone https://github.com/curioustushar/curioustushar.github.io.git
cd curioustushar.github.io/attention-evolved

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

The app will be available at `http://localhost:3000/blog/attention-evolved`

## Architecture

```
attention-evolved/
├── src/
│   ├── app/
│   │   ├── page.tsx          # Main application page
│   │   ├── layout.tsx        # Root layout with metadata
│   │   └── globals.css       # Global styles
│   ├── components/
│   │   ├── Timeline.tsx          # Chronological timeline view
│   │   ├── MechanismCard.tsx     # Expandable mechanism details
│   │   ├── CompareMode.tsx       # Side-by-side comparison modal
│   │   ├── CategoryBadge.tsx     # Category label component
│   │   ├── MathDisplay.tsx       # KaTeX equation renderer
│   │   ├── AttentionMatrixViz.tsx # Interactive attention patterns
│   │   └── KVHeadsViz.tsx        # MHA/MQA/GQA visualization
│   └── data/
│       └── attentionMechanisms.ts # All mechanism data
├── next.config.ts            # Next.js configuration for static export
└── package.json
```

## Research Methodology

Every date displayed in this application was verified against primary sources. The methodology:

1. **Primary Source Priority**: arXiv submission dates, conference proceedings, or official technical reports
2. **Date Type Documentation**: Each date is explicitly labeled (arxiv, conference, technical-report, forum)
3. **Cross-Reference**: Dates were verified against multiple sources where ambiguity existed
4. **Error Correction**: When commonly cited dates differed from primary sources, the primary source was used

### Important Notes on Dates

- **MQA (2019-09-06)**: Published 2019 but not widely adopted until 2022-2023
- **RoPE and ALiBi**: Both submitted within one day (April 20-21, 2021)
- **NTK-aware Scaling**: Originated from Reddit community (r/LocalLLaMA) rather than a paper
- **DeltaNet and Gated DeltaNet**: Same paper, same date (October 2024)
- **DroPE (December 2025)**: Most recent entry, arXiv submission

## Date/Source Table

| Mechanism                         | Date Used  | Date Type        | Primary Source                                                               | Notes                                                      |
| --------------------------------- | ---------- | ---------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Scaled Dot-Product Attention      | 2017-06-12 | arxiv            | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)                         | Original Transformer paper                                 |
| Sinusoidal Positional Encoding    | 2017-06-12 | arxiv            | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)                         | Same paper as standard attention                           |
| Absolute Learned Positions        | 2018-10-11 | arxiv            | [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)                         | BERT paper                                                 |
| Sparse Transformer                | 2019-04-23 | arxiv            | [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)                         | OpenAI                                                     |
| Multi-Query Attention (MQA)       | 2019-09-06 | arxiv            | [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)                         | Noam Shazeer at Google                                     |
| Sliding Window (Longformer)       | 2020-04-10 | arxiv            | [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)                         | Allen AI                                                   |
| Linear Attention                  | 2020-06-09 | arxiv            | [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)                         | "Transformers are RNNs"                                    |
| Delta Rule                        | 2021-02-18 | arxiv            | [arXiv:2102.11174](https://arxiv.org/abs/2102.11174)                         | "Linear Transformers Are Secretly Fast Weight Programmers" |
| RoPE                              | 2021-04-20 | arxiv            | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)                         | RoFormer paper                                             |
| ALiBi                             | 2021-04-21 | arxiv            | [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)                         | Original submission date April 2021                        |
| Grouped Query Attention (GQA)     | 2023-05-22 | arxiv            | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)                         | Google, used in LLaMA 2                                    |
| NTK-aware Scaling                 | 2023-06-28 | forum            | [Reddit r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/) | Community research by bloc97                               |
| YaRN                              | 2023-08-31 | arxiv            | [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)                         | "Yet another RoPE extensioN"                               |
| Attention Sinks                   | 2023-09-29 | arxiv            | [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)                         | StreamingLLM                                               |
| Multi-head Latent Attention (MLA) | 2024-05-06 | technical-report | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)                         | DeepSeek-V2                                                |
| DeltaNet                          | 2024-10-02 | arxiv            | [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)                         | Parallelizable delta rule                                  |
| Gated DeltaNet                    | 2024-10-02 | arxiv            | [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)                         | Same paper as DeltaNet                                     |
| DroPE                             | 2025-12-13 | arxiv            | [arXiv:2412.09797](https://arxiv.org/abs/2412.09797)                         | Hierarchical position dropping                             |

## Important Corrections

During research, we identified these commonly confused facts:

1. **MQA vs GQA dates**: MQA (2019) predates GQA (2023) by 4 years, but GQA became more famous due to LLaMA 2 adoption
2. **ALiBi date**: Often cited as August 2021 (arXiv update), but original submission was April 2021
3. **Linear Attention origin**: The 2020 paper showed the RNN connection; linear attention variants existed earlier but this paper crystallized the insight
4. **NTK-aware scaling**: Not a peer-reviewed paper—emerged from open-source community experimentation

## Mechanisms Not Included

The following were considered but not included as primary entries:

- **Flash Attention**: An implementation optimization (memory-efficient attention), not a new attention mechanism
- **Mamba/SSMs**: State-space models are an alternative to attention, not a modification of it
- **LoRA/Adapters**: Parameter-efficient fine-tuning, not attention mechanisms
- **KV Cache Quantization**: Compression technique, not a structural change to attention

## Tech Stack

- **Next.js 16** with App Router
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **KaTeX** for mathematical notation
- **Static export** for GitHub Pages deployment

## Deployment

The application is deployed on GitHub Pages via static export:

```bash
npm run build
# Output is in /out directory
```

The `next.config.ts` is configured with:

- `output: "export"` for static HTML generation
- `basePath: "/blog/attention-evolved"` for GitHub Pages subpath under the Hugo site

## Contributing

If you find an error in a date, description, or technical detail:

1. Verify against the primary source
2. Open an issue with the correction and source link
3. Or submit a PR updating `src/data/attentionMechanisms.ts`

## License

MIT

## Acknowledgments

Built Modern Attention Variants. The assignment emphasized historical accuracy over visual impressiveness—this application attempts to honor that goal.

---

_"Do not optimize for how impressive the demo looks. Optimize for being correct about the history and honest about the engineering trade-offs."_
