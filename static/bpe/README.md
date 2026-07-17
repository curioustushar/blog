# BPE Tokenizer Widget

Static widget served with the blog at `/blog/bpe/`.

## Local preview

```bash
cd static/bpe
python3 -m http.server 8080
# open http://localhost:8080
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install tokenizers regex requests beautifulsoup4 lxml markdownify
```

## Rebuild corpus + train

```bash
.venv/bin/python scripts/build_wiki_faithful_markdown.py
.venv/bin/python scripts/train_bpe_tokenizer.py
.venv/bin/python scripts/evaluate_bpe_tokenizer.py
```

Outputs:

- `static/bpe/tokenizer.json` — HuggingFace BPE + Metaspace (encode + decode)
- `static/bpe/metrics.json` — training metrics
- `static/bpe/report.json` — widget summary stats
- `static/bpe/corpus/*.faithful.txt` — faithful Markdown corpora (en, hi, te, ta)
