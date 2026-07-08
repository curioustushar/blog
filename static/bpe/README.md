# BPE Tokenizer Widget

Static widget served with the blog at `/blog/bpe/`.

## Local preview

```bash
cd static/bpe
python3 -m http.server 8080
# open http://localhost:8080
```

## Retrain tokenizer

```bash
python3 scripts/train_bpe_tokenizer.py
```

Regenerates `static/bpe/tokenizer.json` and `static/bpe/corpora/*.txt`.
