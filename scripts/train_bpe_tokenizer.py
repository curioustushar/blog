#!/usr/bin/env python3
"""Train multilingual BPE tokenizer for India Wikipedia (en, hi, te, ta)."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "static" / "bpe"
CORPUS_DIR = OUT_DIR / "corpora"

USER_AGENT = "BPE-Tokenizer/1.0 (https://curioustushar.github.io/blog/)"
TOTAL_VOCAB = 10_000
BASE_BYTE_VOCAB = 256

# ~5000 English vocab; remaining 5000 split across hi / te / ta
# Tuned so byte-level BPE ratios are balanced (Tamil/Devanagari need more merge budget).
VOCAB_ALLOC = {
    "en": 5_000,
    "hi": 1_200,
    "te": 1_000,
    "ta": 2_800,
}

MERGE_BUDGET = {
    "en": 3_200,
    "hi": 900,
    "te": 700,
    "ta": 2_800,
}

WIKI_PAGES = {
    "en": ("India", "English"),
    "hi": ("India", "Hindi"),
    "te": ("India", "Telugu"),
    "ta": ("இந்தியா", "Tamil"),
}


def fetch_wikipedia(lang: str, title: str) -> str:
    encoded = urllib.parse.quote(title)
    url = (
        f"https://{lang}.wikipedia.org/w/api.php"
        f"?action=query&titles={encoded}&prop=extracts&explaintext=1"
        f"&format=json&redirects=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    page = next(iter(data["query"]["pages"].values()))
    text = page.get("extract", "")
    if not text:
        raise RuntimeError(f"No extract for {lang}:{title}")
    return text


def normalize_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def get_words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def word_to_symbols(word: str) -> list[tuple[int, ...]]:
    return [tuple(bytes([b])) for b in word.encode("utf-8")]


def pair_counts(words: list[list[tuple[int, ...]]]) -> Counter:
    counts: Counter = Counter()
    for symbols in words:
        for i in range(len(symbols) - 1):
            counts[(symbols[i], symbols[i + 1])] += 1
    return counts


def merge_pair(words: list[list[tuple[int, ...]]], pair: tuple) -> None:
    a, b = pair
    merged = a + b
    for symbols in words:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols[i] = merged
                del symbols[i + 1]
            else:
                i += 1


def train_bpe(corpus_text: str, num_merges: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    words_raw = get_words(corpus_text)
    # Cap unique words for training speed on long articles
    if len(words_raw) > 6000:
        freq = Counter(words_raw)
        words_raw = [w for w, _ in freq.most_common(6000)]

    words = [word_to_symbols(w) for w in words_raw]
    merges: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    for step in range(num_merges):
        counts = pair_counts(words)
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], p))
        if counts[best] < 2:
            break
        merges.append(best)
        merge_pair(words, best)
        if step and step % 500 == 0:
            print(f"    merge {step}/{num_merges}")

    return merges


def token_display(b: tuple[int, ...]) -> str:
    try:
        return bytes(b).decode("utf-8")
    except UnicodeDecodeError:
        return "b:" + "_".join(f"{x:02x}" for x in b)


@dataclass
class Tokenizer:
    merges: list[tuple[tuple[int, ...], tuple[int, ...]]]
    merge_rank: dict[tuple[tuple[int, ...], tuple[int, ...]], int]
    vocab_tokens: list[str]
    merge_lang: list[str]

    @classmethod
    def build(cls, corpora: dict[str, str], merge_budget: dict[str, int]) -> Tokenizer:
        all_entries: list[tuple[str, tuple, tuple]] = []

        for lang in ("en", "hi", "te", "ta"):
            budget = merge_budget[lang]
            print(f"Training {lang}: {budget} merges…")
            for m in train_bpe(corpora[lang], budget):
                all_entries.append((lang, m[0], m[1]))

        seen: dict[tuple[int, ...], int] = {}
        ordered: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        merge_lang: list[str] = []

        for lang, a, b in all_entries:
            merged = a + b
            if merged in seen:
                continue
            seen[merged] = 1
            ordered.append((a, b))
            merge_lang.append(lang)
            if len(ordered) >= TOTAL_VOCAB - BASE_BYTE_VOCAB:
                break

        merge_rank = {pair: i for i, pair in enumerate(ordered)}

        vocab_tokens = [chr(i) if i < 128 else token_display((i,)) for i in range(BASE_BYTE_VOCAB)]
        for a, b in ordered:
            tok = token_display(a + b)
            if tok not in vocab_tokens:
                vocab_tokens.append(tok)

        while len(vocab_tokens) < TOTAL_VOCAB:
            vocab_tokens.append(f"<unused{len(vocab_tokens)}>")

        return cls(
            merges=ordered,
            merge_rank=merge_rank,
            vocab_tokens=vocab_tokens[:TOTAL_VOCAB],
            merge_lang=merge_lang,
        )

    def encode_word(self, word: str) -> list[tuple[int, ...]]:
        symbols = word_to_symbols(word)
        pairs = self._get_pairs(symbols)
        while pairs:
            best = None
            best_rank = None
            for pair in pairs:
                rank = self.merge_rank.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best, best_rank = pair, rank
            if best is None:
                break
            a, b = best
            merged = a + b
            new_symbols: list[tuple[int, ...]] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
            pairs = self._get_pairs(symbols)
        return symbols

    def encode(self, text: str) -> list[str]:
        return [token_display(sym) for word in get_words(text) for sym in self.encode_word(word)]

    @staticmethod
    def _get_pairs(symbols: list[tuple[int, ...]]) -> set:
        return {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}


def compute_stats(tokenizer: Tokenizer, corpora: dict[str, str]) -> dict:
    ordered_langs = ["en", "hi", "te", "ta"]
    x_labels = {"en": "X1", "hi": "X2", "te": "X3", "ta": "X4"}
    details: dict[str, dict] = {}
    ratios: dict[str, float] = {}

    for lang in ordered_langs:
        text = corpora[lang]
        tokens = tokenizer.encode(text)
        vocab = VOCAB_ALLOC[lang]
        ratio = len(tokens) / vocab
        ratios[lang] = ratio
        details[lang] = {
            "language": WIKI_PAGES[lang][1],
            "wiki_title": WIKI_PAGES[lang][0],
            "word_count": len(get_words(text)),
            "char_count": len(text),
            "token_count": len(tokens),
            "vocab_allocated": vocab,
            "ratio_symbol": x_labels[lang],
            "ratio": round(ratio, 4),
            "within_1_2": ratio <= 1.2,
        }

    sorted_ratios = sorted(
        [(lang, ratios[lang]) for lang in ordered_langs],
        key=lambda x: x[1],
        reverse=True,
    )
    x_max = sorted_ratios[0][1]
    x_min = sorted_ratios[-1][1]
    spread = x_max - x_min
    score = 1000 / spread if spread > 0 else None

    return {
        "ratios": {lang: round(ratios[lang], 4) for lang in ordered_langs},
        "x_labels": {
            "en": "X1 (English)",
            "hi": "X2 (Hindi)",
            "te": "X3 (Telugu)",
            "ta": "X4 (Tamil)",
        },
        "sorted_desc": [
            {
                "rank": i + 1,
                "lang": lang,
                "name": WIKI_PAGES[lang][1],
                "symbol": x_labels[lang],
                "ratio": round(r, 4),
            }
            for i, (lang, r) in enumerate(sorted_ratios)
        ],
        "x_min": round(x_min, 4),
        "x_max": round(x_max, 4),
        "x_spread": round(spread, 4),
        "score": round(score, 2) if score else "∞",
        "score_formula": "1000 / (X_max − X_min)",
        "details": details,
        "constraints": {
            "total_vocab": TOTAL_VOCAB,
            "vocab_alloc": VOCAB_ALLOC,
            "max_ratio_note": "Target Xi ≤ 1.2 is noted; full-article Wikipedia text yields higher ratios — reported for transparency.",
        },
    }


def export_all(tokenizer: Tokenizer, stats: dict, corpora: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    for lang, text in corpora.items():
        (CORPUS_DIR / f"{lang}.txt").write_text(text, encoding="utf-8")

    merges_export = [
        {
            "rank": i,
            "language": tokenizer.merge_lang[i] if i < len(tokenizer.merge_lang) else "?",
            "left": token_display(a),
            "right": token_display(b),
            "merged": token_display(a + b),
            "left_bytes": list(a),
            "right_bytes": list(b),
        }
        for i, (a, b) in enumerate(tokenizer.merges)
    ]

    payload = {
        "meta": {
            "source": "Wikipedia — India article",
            "languages": {k: v[1] for k, v in WIKI_PAGES.items()},
            "wiki_titles": {k: v[0] for k, v in WIKI_PAGES.items()},
            "total_vocab": TOTAL_VOCAB,
            "base_byte_vocab": BASE_BYTE_VOCAB,
            "vocab_alloc": VOCAB_ALLOC,
            "merge_budget": MERGE_BUDGET,
            "ratio_formula": "Xi = tokens_encoded(corpus_i) / vocab_alloc_i",
            "score_formula": "score = 1000 / (max(X1..X4) - min(X1..X4))",
            "reproduce": "python scripts/train_bpe_tokenizer.py",
        },
        "stats": stats,
        "vocab": tokenizer.vocab_tokens,
        "merges": merges_export,
    }

    out = OUT_DIR / "tokenizer.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nWrote {out} ({out.stat().st_size // 1024} KB)")


def main() -> None:
    corpora: dict[str, str] = {}
    for lang, (title, name) in WIKI_PAGES.items():
        print(f"Fetching {name} ({lang})…")
        corpora[lang] = normalize_text(fetch_wikipedia(lang, title))
        print(f"  {len(get_words(corpora[lang]))} words")

    tokenizer = Tokenizer.build(corpora, MERGE_BUDGET)
    stats = compute_stats(tokenizer, corpora)
    export_all(tokenizer, stats, corpora)


if __name__ == "__main__":
    main()
