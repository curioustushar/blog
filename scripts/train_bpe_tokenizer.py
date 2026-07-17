#!/usr/bin/env python3
"""
Train shared 10k BPE tokenizer for faithful Markdown corpora (en, hi, te, ta).

Uses HuggingFace tokenizers with Metaspace pre-tokenizer/decoder so
decode(encode(text)) preserves visible characters (punctuation, URLs, etc.).

Run:
    python scripts/build_wiki_faithful_markdown.py
    python scripts/train_bpe_tokenizer.py
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import regex
from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "static" / "bpe" / "corpus"
OUT_DIR = ROOT / "static" / "bpe"
OUT_TOKENIZER = OUT_DIR / "tokenizer.json"
OUT_METRICS = OUT_DIR / "metrics.json"
OUT_REPORT = OUT_DIR / "report.json"

LANGS = ["en", "hi", "te", "ta"]
LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
}
WIKI_TITLES = {
    "en": "India",
    "hi": "भारत",
    "te": "భారతదేశం",
    "ta": "இந்தியா",
}
X_LABELS = {
    "en": "X1 (English)",
    "hi": "X2 (Hindi)",
    "te": "X3 (Telugu)",
    "ta": "X4 (Tamil)",
}
WEIGHTS = {"en": 3, "hi": 4, "te": 4, "ta": 2}
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")
ROUNDTRIP_SAMPLE = "India's population is 1,428,627,663."


def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))


def visible_non_ws(text: str) -> str:
    return regex.sub(r"\s", "", text)


def make_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="never")
    tokenizer.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="never")
    return tokenizer


def assert_faithful_roundtrip(tokenizer: Tokenizer) -> None:
    enc = tokenizer.encode(ROUNDTRIP_SAMPLE)
    decoded = tokenizer.decode(enc.ids)
    if visible_non_ws(decoded) != visible_non_ws(ROUNDTRIP_SAMPLE):
        raise ValueError(
            "Tokenizer failed faithful roundtrip gate: "
            f"decode(encode({ROUNDTRIP_SAMPLE!r})) -> {decoded!r}"
        )


def build_report(metrics: dict) -> dict:
    ratios = metrics["ratios"]
    sorted_langs = sorted(LANGS, key=lambda code: ratios[code], reverse=True)
    sorted_desc = []
    for rank, code in enumerate(sorted_langs, start=1):
        ratio = ratios[code]
        sorted_desc.append({
            "rank": rank,
            "lang": code,
            "name": LANG_NAMES[code],
            "symbol": X_LABELS[code].split()[0],
            "eval_page": WIKI_TITLES[code],
            "faithful_units": metrics["faithful_units"][code],
            "token_count": metrics["token_counts"][code],
            "fertility": round(ratio, 6),
            "ratio": round(ratio, 6),
            "threshold": 1.2,
            "threshold_label": "≤ 1.2",
            "status": "pass" if ratio <= 1.2 else "fail",
        })

    details = {}
    for code in LANGS:
        ratio = ratios[code]
        units = metrics["faithful_units"][code]
        tokens = metrics["token_counts"][code]
        details[code] = {
            "language": LANG_NAMES[code],
            "wiki_title": WIKI_TITLES[code],
            "eval_page": WIKI_TITLES[code],
            "faithful_units": units,
            "pretoken_words": units,
            "merged_tokens": tokens,
            "word_count": units,
            "char_count": len(
                (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
            ),
            "token_count": tokens,
            "ratio_symbol": X_LABELS[code].split()[0],
            "fertility": round(ratio, 6),
            "ratio": round(ratio, 6),
            "calculation": f"{tokens} / {units} = {ratio:.6f}",
            "threshold": 1.2,
            "threshold_label": "≤ 1.2",
            "status": "pass" if ratio <= 1.2 else "fail",
            "within_threshold": ratio <= 1.2,
        }

    spread = metrics["spread"]
    score = metrics["score"]
    min_lang = min(LANGS, key=lambda c: ratios[c])
    max_lang = max(LANGS, key=lambda c: ratios[c])

    return {
        "metric": "faithful_unit_fertility",
        "metric_formula": "Xi = token_count / faithful_unit_count",
        "pretoken_note": (
            "Faithful unit = one Unicode letter/mark/number run OR one visible "
            "non-space punctuation/symbol character."
        ),
        "roundtrip_sample": ROUNDTRIP_SAMPLE,
        "roundtrip_ok": True,
        "ratios": {code: round(ratios[code], 6) for code in LANGS},
        "fertilities": {code: round(ratios[code], 6) for code in LANGS},
        "x_labels": X_LABELS,
        "sorted_desc": sorted_desc,
        "x_min": round(ratios[min_lang], 6),
        "x_max": round(ratios[max_lang], 6),
        "x_spread": round(spread, 6),
        "score": round(score, 2),
        "score_formula": "1000 / (max_fertility - min_fertility)",
        "hindi_penalty_factor": round(metrics["hindi_penalty_factor"], 6),
        "hindi_adjusted_score": round(metrics["hindi_adjusted_score"], 2),
        "details": details,
        "thresholds": {"all_languages": 1.2},
        "constraints": {
            "total_vocab": 10_000,
            "variant": "wiki_faithful_markdown",
            "languages": LANG_NAMES,
            "training_weights": WEIGHTS,
        },
    }


def train() -> tuple[Tokenizer, dict]:
    texts = {
        code: (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        for code in LANGS
    }
    units = {code: faithful_units(text) for code, text in texts.items()}

    with tempfile.TemporaryDirectory() as tmp:
        files: list[str] = []
        tmpdir = Path(tmp)
        for code, text in texts.items():
            path = tmpdir / f"{code}.txt"
            path.write_text(text, encoding="utf-8")
            files.extend([str(path)] * WEIGHTS[code])

        tokenizer = make_tokenizer()
        trainer = BpeTrainer(
            vocab_size=10_000,
            min_frequency=1,
            special_tokens=["<unk>"],
        )
        tokenizer.train(files, trainer)

    assert_faithful_roundtrip(tokenizer)

    token_counts = {code: len(tokenizer.encode(text).ids) for code, text in texts.items()}
    ratios = {code: token_counts[code] / units[code] for code in LANGS}
    spread = max(ratios.values()) - min(ratios.values())
    score = 1000 / spread
    hindi_penalty = math.exp(max(0.0, ratios["hi"] / 1.2 - 1.0))

    metrics = {
        "variant": "wiki_faithful_markdown",
        "languages": LANG_NAMES,
        "wiki_titles": WIKI_TITLES,
        "weights": WEIGHTS,
        "vocab_size": tokenizer.get_vocab_size(),
        "faithful_units": units,
        "unit_policy": (
            "Counts each contiguous Unicode letter/mark/number run as one unit and "
            "each visible non-space punctuation/symbol character as one unit."
        ),
        "token_counts": token_counts,
        "ratios": ratios,
        "spread": spread,
        "score": score,
        "hindi_penalty_factor": hindi_penalty,
        "hindi_adjusted_score": score / hindi_penalty,
        "roundtrip_sample": ROUNDTRIP_SAMPLE,
        "roundtrip_ok": True,
        "reproduce": (
            "python scripts/build_wiki_faithful_markdown.py && "
            "python scripts/train_bpe_tokenizer.py"
        ),
    }
    return tokenizer, metrics


def main() -> int:
    missing = [code for code in LANGS if not (CORPUS / f"{code}.faithful.txt").exists()]
    if missing:
        raise SystemExit(
            f"Missing corpus files for {missing}. "
            "Run: python scripts/build_wiki_faithful_markdown.py"
        )

    tokenizer, metrics = train()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUT_TOKENIZER))
    OUT_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    report = build_report(metrics)
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_TOKENIZER}")
    print(f"Wrote {OUT_METRICS}")
    print(f"Wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
