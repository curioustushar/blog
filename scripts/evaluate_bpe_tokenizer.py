#!/usr/bin/env python3
"""Evaluate tokenizer.json on the faithful Markdown corpus."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import regex
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "static" / "bpe" / "corpus"
TOKENIZER = ROOT / "static" / "bpe" / "tokenizer.json"
LANGS = ["en", "hi", "te", "ta"]
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")
ROUNDTRIP_SAMPLE = "India's population is 1,428,627,663."


def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))


def visible_non_ws(text: str) -> str:
    return regex.sub(r"\s", "", text)


def check_roundtrip(tokenizer: Tokenizer) -> None:
    if not hasattr(tokenizer, "decode"):
        raise RuntimeError("tokenizer has no decode method")
    enc = tokenizer.encode(ROUNDTRIP_SAMPLE)
    decoded = tokenizer.decode(enc.ids)
    if visible_non_ws(decoded) != visible_non_ws(ROUNDTRIP_SAMPLE):
        raise RuntimeError(
            "Tokenizer is not acceptable for faithful Markdown evaluation because "
            f"decode(encode(text)) does not preserve visible text: "
            f"{ROUNDTRIP_SAMPLE!r} -> {decoded!r}"
        )


def main() -> int:
    if not TOKENIZER.exists():
        print(f"Missing {TOKENIZER}", file=sys.stderr)
        return 1

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    check_roundtrip(tokenizer)

    rows = {}
    for code in LANGS:
        text = (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        units = faithful_units(text)
        tokens = len(tokenizer.encode(text).ids)
        rows[code] = {"tokens": tokens, "faithful_units": units, "ratio": tokens / units}

    ratios = [row["ratio"] for row in rows.values()]
    spread = max(ratios) - min(ratios)
    score = 1000 / spread
    hindi_penalty = math.exp(max(0.0, rows["hi"]["ratio"] / 1.2 - 1.0))
    result = {
        "roundtrip_ok": True,
        "roundtrip_sample": ROUNDTRIP_SAMPLE,
        "rows": rows,
        "spread": spread,
        "score": score,
        "hindi_penalty_factor": hindi_penalty,
        "hindi_adjusted_score": score / hindi_penalty,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
