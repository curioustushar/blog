"""Historical replay and fork verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .ledgers import ConsumptionLedger, ConsumptionRecord


def verify_replay(
    original: List[ConsumptionRecord],
    replayed: List[ConsumptionRecord],
) -> dict:
    original_hashes = [r.batch_hash for r in original]
    replayed_hashes = [r.batch_hash for r in replayed]
    match = original_hashes == replayed_hashes
    return {
        "match": match,
        "original_count": len(original_hashes),
        "replayed_count": len(replayed_hashes),
        "original_hashes": original_hashes,
        "replayed_hashes": replayed_hashes,
        "mismatches": [
            {"step": o.global_step, "original": o.batch_hash, "replayed": r.batch_hash}
            for o, r in zip(original, replayed)
            if o.batch_hash != r.batch_hash
        ],
    }


def save_replay_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
