"""Checkpoint save/load with ledger binding."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class Checkpoint:
    checkpoint_id: str
    global_step: int
    ledger_offset: int
    next_batch_id: str
    branch_id: str
    run_id: str
    rng_state: int
    tokens_seen: int
    lane_counts: Dict[str, int]
    mixture_stage: int


def save_checkpoint(cp: Checkpoint, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cp), indent=2))


def load_checkpoint(path: Path) -> Checkpoint:
    data = json.loads(path.read_text())
    return Checkpoint(**data)
