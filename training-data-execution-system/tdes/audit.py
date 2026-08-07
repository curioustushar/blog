"""Audit reconstruction from artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .ledgers import ConsumptionLedger, LearningLedger


def run_audit(
    consumption_path: Path,
    learning_path: Path,
    checkpoint_step: int,
) -> Dict:
    consumption = ConsumptionLedger(consumption_path, run_id="audit")
    learning = LearningLedger(learning_path)

    cons_records = [r for r in consumption.read_all() if r.global_step <= checkpoint_step]
    learn_records = learning.read_all()

    shards = sorted({sid for r in cons_records for sid in r.shard_ids})
    samples = sorted({sid for r in cons_records for sid in r.sample_ids})
    lanes: Dict[str, int] = {}
    for r in cons_records:
        lanes[r.mixture_lane] = lanes.get(r.mixture_lane, 0) + 1

    losses = [lr.loss for lr in learn_records if lr.global_step <= checkpoint_step]
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    return {
        "checkpoint_step": checkpoint_step,
        "consumed_shards": shards,
        "consumed_samples": samples,
        "batch_count": len(cons_records),
        "mixture_lane_counts": lanes,
        "average_loss": avg_loss,
        "opus_decision_ids": [r.opus_decision_id for r in cons_records],
    }


def save_audit_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
