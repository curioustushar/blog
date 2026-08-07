"""Consumption and learning ledgers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .packing import PackedBatch


@dataclass
class ConsumptionRecord:
    run_id: str
    branch_id: str
    global_step: int
    checkpoint_id: Optional[str]
    batch_id: str
    shard_ids: List[str]
    sample_ids: List[str]
    token_spans: List[tuple]
    loss_mask_hash: str
    mixture_lane: str
    curriculum_stage: int
    opus_decision_id: str
    batch_hash: str
    timestamp: str


@dataclass
class LearningRecord:
    batch_id: str
    global_step: int
    loss: float
    sample_ids: List[str]
    gradient_norm: float
    learning_rate: float
    checkpoint_id: Optional[str]
    branch_id: str


class ConsumptionLedger:
    def __init__(self, path: Path, run_id: str, branch_id: str = "main"):
        self.path = path
        self.run_id = run_id
        self.branch_id = branch_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._offset = 0
        if self.path.exists():
            self._offset = sum(1 for _ in open(self.path))

    @property
    def offset(self) -> int:
        return self._offset

    def append(self, batch: PackedBatch, opus_decision_id: str, checkpoint_id: Optional[str] = None) -> ConsumptionRecord:
        record = ConsumptionRecord(
            run_id=self.run_id,
            branch_id=self.branch_id,
            global_step=batch.global_step,
            checkpoint_id=checkpoint_id,
            batch_id=batch.batch_id,
            shard_ids=batch.shard_ids,
            sample_ids=batch.sample_ids,
            token_spans=batch.token_spans,
            loss_mask_hash=batch.loss_mask_hash,
            mixture_lane=batch.mixture_lane,
            curriculum_stage=batch.curriculum_stage,
            opus_decision_id=opus_decision_id,
            batch_hash=batch.batch_hash,
            timestamp=datetime.now().isoformat(),
        )
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        self._offset += 1
        return record

    def read_all(self) -> List[ConsumptionRecord]:
        if not self.path.exists():
            return []
        records = []
        with open(self.path) as f:
            for line in f:
                data = json.loads(line)
                records.append(ConsumptionRecord(**data))
        return records

    def read_range(self, start_step: int, end_step: int) -> List[ConsumptionRecord]:
        return [r for r in self.read_all() if start_step <= r.global_step <= end_step]


class LearningLedger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: LearningRecord) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def read_all(self) -> List[LearningRecord]:
        if not self.path.exists():
            return []
        records = []
        with open(self.path) as f:
            for line in f:
                records.append(LearningRecord(**json.loads(line)))
        return records
