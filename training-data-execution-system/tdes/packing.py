"""Sequence packing with attention, loss, and position masks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np

from .shards import Shard

PAD_TOKEN = 0
EOS_TOKEN = 3


@dataclass
class PackedBatch:
    """A packed training batch with full mask metadata."""
    batch_id: str
    global_step: int
    input_ids: List[int]
    attention_mask: List[List[int]]
    loss_mask: List[int]
    position_ids: List[int]
    shard_ids: List[str]
    sample_ids: List[str]
    token_spans: List[Tuple[int, int]]
    mixture_lane: str
    curriculum_stage: int
    seq_len: int
    batch_hash: str
    loss_mask_hash: str


def hash_loss_mask(loss_mask: List[int]) -> str:
    payload = json.dumps(loss_mask, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def hash_batch(batch: PackedBatch) -> str:
    payload = json.dumps(
        {
            "batch_id": batch.batch_id,
            "input_ids": batch.input_ids,
            "loss_mask": batch.loss_mask,
            "position_ids": batch.position_ids,
            "shard_ids": batch.shard_ids,
            "token_spans": batch.token_spans,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _build_causal_mask(seq_len: int, doc_boundaries: List[Tuple[int, int]]) -> List[List[int]]:
    """Causal attention within each document; blocked across documents."""
    mask = [[0] * seq_len for _ in range(seq_len)]
    for start, end in doc_boundaries:
        for i in range(start, end):
            for j in range(start, i + 1):
                mask[i][j] = 1
    return mask


def pack_shard_segment(
    batch_id: str,
    global_step: int,
    shard: Shard,
    doc_start_idx: int,
    seq_len: int,
    mixture_lane: str,
    curriculum_stage: int,
) -> PackedBatch:
    """Pack one or more documents from a shard into a fixed-length batch."""
    input_ids = [PAD_TOKEN] * seq_len
    loss_mask = [0] * seq_len
    position_ids = [0] * seq_len
    shard_ids: List[str] = []
    sample_ids: List[str] = []
    token_spans: List[Tuple[int, int]] = []
    doc_boundaries: List[Tuple[int, int]] = []

    pos = 0
    doc_idx = doc_start_idx

    while doc_idx < len(shard.document_ids) and pos < seq_len:
        start = int(shard.offsets[doc_idx])
        end = int(shard.offsets[doc_idx + 1]) if doc_idx + 1 < len(shard.offsets) else len(shard.token_ids)
        tokens = shard.token_ids[start:end].tolist()

        if pos + len(tokens) + 1 > seq_len:
            break

        doc_start = pos
        for tok in tokens:
            input_ids[pos] = int(tok)
            loss_mask[pos] = 1
            position_ids[pos] = pos - doc_start
            pos += 1

        if pos < seq_len:
            input_ids[pos] = EOS_TOKEN
            loss_mask[pos] = 0  # EOS is boundary, not predicted
            position_ids[pos] = pos - doc_start
            pos += 1

        shard_ids.append(shard.shard_id)
        sample_ids.append(shard.document_ids[doc_idx])
        token_spans.append((start, end))
        doc_boundaries.append((doc_start, pos))
        doc_idx += 1

    attention_mask = _build_causal_mask(seq_len, doc_boundaries)
    loss_mask_hash = hash_loss_mask(loss_mask)

    batch = PackedBatch(
        batch_id=batch_id,
        global_step=global_step,
        input_ids=input_ids,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        position_ids=position_ids,
        shard_ids=shard_ids,
        sample_ids=sample_ids,
        token_spans=[(int(a), int(b)) for a, b in token_spans],
        mixture_lane=mixture_lane,
        curriculum_stage=curriculum_stage,
        seq_len=seq_len,
        batch_hash="",
        loss_mask_hash=loss_mask_hash,
    )
    batch.batch_hash = hash_batch(batch)
    return batch


def packing_utilization(batch: PackedBatch) -> float:
  useful = sum(batch.loss_mask)
  return useful / batch.seq_len if batch.seq_len else 0.0


def save_packed_batch(batch: PackedBatch, path) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(batch)
    p.write_text(json.dumps(data, indent=2))
