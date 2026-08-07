"""Deterministic training loop with ledgers and batch generation."""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from .ledgers import ConsumptionLedger, LearningLedger, LearningRecord
from .manifests import Manifest
from .mixture import sample_lane, stage_for_tokens
from .opus import OpusDecision, decide
from .packing import PackedBatch, pack_shard_segment, save_packed_batch
from .shards import Shard


@dataclass
class TrainingState:
    global_step: int
    tokens_seen: int
    lane_counts: Dict[str, int]
    next_batch_num: int
    branch_id: str
    master_seed: int


def pseudo_loss(batch: PackedBatch) -> float:
    useful = sum(batch.loss_mask)
    digest = hashlib.sha256(batch.batch_hash.encode()).hexdigest()
    base = int(digest[:6], 16) / 0xFFFFFF
    return round(2.5 - base + (useful / max(batch.seq_len, 1)) * 0.5, 4)


def _shards_by_lane(shards: List[Shard], manifests: List[Manifest]) -> Dict[str, List[Shard]]:
    manifest_map = {m.shard_id: m for m in manifests}
    by_lane: Dict[str, List[Shard]] = {}
    for shard in shards:
        lane = manifest_map[shard.shard_id].source
        by_lane.setdefault(lane, []).append(shard)
    return by_lane


def _pick_shard(rng: random.Random, lane: str, by_lane: Dict[str, List[Shard]]) -> Tuple[Shard, int]:
    pool = by_lane.get(lane) or next(iter(by_lane.values()))
    shard = rng.choice(pool)
    doc_idx = rng.randrange(len(shard.document_ids))
    return shard, doc_idx


def build_batch(
    state: TrainingState,
    shards: List[Shard],
    manifests: List[Manifest],
    curriculum,
    seq_len: int,
    opus_thresholds: dict,
) -> Tuple[Optional[PackedBatch], OpusDecision]:
    rng = random.Random(state.master_seed + state.global_step)
    stage_id, weights, floors = stage_for_tokens(state.tokens_seen, curriculum)
    lane = sample_lane(
        rng,
        weights,
        state.lane_counts,
        max(state.global_step, 1),
        floors,
    )
    force = state.global_step > 5 and (
        state.lane_counts.get("math", 0) / max(state.global_step, 1) < floors.get("math", 0)
    )

    shard, doc_idx = _pick_shard(rng, lane, _shards_by_lane(shards, manifests))
    opus = decide(shard.shard_id, lane, state.global_step, opus_thresholds, force_protected=force)

    if opus.decision in ("reject", "defer"):
        return None, opus

    batch_id = f"batch_{state.next_batch_num:05d}"
    batch = pack_shard_segment(
        batch_id=batch_id,
        global_step=state.global_step,
        shard=shard,
        doc_start_idx=doc_idx,
        seq_len=seq_len,
        mixture_lane=lane,
        curriculum_stage=stage_id,
    )
    return batch, opus


def train_steps(
    state: TrainingState,
    shards: List[Shard],
    manifests: List[Manifest],
    curriculum,
    seq_len: int,
    opus_thresholds: dict,
    consumption: ConsumptionLedger,
    learning: LearningLedger,
    packed_dir: Path,
    num_steps: int,
    learning_rate: float,
    checkpoint_id: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_at: Optional[set[int]] = None,
    run_id: str = "run_001",
) -> Tuple[TrainingState, List[PackedBatch], List[OpusDecision], float]:
    batches: List[PackedBatch] = []
    decisions: List[OpusDecision] = []
    start = time.perf_counter()
    tokens_processed = 0

    for _ in range(num_steps):
        batch, opus = build_batch(state, shards, manifests, curriculum, seq_len, opus_thresholds)
        decisions.append(opus)
        if batch is None:
            state.global_step += 1
            if checkpoint_dir and checkpoint_at and state.global_step in checkpoint_at:
                cp = make_checkpoint(state, run_id)
                save_checkpoint(cp, checkpoint_dir / f"{cp.checkpoint_id}.json")
            continue

        save_packed_batch(batch, packed_dir / f"{batch.batch_id}.json")
        consumption.append(batch, opus.candidate_id, checkpoint_id=checkpoint_id)
        loss = pseudo_loss(batch)
        learning.append(
            LearningRecord(
                batch_id=batch.batch_id,
                global_step=state.global_step,
                loss=loss,
                sample_ids=batch.sample_ids,
                gradient_norm=round(loss * 0.1, 4),
                learning_rate=learning_rate,
                checkpoint_id=checkpoint_id,
                branch_id=state.branch_id,
            )
        )

        batches.append(batch)
        tokens_processed += sum(batch.loss_mask)
        state.lane_counts[batch.mixture_lane] = state.lane_counts.get(batch.mixture_lane, 0) + 1
        state.tokens_seen += sum(batch.loss_mask)
        state.global_step += 1
        state.next_batch_num += 1

        if checkpoint_dir and checkpoint_at and state.global_step in checkpoint_at:
            cp = make_checkpoint(state, run_id)
            save_checkpoint(cp, checkpoint_dir / f"{cp.checkpoint_id}.json")

    elapsed = max(time.perf_counter() - start, 1e-6)
    return state, batches, decisions, tokens_processed / elapsed


def make_checkpoint(state: TrainingState, run_id: str) -> Checkpoint:
    return Checkpoint(
        checkpoint_id=f"ckpt_{state.global_step:05d}",
        global_step=state.global_step,
        ledger_offset=state.global_step,
        next_batch_id=f"batch_{state.next_batch_num:05d}",
        branch_id=state.branch_id,
        run_id=run_id,
        rng_state=state.master_seed,
        tokens_seen=state.tokens_seen,
        lane_counts=dict(state.lane_counts),
        mixture_stage=1,
    )


def state_from_checkpoint(cp: Checkpoint) -> TrainingState:
    batch_num = int(cp.next_batch_id.split("_")[1])
    return TrainingState(
        global_step=cp.global_step,
        tokens_seen=cp.tokens_seen,
        lane_counts=dict(cp.lane_counts),
        next_batch_num=batch_num,
        branch_id=cp.branch_id,
        master_seed=cp.rng_state,
    )
