"""Tests for TDES core invariants."""

import json
import random
from pathlib import Path

import pytest

from tdes.firewall import filter_training_shards
from tdes.manifests import Manifest
from tdes.opus import decide
from tdes.packing import hash_batch, pack_shard_segment, packing_utilization
from tdes.replay import verify_replay
from tdes.shards import Shard
from tdes.training import TrainingState, build_batch


def _fake_shard() -> Shard:
    import numpy as np
    return Shard(
        shard_id="shard_test",
        token_ids=np.array([10, 11, 12, 13, 14, 15], dtype=np.int32),
        document_ids=["doc_a", "doc_b"],
        offsets=np.array([0, 3, 6], dtype=np.int32),
        content_hash="abc",
        created_at="2026-01-01",
    )


def _fake_manifest() -> Manifest:
    return Manifest(
        shard_id="shard_test",
        content_hash="abc",
        tokenizer_hash="tok",
        token_count=6,
        document_count=2,
        split="train",
        source="math",
        created_at="2026-01-01",
    )


def test_packing_masks_and_hash():
    batch = pack_shard_segment("batch_00001", 1, _fake_shard(), 0, 16, "math", 1)
    assert len(batch.input_ids) == 16
    assert len(batch.loss_mask) == 16
    assert len(batch.position_ids) == 16
    assert len(batch.attention_mask) == 16
    assert batch.batch_hash == hash_batch(batch)
    assert packing_utilization(batch) > 0


def test_eval_firewall_blocks_eval():
    manifests = [
        Manifest("s1", "h", "t", 10, 1, "train", "math", "now"),
        Manifest("s2", "h", "t", 10, 1, "eval", "math", "now"),
    ]
    allowed, blocked = filter_training_shards(manifests, Path("/tmp/tdes_test_manifests"))
    assert len(allowed) == 1
    assert len(blocked) == 1
    assert blocked[0].split == "eval"


def test_opus_all_decision_types():
    thresholds = {"accept": 0.7, "reject": 0.3}
    decisions = {decide("s", "math", i, thresholds, force_protected=(i % 20 == 0)).decision for i in range(40)}
    assert "accept" in decisions
    assert "reject" in decisions
    assert "defer" in decisions
    assert "protected" in decisions


def test_deterministic_batch_build():
    import config
    shard = _fake_shard()
    manifest = _fake_manifest()
    state = TrainingState(4, 0, {}, 1, "main", 42)
    thresholds = {"accept": 0.0, "reject": -1.0}
    b1, _ = build_batch(state, [shard], [manifest], config.CURRICULUM, 16, thresholds)
    b2, _ = build_batch(state, [shard], [manifest], config.CURRICULUM, 16, thresholds)
    assert b1 is not None and b2 is not None
    assert b1.batch_hash == b2.batch_hash


def test_replay_hash_match():
    from tdes.ledgers import ConsumptionRecord

    rec = ConsumptionRecord(
        run_id="r",
        branch_id="main",
        global_step=1,
        checkpoint_id=None,
        batch_id="batch_00001",
        shard_ids=["s"],
        sample_ids=["d"],
        token_spans=[(0, 3)],
        loss_mask_hash="h",
        mixture_lane="math",
        curriculum_stage=1,
        opus_decision_id="o",
        batch_hash="hash1",
        timestamp="now",
    )
    report = verify_replay([rec], [rec])
    assert report["match"] is True
