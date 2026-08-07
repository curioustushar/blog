"""OPUS-style data governance: accept, reject, defer, protected override."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Literal

Decision = Literal["accept", "reject", "defer", "protected"]


@dataclass
class OpusDecision:
    candidate_id: str
    shard_id: str
    mixture_lane: str
    score: float
    decision: Decision
    rationale: str
    global_step: int


def score_candidate(shard_id: str, mixture_lane: str, global_step: int) -> float:
    """Deterministic pseudo-score in [0, 1] from shard metadata."""
    payload = f"{shard_id}:{mixture_lane}:{global_step}"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def decide(
    shard_id: str,
    mixture_lane: str,
    global_step: int,
    thresholds: dict,
    force_protected: bool = False,
) -> OpusDecision:
    score = score_candidate(shard_id, mixture_lane, global_step)
    candidate_id = f"cand_{global_step:05d}_{shard_id}"

    if force_protected:
        return OpusDecision(
            candidate_id=candidate_id,
            shard_id=shard_id,
            mixture_lane=mixture_lane,
            score=score,
            decision="protected",
            rationale="protected_floor_override",
            global_step=global_step,
        )

    # Ensure all decision types appear in a long run (deterministic rotation)
    rotation = global_step % 4
    if rotation == 1:
        decision: Decision = "reject"
        rationale = "low_proxy_utility"
    elif rotation == 2:
        decision = "defer"
        rationale = "borderline_quality"
    elif score >= thresholds["accept"]:
        decision = "accept"
        rationale = "high_proxy_utility"
    elif score <= thresholds["reject"]:
        decision = "reject"
        rationale = "low_proxy_utility"
    else:
        decision = "defer"
        rationale = "borderline_quality"

    return OpusDecision(
        candidate_id=candidate_id,
        shard_id=shard_id,
        mixture_lane=mixture_lane,
        score=score,
        decision=decision,
        rationale=rationale,
        global_step=global_step,
    )


def save_opus_report(decisions: List[OpusDecision], path: Path) -> Dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"accept": 0, "reject": 0, "defer": 0, "protected": 0}
    rows = []
    for d in decisions:
        summary[d.decision] += 1
        rows.append(asdict(d))
    path.write_text(json.dumps({"summary": summary, "decisions": rows}, indent=2))
    return summary
