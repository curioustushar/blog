"""Mixture timeline compiler and lane sampler."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

from .manifests import Manifest


@dataclass
class MixtureSchedule:
    """Compiled executable mixture schedule."""
    stages: List[dict]
    protected_floors: Dict[str, float]
    total_training_tokens: int


def compile_schedule(curriculum, allowed_manifests: List[Manifest]) -> MixtureSchedule:
    """Compile curriculum stages and verify shard supply per lane."""
    lane_supply = {}
    for m in allowed_manifests:
        lane_supply[m.source] = lane_supply.get(m.source, 0) + m.token_count

    stages = []
    for stage in curriculum:
        stages.append(
            {
                "stage_id": stage.stage_id,
                "token_range": stage.token_range,
                "lane_weights": stage.lane_weights,
                "protected_floors": stage.protected_floors,
                "lane_supply": {k: lane_supply.get(k, 0) for k in stage.lane_weights},
            }
        )

    floors = curriculum[-1].protected_floors if curriculum else {}
    total = sum(lane_supply.values())
    return MixtureSchedule(stages=stages, protected_floors=floors, total_training_tokens=total)


def stage_for_tokens(tokens_seen: int, curriculum) -> Tuple[int, Dict[str, float], Dict[str, float]]:
    for stage in curriculum:
        start, end = stage.token_range
        if tokens_seen < end:
            return stage.stage_id, stage.lane_weights, stage.protected_floors
    last = curriculum[-1]
    return last.stage_id, last.lane_weights, last.protected_floors


def sample_lane(
    rng: random.Random,
    weights: Dict[str, float],
    lane_counts: Dict[str, int],
    total_samples: int,
    protected_floors: Dict[str, float],
) -> str:
    """Sample a lane; enforce protected floors when under quota."""
    if total_samples > 0:
        for lane, floor in protected_floors.items():
            actual = lane_counts.get(lane, 0) / total_samples
            if actual < floor and lane in weights:
                return lane

    lanes = list(weights.keys())
    probs = [weights[l] for l in lanes]
    return rng.choices(lanes, weights=probs, k=1)[0]


def measure_compliance(records: List[dict], planned_weights: Dict[str, float], tol: float = 0.15) -> dict:
    counts: Dict[str, int] = {}
    for r in records:
        lane = r["mixture_lane"]
        counts[lane] = counts.get(lane, 0) + 1
    total = sum(counts.values()) or 1
    actual = {k: v / total for k, v in counts.items()}
    compliant = all(abs(actual.get(l, 0) - w) <= tol for l, w in planned_weights.items())
    return {"planned": planned_weights, "actual": actual, "compliant": compliant}


def save_mixture_report(schedule: MixtureSchedule, compliance: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schedule": asdict(schedule), "compliance": compliance}, indent=2))
