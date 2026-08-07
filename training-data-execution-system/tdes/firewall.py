"""Evaluation firewall - blocks eval/validation shards from training."""

from typing import List, Tuple
from pathlib import Path
import json
import logging

from .manifests import Manifest

log = logging.getLogger(__name__)


def filter_training_shards(manifests: List[Manifest], output_dir: Path) -> Tuple[List[Manifest], List[Manifest]]:
    """Filter out evaluation and validation shards.
    
    Args:
        manifests: All manifests
        output_dir: Directory to save blocked shard list
        
    Returns:
        (allowed_manifests, blocked_manifests)
    """
    allowed = []
    blocked = []
    
    for manifest in manifests:
        if manifest.split == "train":
            allowed.append(manifest)
        else:
            blocked.append(manifest)
            log.info(f"[BLOCK] eval shard: {manifest.shard_id} (split={manifest.split})")
    
    # Save blocked shards
    output_dir.mkdir(parents=True, exist_ok=True)
    blocked_path = output_dir / "blocked_shards.json"
    
    with open(blocked_path, "w") as f:
        json.dump({
            "blocked_count": len(blocked),
            "blocked_shards": [
                {
                    "shard_id": m.shard_id,
                    "split": m.split,
                    "source": m.source,
                    "token_count": m.token_count
                }
                for m in blocked
            ]
        }, f, indent=2)
    
    log.info(f"[PASS] eval_shard_blocked: {len(blocked)} shards blocked")
    log.info(f"Training shards: {len(allowed)}, Blocked shards: {len(blocked)}")
    
    return allowed, blocked
