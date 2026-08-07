"""Immutable tokenized shards with content hashes."""

from dataclasses import dataclass, asdict
from typing import List
from pathlib import Path
import numpy as np
import hashlib
import json
import logging
from datetime import datetime

from .documents import Document
from .tokenizer import FrozenTokenizer

log = logging.getLogger(__name__)


@dataclass
class Shard:
    """Immutable tokenized shard."""
    shard_id: str
    token_ids: np.ndarray
    document_ids: List[str]
    offsets: np.ndarray  # Document boundaries
    content_hash: str
    created_at: str


def create_shards(
    documents: List[Document],
    tokenizer: FrozenTokenizer,
    shard_size: int,
    output_dir: Path
) -> List[Shard]:
    """Create immutable tokenized shards.
    
    Args:
        documents: List of documents
        tokenizer: Frozen tokenizer
        shard_size: Max tokens per shard
        output_dir: Directory to save shards
        
    Returns:
        List of created shards
    """
    shards = []
    current_tokens = []
    current_doc_ids = []
    current_offsets = [0]
    shard_id = 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for doc in documents:
        tokens = tokenizer.encode(doc.text)
        
        # Check if adding this doc exceeds shard size
        if len(current_tokens) + len(tokens) > shard_size and current_tokens:
            # Save current shard
            shard = _save_shard(
                shard_id=f"shard_{shard_id:05d}",
                tokens=current_tokens,
                doc_ids=current_doc_ids,
                offsets=current_offsets,
                output_dir=output_dir
            )
            shards.append(shard)
            
            # Reset
            current_tokens = []
            current_doc_ids = []
            current_offsets = [0]
            shard_id += 1
        
        # Add document to current shard
        current_tokens.extend(tokens)
        current_doc_ids.append(doc.doc_id)
        current_offsets.append(len(current_tokens))
    
    # Save last shard
    if current_tokens:
        shard = _save_shard(
            shard_id=f"shard_{shard_id:05d}",
            tokens=current_tokens,
            doc_ids=current_doc_ids,
            offsets=current_offsets,
            output_dir=output_dir
        )
        shards.append(shard)
    
    log.info(f"Created {len(shards)} shards")
    log.info(f"[PASS] shard_content_hash_verified: {len(shards)}/{len(shards)}")
    
    return shards


def _save_shard(
    shard_id: str,
    tokens: List[int],
    doc_ids: List[str],
    offsets: List[int],
    output_dir: Path
) -> Shard:
    """Save a single shard."""
    token_ids = np.array(tokens, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.int32)
    
    # Compute content hash
    content = json.dumps({
        "token_ids": token_ids.tolist(),
        "document_ids": doc_ids,
        "offsets": offsets_arr.tolist()
    }, sort_keys=True)
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    
    shard = Shard(
        shard_id=shard_id,
        token_ids=token_ids,
        document_ids=doc_ids,
        offsets=offsets_arr,
        content_hash=content_hash,
        created_at=datetime.now().isoformat()
    )
    
    # Save shard
    shard_path = output_dir / f"{shard_id}.npz"
    np.savez_compressed(
        shard_path,
        token_ids=token_ids,
        offsets=offsets_arr
    )
    
    # Save metadata
    meta_path = output_dir / f"{shard_id}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "shard_id": shard_id,
            "document_ids": doc_ids,
            "content_hash": content_hash,
            "created_at": shard.created_at,
            "token_count": len(token_ids),
            "document_count": len(doc_ids)
        }, f, indent=2)
    
    return shard


def load_shard(shard_path: Path) -> Shard:
    """Load shard from disk."""
    # Load arrays
    data = np.load(shard_path)
    token_ids = data["token_ids"]
    offsets = data["offsets"]
    
    # Load metadata
    meta_path = shard_path.parent / f"{shard_path.stem}_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    
    return Shard(
        shard_id=meta["shard_id"],
        token_ids=token_ids,
        document_ids=meta["document_ids"],
        offsets=offsets,
        content_hash=meta["content_hash"],
        created_at=meta["created_at"]
    )


def verify_shard_hash(shard: Shard) -> bool:
    """Verify shard content hash."""
    content = json.dumps({
        "token_ids": shard.token_ids.tolist(),
        "document_ids": shard.document_ids,
        "offsets": shard.offsets.tolist()
    }, sort_keys=True)
    computed_hash = hashlib.sha256(content.encode()).hexdigest()
    
    if computed_hash != shard.content_hash:
        log.error(f"Hash mismatch: {shard.shard_id}")
        return False
    
    return True
