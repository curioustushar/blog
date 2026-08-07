"""Manifest generation and validation."""

from dataclasses import dataclass, asdict
from typing import List
from pathlib import Path
import json
import logging

from .shards import Shard
from .documents import Document

log = logging.getLogger(__name__)


@dataclass
class Manifest:
    """Shard manifest with metadata."""
    shard_id: str
    content_hash: str
    tokenizer_hash: str
    token_count: int
    document_count: int
    split: str  # train, eval, validation
    source: str  # wikipedia, books, code, math
    created_at: str
    version: str = "1.0"


def create_manifests(
    shards: List[Shard],
    documents: List[Document],
    tokenizer_hash: str,
    output_dir: Path
) -> List[Manifest]:
    """Create manifests for shards.
    
    Args:
        shards: List of shards
        documents: Original documents
        tokenizer_hash: Frozen tokenizer hash
        output_dir: Directory to save manifests
        
    Returns:
        List of created manifests
    """
    # Create doc_id -> document mapping
    doc_map = {doc.doc_id: doc for doc in documents}
    
    manifests = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for shard in shards:
        # Determine split and source from documents in shard
        splits = set()
        sources = set()
        
        for doc_id in shard.document_ids:
            if doc_id in doc_map:
                splits.add(doc_map[doc_id].split)
                sources.add(doc_map[doc_id].source)
        
        # Use most common split/source (or first if tie)
        split = list(splits)[0] if splits else "train"
        source = list(sources)[0] if sources else "mixed"
        
        manifest = Manifest(
            shard_id=shard.shard_id,
            content_hash=shard.content_hash,
            tokenizer_hash=tokenizer_hash,
            token_count=len(shard.token_ids),
            document_count=len(shard.document_ids),
            split=split,
            source=source,
            created_at=shard.created_at
        )
        
        manifests.append(manifest)
        
        # Save manifest
        manifest_path = output_dir / f"{shard.shard_id}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(asdict(manifest), f, indent=2)
    
    log.info(f"Created {len(manifests)} manifests")
    
    return manifests


def validate_manifest(shard: Shard, manifest: Manifest, tokenizer_hash: str) -> bool:
    """Validate shard against manifest.
    
    Returns:
        True if valid, False otherwise
    """
    # Check content hash
    if shard.content_hash != manifest.content_hash:
        log.error(f"Content hash mismatch: {shard.shard_id}")
        return False
    
    # Check tokenizer hash
    if manifest.tokenizer_hash != tokenizer_hash:
        log.error(f"Tokenizer hash mismatch: {shard.shard_id}")
        return False
    
    # Check token count
    if len(shard.token_ids) != manifest.token_count:
        log.error(f"Token count mismatch: {shard.shard_id}")
        return False
    
    return True


def load_manifests(manifest_dir: Path) -> List[Manifest]:
    """Load all manifests from directory."""
    manifests = []
    
    for manifest_path in sorted(manifest_dir.glob("*_manifest.json")):
        with open(manifest_path) as f:
            data = json.load(f)
            manifests.append(Manifest(**data))
    
    return manifests
