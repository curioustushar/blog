"""Frozen tokenizer with hash verification."""

import hashlib
import json
import logging
from pathlib import Path
from typing import List
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

log = logging.getLogger(__name__)


class FrozenTokenizer:
    """Tokenizer with immutable vocabulary and content hash."""
    
    def __init__(self, tokenizer: HFTokenizer, tokenizer_hash: str):
        self.tokenizer = tokenizer
        self.tokenizer_hash = tokenizer_hash
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text).ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids)
    
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


def build_tokenizer(texts: List[str], vocab_size: int, output_dir: Path) -> FrozenTokenizer:
    """Build and freeze a tokenizer.
    
    Args:
        texts: Training texts
        vocab_size: Vocabulary size
        output_dir: Directory to save tokenizer
        
    Returns:
        FrozenTokenizer with computed hash
    """
    # Build tokenizer
    tokenizer = HFTokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]
    )
    
    tokenizer.train_from_iterator(texts, trainer)
    
    # Save tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    
    # Compute hash
    vocab = tokenizer.get_vocab()
    vocab_json = json.dumps(sorted(vocab.items()), sort_keys=True)
    tokenizer_hash = hashlib.sha256(vocab_json.encode()).hexdigest()
    
    # Save hash
    hash_path = output_dir / "tokenizer_hash.txt"
    hash_path.write_text(tokenizer_hash)
    
    log.info(f"Tokenizer built: vocab_size={vocab_size}, hash={tokenizer_hash[:12]}...")
    
    return FrozenTokenizer(tokenizer, tokenizer_hash)


def load_tokenizer(tokenizer_dir: Path) -> FrozenTokenizer:
    """Load frozen tokenizer and verify hash."""
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    hash_path = tokenizer_dir / "tokenizer_hash.txt"
    
    # Load tokenizer
    tokenizer = HFTokenizer.from_file(str(tokenizer_path))
    
    # Load saved hash
    saved_hash = hash_path.read_text().strip()
    
    # Recompute hash
    vocab = tokenizer.get_vocab()
    vocab_json = json.dumps(sorted(vocab.items()), sort_keys=True)
    computed_hash = hashlib.sha256(vocab_json.encode()).hexdigest()
    
    # Verify
    if saved_hash != computed_hash:
        raise ValueError(f"Tokenizer hash mismatch! Saved: {saved_hash}, Computed: {computed_hash}")
    
    log.info(f"[PASS] tokenizer_hash_verified: {saved_hash[:12]}...")
    
    return FrozenTokenizer(tokenizer, saved_hash)
