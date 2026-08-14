#!/usr/bin/env python3
"""
Minimal 30-second injectivity test on 100 tokens.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from kronecker_encoder import KroneckerEncoderStaged

# Test vocabulary
vocab = [
    # Common words
    "the", "a", "is", "and", "to", "of", "in", "for", "on", "with",
    # Variations (test case sensitivity)
    "test", "Test", "TEST", " test", "test ",
    # Repeated characters (adversarial)
    "a", "aa", "aaa", "aaaa", "aaaaa",
    "b", "bb", "bbb",
    # Patterns
    "abc", "abcabc", "abcabcabc",
    "prefix", "prefix1", "prefix2",
    # Numbers
    "0", "1", "01", "10", "100", "1000",
    # Unicode
    "café", "naïve", "世界", "🔥",
] + [f"word{i}" for i in range(50)]  # Add 50 synthetic tokens

print(f"Testing {len(vocab)} tokens\n")

# Create encoder
encoder = KroneckerEncoderStaged(
    char_dim=256,
    pos_dim=32,
    apply_length_norm=True,
    apply_z_norm=False,
    apply_projection=False,
)

# Encode all
representations = {}
for token in vocab:
    rep, stages = encoder.encode(token, return_stages=True)
    representations[token] = rep

# Check for exact duplicates
hashes = {}
for token, rep in representations.items():
    h = hash(rep.tobytes())
    if h not in hashes:
        hashes[h] = []
    hashes[h].append(token)

# Find collisions
collisions = [tokens for tokens in hashes.values() if len(tokens) > 1]

print(f"Results:")
print(f"  Total tokens: {len(vocab)}")
print(f"  Unique representations: {len(hashes)}")
print(f"  Collision groups: {len(collisions)}\n")

if collisions:
    print(f"⚠️  COLLISIONS FOUND:\n")
    for group in collisions:
        print(f"  Colliding tokens: {group}")
        for token in group:
            print(f"    '{token}' → bytes: {token.encode('utf-8').hex()}")
        print()
    print(f"VERDICT: Collision detected — representation is NOT injective")
else:
    print(f"✅ No collisions found in {len(vocab)} tokens")
    
    # Compute minimum distance
    min_dist = float('inf')
    min_pair = None
    tokens_list = list(vocab)
    for i in range(len(tokens_list)):
        for j in range(i+1, len(tokens_list)):
            t1, t2 = tokens_list[i], tokens_list[j]
            dist = np.linalg.norm(representations[t1] - representations[t2])
            if dist < min_dist:
                min_dist = dist
                min_pair = (t1, t2)
    
    print(f"  Minimum pairwise distance: {min_dist:.2e}")
    print(f"  Closest pair: '{min_pair[0]}' vs '{min_pair[1]}'")
    print()
    print(f"VERDICT: No collision found (empirical, N={len(vocab)})")

print(f"\nDimension: {representations[vocab[0]].shape[0]} (char_dim × pos_dim = 256 × 32 = 8192)")
