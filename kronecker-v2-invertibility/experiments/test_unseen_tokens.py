#!/usr/bin/env python3
"""
Test algebraic decoder on UNSEEN tokens (not in training vocabulary).

This is the critical test that distinguishes:
- "Can identify members of a vocabulary" (lookup table)
- "Representation contains enough structure to reconstruct arbitrary inputs" (true inversion)
"""

import sys
from pathlib import Path
import numpy as np
import random

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from kronecker_encoder import KroneckerEncoderStaged, algebraic_decode_raw_kronecker

def generate_unseen_tokens(seed: int = 99) -> list:
    """Generate tokens that were NOT in the training vocabulary."""
    random.seed(seed)
    np.random.seed(seed)
    
    unseen = []
    
    # 1. Random byte strings (valid UTF-8)
    for _ in range(50):
        length = random.randint(1, 20)
        token = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789 ', k=length))
        unseen.append(token)
    
    # 2. Uncommon words
    uncommon = [
        "xylophone", "quizzical", "zephyr", "fjord", "rhythmic",
        "cryptocurrency", "quantum", "Byzantine", "archipelago",
        "serendipity", "ephemeral", "ubiquitous", "paradigm"
    ]
    unseen.extend(uncommon)
    
    # 3. Very long strings
    unseen.append("a" * 100)
    unseen.append("thequickbrownfoxjumpsoverthelazydog" * 2)
    
    # 4. Numbers
    for i in range(1000, 1100):
        unseen.append(str(i))
    
    # 5. Special patterns
    unseen.extend([
        "____", "....", "----", "####",
        "test123", "ABC123", "abc-def-ghi",
    ])
    
    # 6. Unicode (if supported)
    unseen.extend([
        "münchen", "Москва", "東京", "مرحبا",
        "こんにちは", "🚀🔥💯",
    ])
    
    return unseen

def main():
    print("\n" + "="*70)
    print("UNSEEN TOKEN TEST")
    print("="*70)
    print("\nObjective: Test algebraic decoder on tokens NEVER seen during")
    print("           vocabulary construction. This proves the decoder uses")
    print("           mathematical structure, not memorization.\n")
    
    # Create encoder (same config as Phase 1)
    encoder = KroneckerEncoderStaged(
        char_dim=256,
        pos_dim=32,
        apply_length_norm=True,
        apply_z_norm=False,  # Test on raw Kronecker first
        apply_projection=False,
    )
    
    # Generate unseen tokens
    unseen_tokens = generate_unseen_tokens(seed=99)
    print(f"Generated {len(unseen_tokens)} unseen tokens\n")
    
    # Test decoder
    results = {
        'total': 0,
        'exact_match': 0,
        'byte_match': 0,
        'failed': 0,
        'truncated': 0,
    }
    
    failures = []
    
    for token in unseen_tokens:
        results['total'] += 1
        
        # Encode
        try:
            stages = encoder.encode_all_stages(token)
            kappa = stages['stage2_kronecker']
        except Exception as e:
            failures.append((token, f'encode_failed: {e}'))
            results['failed'] += 1
            continue
        
        # Decode
        reconstructed_bytes = algebraic_decode_raw_kronecker(
            kappa,
            char_dim=encoder.char_dim,
            pos_dim=encoder.pos_dim,
            length_normalized=encoder.apply_length_norm,
        )
        
        # Check
        original_bytes = token.encode('utf-8')
        
        if reconstructed_bytes is None:
            results['failed'] += 1
            failures.append((token, 'decode_returned_none'))
        elif len(original_bytes) > encoder.pos_dim:
            # Token was truncated
            truncated_original = encoder.token_to_bytes(token, max_len=encoder.pos_dim)
            if reconstructed_bytes == truncated_original:
                results['truncated'] += 1
            else:
                results['failed'] += 1
                failures.append((token, f'truncation_mismatch'))
        elif reconstructed_bytes == original_bytes:
            results['exact_match'] += 1
        else:
            results['failed'] += 1
            failures.append((token, f'mismatch: got {reconstructed_bytes.hex()}, expected {original_bytes.hex()}'))
    
    # Report
    print(f"{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}\n")
    print(f"Total unseen tokens: {results['total']}")
    print(f"Exact reconstructions: {results['exact_match']} ({100*results['exact_match']/results['total']:.1f}%)")
    print(f"Truncated (but correct): {results['truncated']} ({100*results['truncated']/results['total']:.1f}%)")
    print(f"Failures: {results['failed']} ({100*results['failed']/results['total']:.1f}%)")
    
    if results['exact_match'] + results['truncated'] == results['total']:
        print(f"\n✅ **PERFECT RECONSTRUCTION ON UNSEEN TOKENS**")
        print(f"   The decoder uses mathematical structure, not memorization!")
    else:
        print(f"\n⚠️  Some failures detected:")
        for token, reason in failures[:10]:
            print(f"  '{token}': {reason}")
    
    # Key insight
    print(f"\n{'='*70}")
    print(f"INTERPRETATION")
    print(f"{'='*70}\n")
    
    success_rate = (results['exact_match'] + results['truncated']) / results['total']
    
    if success_rate >= 0.99:
        print("VERDICT: The algebraic decoder successfully reconstructs tokens")
        print("         it has NEVER seen before. This proves the Kronecker")
        print("         representation contains sufficient mathematical structure")
        print("         for invertible decoding without vocabulary lookup.")
    elif success_rate >= 0.90:
        print("VERDICT: Mostly successful, but some edge cases fail.")
        print("         Further investigation needed.")
    else:
        print("VERDICT: Significant failure rate suggests the decoder")
        print("         may be limited or the representation loses information.")
    
    print()

if __name__ == "__main__":
    main()
