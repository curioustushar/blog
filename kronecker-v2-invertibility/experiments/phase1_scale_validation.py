#!/usr/bin/env python3
"""
Kronecker V2 — Phase 1: Scale Validation

Objective: Determine if Kronecker representation remains uniquely decodable
at 50K-1M tokens, especially after z-normalization and projection.

This experiment answers:
1. Is raw Kronecker injective at scale?
2. Which stage first introduces collisions?
3. Does the complete pipeline remain collision-free?
4. Can we build an algebraic decoder that works on unseen tokens?

Usage:
    python experiments/phase1_scale_validation.py --vocab-size 50000
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from kronecker_encoder import KroneckerEncoderStaged

# ============================================================================
# REPRODUCIBILITY: Log everything
# ============================================================================

def log_environment():
    """Log Python version, numpy version, dtypes, hardware for reproducibility."""
    import platform
    import sys
    
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "numpy_float_default": str(np.float64),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    print("\n" + "="*70)
    print("ENVIRONMENT")
    print("="*70)
    for key, val in env_info.items():
        print(f"  {key}: {val}")
    print()
    
    return env_info


# ============================================================================
# VOCABULARY GENERATION
# ============================================================================

def load_real_vocabulary(name: str = "gpt2", max_tokens: Optional[int] = None) -> List[str]:
    """
    Load real vocabulary from transformers.
    
    Returns empty list if transformers unavailable.
    """
    try:
        from transformers import AutoTokenizer
        print(f"Loading vocabulary: {name}")
        tokenizer = AutoTokenizer.from_pretrained(name)
        
        tokens = []
        for i in range(tokenizer.vocab_size):
            try:
                token = tokenizer.decode([i])
                tokens.append(token)
            except Exception:
                continue
        
        if max_tokens:
            tokens = tokens[:max_tokens]
        
        print(f"✓ Loaded {len(tokens)} tokens from {name}")
        return tokens
    
    except ImportError:
        print("⚠️  transformers not installed")
        return []
    except Exception as e:
        print(f"⚠️  Failed to load {name}: {e}")
        return []


def generate_synthetic_vocabulary(size: int, max_length: int = 32, seed: int = 42) -> List[str]:
    """
    Generate deterministic synthetic vocabulary.
    
    This is NOT random - it's a systematic construction covering:
    - All single bytes
    - Common 2-byte patterns
    - Length variations
    - Repeated characters
    - Edge cases
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    vocab = []
    
    # 1. All single bytes (256 tokens)
    for byte_val in range(256):
        try:
            token = bytes([byte_val]).decode('utf-8', errors='ignore')
            if token and token not in vocab:
                vocab.append(token)
        except:
            pass
    
    # 2. Common 2-byte patterns
    for b1 in range(0, 256, 4):  # Sample every 4th
        for b2 in range(0, 256, 4):
            try:
                token = bytes([b1, b2]).decode('utf-8', errors='ignore')
                if token and len(token) > 0 and token not in vocab:
                    vocab.append(token)
            except:
                pass
    
    # 3. Length variations (repeated 'a')
    for length in [1, 2, 3, 4, 5, 8, 16, 24, 32, 40, 50]:
        vocab.append('a' * length)
    
    # 4. Common words
    common = ["the", "a", "an", "is", "and", "to", "of", "in", "for", "that"]
    vocab.extend(common)
    
    # 5. Fill remaining with random valid UTF-8
    import string
    chars = string.ascii_letters + string.digits + " .,!?"
    while len(vocab) < size:
        length = random.randint(1, min(max_length, 32))
        token = ''.join(random.choices(chars, k=length))
        if token not in vocab:
            vocab.append(token)
    
    # Deduplicate and truncate
    vocab = list(dict.fromkeys(vocab))[:size]
    
    return vocab


def analyze_vocabulary_stats(vocab: List[str]) -> Dict:
    """Compute statistics about vocabulary."""
    lengths = [len(token.encode('utf-8')) for token in vocab]
    
    return {
        'size': len(vocab),
        'max_length': max(lengths),
        'min_length': min(lengths),
        'mean_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'std_length': np.std(lengths),
        'lengths_over_32': sum(1 for l in lengths if l > 32),
        'percent_over_32': 100 * sum(1 for l in lengths if l > 32) / len(vocab),
    }


# ============================================================================
# COLLISION DETECTION (Every Stage)
# ============================================================================

def canonical_representation(arr: np.ndarray, precision: int = 12) -> str:
    """
    Create canonical string representation for exact comparison.
    
    For sparse arrays, stores (index, value) pairs.
    For dense arrays, rounds to precision decimal places.
    """
    # Find non-zero entries
    nonzero_indices = np.nonzero(arr)[0]
    
    if len(nonzero_indices) < len(arr) * 0.1:  # Sparse (< 10% nonzero)
        # Store as sorted (index, value) pairs
        pairs = []
        for idx in nonzero_indices:
            val = np.round(arr[idx], decimals=precision)
            pairs.append((int(idx), float(val)))
        return str(sorted(pairs))
    else:
        # Dense: round entire array
        rounded = np.round(arr, decimals=precision)
        return str(rounded.tobytes())


def detect_collisions_exact(
    tokens: List[str],
    encoder: KroneckerEncoderStaged,
    stage_name: str,
    precision: int = 12,
) -> Tuple[int, List[Tuple[str, str]], Dict]:
    """
    Detect exact collisions using canonical representations.
    
    Returns:
        unique_count: Number of unique representations
        collisions: List of (token_a, token_b) pairs that collide
        stats: Additional statistics
    """
    print(f"\n{'='*70}")
    print(f"Stage: {stage_name}")
    print(f"{'='*70}")
    
    # Encode all tokens
    representations = {}
    canonical_reps = {}
    
    for token in tokens:
        stages = encoder.encode_all_stages(token)
        if stage_name not in stages:
            print(f"⚠️  Stage {stage_name} not found")
            return 0, [], {}
        
        rep = stages[stage_name]
        representations[token] = rep
        canonical_reps[token] = canonical_representation(rep, precision)
    
    # Find duplicates
    rep_to_tokens = defaultdict(list)
    for token, canonical in canonical_reps.items():
        rep_to_tokens[canonical].append(token)
    
    # Extract collision pairs
    collisions = []
    for canonical, token_list in rep_to_tokens.items():
        if len(token_list) > 1:
            # All pairs in this collision group
            for i in range(len(token_list)):
                for j in range(i+1, len(token_list)):
                    collisions.append((token_list[i], token_list[j]))
    
    unique_count = len(rep_to_tokens)
    
    # Compute distance statistics (sample for speed)
    tokens_sample = list(tokens[:min(1000, len(tokens))])
    distances = []
    for i in range(len(tokens_sample)):
        for j in range(i+1, min(i+50, len(tokens_sample))):
            t1, t2 = tokens_sample[i], tokens_sample[j]
            if t1 in representations and t2 in representations:
                dist = np.linalg.norm(representations[t1] - representations[t2])
                if dist > 0:  # Exclude exact duplicates
                    distances.append(dist)
    
    stats = {
        'total_tokens': len(tokens),
        'unique_representations': unique_count,
        'collision_count': len(collisions),
        'collision_groups': len([g for g in rep_to_tokens.values() if len(g) > 1]),
        'min_distance': float(np.min(distances)) if distances else 0.0,
        'median_distance': float(np.median(distances)) if distances else 0.0,
        'mean_distance': float(np.mean(distances)) if distances else 0.0,
    }
    
    # Report
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Unique representations: {stats['unique_representations']}")
    print(f"Collision pairs: {stats['collision_count']}")
    
    if collisions:
        print(f"\n⚠️  COLLISIONS DETECTED:")
        for t1, t2 in collisions[:5]:  # Show first 5
            print(f"  '{t1}' ≡ '{t2}'")
            print(f"    bytes: {t1.encode('utf-8').hex()} vs {t2.encode('utf-8').hex()}")
    else:
        print(f"✅ No collisions")
        print(f"  Min pairwise distance: {stats['min_distance']:.2e}")
        print(f"  Median distance: {stats['median_distance']:.2e}")
    
    return unique_count, collisions, stats


# ============================================================================
# ALGEBRAIC DECODER
# ============================================================================

def algebraic_decode_raw_kronecker(
    kappa: np.ndarray,
    char_dim: int = 256,
    pos_dim: int = 32,
    length_normalized: bool = True,
    tolerance: float = 1e-6,
) -> Optional[bytes]:
    """
    Algebraic decoder: reconstruct byte sequence from raw Kronecker representation.
    
    Algorithm:
    1. Find non-zero positions in κ(b)
    2. Extract (byte_val, position) from each index
    3. Reconstruct byte sequence
    4. Verify consistency
    
    This should run in O(D) time, independent of vocabulary size.
    
    Returns:
        bytes: Reconstructed byte sequence
        None: If decoding fails
    """
    D = char_dim * pos_dim
    
    if len(kappa) != D:
        return None
    
    # Find non-zero entries
    nonzero_indices = np.where(np.abs(kappa) > tolerance)[0]
    
    if len(nonzero_indices) == 0:
        return b''  # Empty token
    
    # Extract byte-position pairs
    byte_position_pairs = []
    for idx in nonzero_indices:
        byte_val = idx // pos_dim
        position = idx % pos_dim
        magnitude = kappa[idx]
        
        byte_position_pairs.append((position, byte_val, magnitude))
    
    # Sort by position
    byte_position_pairs.sort()
    
    # Verify all magnitudes are approximately equal (if length-normalized)
    magnitudes = [mag for _, _, mag in byte_position_pairs]
    if length_normalized and len(magnitudes) > 1:
        expected_magnitude = magnitudes[0]
        if not all(abs(mag - expected_magnitude) < tolerance for mag in magnitudes):
            # Magnitudes don't match - possibly corrupted
            pass  # Continue anyway, might still work
    
    # Verify positions are consecutive (0, 1, 2, ...)
    positions = [pos for pos, _, _ in byte_position_pairs]
    expected_positions = list(range(len(positions)))
    if positions != expected_positions:
        # Non-consecutive positions - possibly corrupted
        return None
    
    # Extract bytes
    byte_sequence = bytes([byte_val for _, byte_val, _ in byte_position_pairs])
    
    return byte_sequence


def test_algebraic_decoder(
    encoder: KroneckerEncoderStaged,
    test_tokens: List[str],
) -> Dict:
    """
    Test algebraic decoder on a set of tokens.
    
    Returns reconstruction statistics.
    """
    print(f"\n{'='*70}")
    print(f"Testing Algebraic Decoder")
    print(f"{'='*70}")
    
    results = {
        'total': len(test_tokens),
        'exact_match': 0,
        'byte_match': 0,
        'failed': 0,
        'decode_times': [],
    }
    
    failures = []
    
    for token in test_tokens:
        # Encode
        stages = encoder.encode_all_stages(token)
        kappa = stages['stage2_kronecker']
        
        # Decode
        start = time.time()
        reconstructed_bytes = algebraic_decode_raw_kronecker(
            kappa,
            char_dim=encoder.char_dim,
            pos_dim=encoder.pos_dim,
            length_normalized=encoder.apply_length_norm,
        )
        decode_time = time.time() - start
        results['decode_times'].append(decode_time)
        
        # Check
        original_bytes = token.encode('utf-8')[:encoder.pos_dim]
        
        if reconstructed_bytes is None:
            results['failed'] += 1
            failures.append((token, 'decode_failed'))
        elif reconstructed_bytes == original_bytes:
            results['exact_match'] += 1
        elif reconstructed_bytes[:min(len(reconstructed_bytes), len(original_bytes))] == original_bytes[:min(len(reconstructed_bytes), len(original_bytes))]:
            results['byte_match'] += 1
        else:
            results['failed'] += 1
            failures.append((token, f'mismatch: {reconstructed_bytes.hex()} vs {original_bytes.hex()}'))
    
    # Stats
    results['mean_decode_time'] = np.mean(results['decode_times'])
    results['median_decode_time'] = np.median(results['decode_times'])
    
    # Report
    print(f"Total tokens: {results['total']}")
    print(f"Exact reconstructions: {results['exact_match']} ({100*results['exact_match']/results['total']:.1f}%)")
    print(f"Partial matches: {results['byte_match']}")
    print(f"Failures: {results['failed']}")
    print(f"Mean decode time: {results['mean_decode_time']*1e6:.1f} μs")
    
    if failures:
        print(f"\nFirst 5 failures:")
        for token, reason in failures[:5]:
            print(f"  '{token}': {reason}")
    
    return results


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Scale Validation")
    parser.add_argument('--vocab-size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--vocab-name', type=str, default='synthetic', help='Vocabulary source')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--precision', type=int, default=12, help='Decimal precision for collision detection')
    parser.add_argument('--output-dir', type=str, default='results/phase1', help='Output directory')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("KRONECKER V2 — PHASE 1: SCALE VALIDATION")
    print("="*70)
    print(f"\nObjective: Test Kronecker invertibility at {args.vocab_size} tokens")
    print(f"Vocabulary: {args.vocab_name}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Log environment
    env_info = log_environment()
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load vocabulary
    print(f"Loading vocabulary ({args.vocab_size} tokens)...")
    if args.vocab_name == 'synthetic':
        vocab = generate_synthetic_vocabulary(args.vocab_size, seed=args.seed)
    else:
        vocab = load_real_vocabulary(args.vocab_name, max_tokens=args.vocab_size)
        if not vocab:
            print("Falling back to synthetic vocabulary")
            vocab = generate_synthetic_vocabulary(args.vocab_size, seed=args.seed)
    
    vocab_stats = analyze_vocabulary_stats(vocab)
    print(f"\nVocabulary Statistics:")
    for key, val in vocab_stats.items():
        print(f"  {key}: {val}")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Raw Kronecker',
            'apply_length_norm': True,
            'apply_z_norm': False,
            'apply_projection': False,
            'stages': ['stage2_kronecker'],
        },
        {
            'name': 'With Z-Normalization',
            'apply_length_norm': True,
            'apply_z_norm': True,
            'apply_projection': False,
            'stages': ['stage2_kronecker', 'stage3_znorm'],
        },
        {
            'name': 'With Projection',
            'apply_length_norm': True,
            'apply_z_norm': True,
            'apply_projection': True,
            'd_model': 768,
            'stages': ['stage2_kronecker', 'stage3_znorm', 'stage4_projection'],
        },
    ]
    
    all_results = {}
    
    for config in test_configs:
        print(f"\n{'#'*70}")
        print(f"# TEST: {config['name']}")
        print(f"{'#'*70}")
        
        # Create encoder
        encoder = KroneckerEncoderStaged(
            char_dim=256,
            pos_dim=32,
            d_model=config.get('d_model'),
            apply_length_norm=config['apply_length_norm'],
            apply_z_norm=config['apply_z_norm'],
            apply_projection=config['apply_projection'],
            projection_seed=args.seed,
        )
        
        # Test each stage
        config_results = {}
        for stage_name in config['stages']:
            unique, collisions, stats = detect_collisions_exact(
                vocab, encoder, stage_name, precision=args.precision
            )
            config_results[stage_name] = {
                'unique_count': unique,
                'collision_pairs': collisions[:10],  # Save first 10
                'stats': stats,
            }
        
        # Test algebraic decoder (only for raw Kronecker)
        if config['name'] == 'Raw Kronecker':
            decoder_results = test_algebraic_decoder(encoder, vocab[:1000])  # Test on subset
            config_results['decoder'] = decoder_results
        
        all_results[config['name']] = config_results
    
    # Save results
    output_file = output_dir / f"scale_test_{args.vocab_size}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'vocab_size': args.vocab_size,
            'vocab_name': args.vocab_name,
            'seed': args.seed,
            'vocab_stats': vocab_stats,
            'env_info': env_info,
            'results': all_results,
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}\n")
    print(f"{'Stage':<30} {'Unique':<10} {'Collisions':<12} {'Min Distance':<15}")
    print(f"{'-'*70}")
    
    for config_name, config_result in all_results.items():
        for stage_name, stage_result in config_result.items():
            if stage_name == 'decoder':
                continue
            stats = stage_result['stats']
            print(f"{stage_name:<30} {stats['unique_representations']:<10} {stats['collision_count']:<12} {stats['min_distance']:<15.2e}")
    
    print(f"\n{'='*70}")
    print("PHASE 1 TEST COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
