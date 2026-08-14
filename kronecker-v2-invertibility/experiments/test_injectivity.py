#!/usr/bin/env python3
"""
Kronecker V2 — Phase 0: Injectivity Falsification Experiment

Objective: Determine if Kronecker representation is injective and where information loss occurs.

Usage:
    python experiments/test_injectivity.py
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kronecker_encoder import (
    KroneckerEncoderStaged,
    find_collisions,
    compute_min_pairwise_distance,
)


def load_vocabulary(vocab_name: str = "gpt2", max_tokens: Optional[int] = None) -> List[str]:
    """
    Load a real vocabulary.
    
    Parameters
    ----------
    vocab_name : str
        Tokenizer name (e.g., 'gpt2', 'gpt2-large')
    max_tokens : int, optional
        Limit vocabulary size for faster testing
    
    Returns
    -------
    tokens : list of str
        Vocabulary tokens
    """
    try:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            print("⚠️  transformers not installed. Using synthetic vocabulary.")
            return generate_synthetic_vocabulary(max_tokens or 10000)
        
        tokenizer = AutoTokenizer.from_pretrained(vocab_name)
        
        # Get all tokens
        vocab_size = tokenizer.vocab_size
        tokens = []
        
        for i in range(vocab_size):
            try:
                token = tokenizer.decode([i])
                tokens.append(token)
            except Exception:
                # Some token IDs might not decode cleanly
                continue
        
        if max_tokens is not None:
            tokens = tokens[:max_tokens]
        
        print(f"✓ Loaded {len(tokens)} tokens from {vocab_name}")
        return tokens
    
    except Exception as e:
        print(f"⚠️  Failed to load {vocab_name}: {e}")
        print(f"Falling back to synthetic vocabulary")
        return generate_synthetic_vocabulary(max_tokens or 10000)


def generate_synthetic_vocabulary(size: int = 10000) -> List[str]:
    """Generate synthetic test vocabulary."""
    import random
    import string
    
    vocab = []
    
    # Add common words
    common = ["the", "a", "an", "is", "are", "was", "were", "be", "been", "being"]
    vocab.extend(common)
    
    # Add random strings
    for _ in range(size - len(common)):
        length = random.randint(1, 20)
        token = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
        vocab.append(token)
    
    return vocab


def generate_adversarial_vocabulary() -> List[str]:
    """Generate adversarial test cases designed to find collisions."""
    adversarial = []
    
    # Repeated characters
    for char in ['a', 'b', '0', '1', 'z']:
        for length in [1, 2, 3, 4, 5, 8, 16, 32, 64]:
            adversarial.append(char * length)
    
    # Patterns
    patterns = [
        "abc" * 10,
        "ab" * 15,
        "0123456789" * 3,
        "a" + "b" * 31,
        "prefix",
        "prefix1",
        "prefix2",
        "prefixA",
        "test",
        "Test",
        "TEST",
        " test",
        "test ",
        ".test",
    ]
    adversarial.extend(patterns)
    
    # Permutations
    for perm in ["abc", "acb", "bac", "bca", "cab", "cba"]:
        adversarial.append(perm)
    
    # Unicode
    adversarial.extend(["hello", "héllo", "hëllo", "世界", "🔥", "café"])
    
    return adversarial


def run_collision_analysis(
    tokens: List[str],
    encoder: KroneckerEncoderStaged,
    stages_to_test: List[str],
) -> Dict:
    """
    Run collision detection at each pipeline stage.
    
    Returns
    -------
    results : dict
        Summary of collisions at each stage
    """
    results = {
        'vocab_size': len(tokens),
        'stages': {},
    }
    
    for stage_name in stages_to_test:
        print(f"\n{'='*70}")
        print(f"STAGE: {stage_name}")
        print(f"{'='*70}")
        
        unique_count, collisions = find_collisions(
            tokens, encoder, stage_name, tolerance=1e-10
        )
        
        min_dist = compute_min_pairwise_distance(
            tokens, encoder, stage_name, sample_size=min(1000, len(tokens))
        )
        
        results['stages'][stage_name] = {
            'unique_count': unique_count,
            'collision_count': len(collisions),
            'collisions': [
                {
                    'token_a': t1,
                    'token_b': t2,
                    'distance': float(dist),
                    'bytes_a': t1.encode('utf-8').hex(),
                    'bytes_b': t2.encode('utf-8').hex(),
                }
                for t1, t2, dist in collisions[:10]  # Save first 10
            ],
            'min_distance': float(min_dist),
        }
    
    return results


def analyze_projection_properties(encoder: KroneckerEncoderStaged, vocab: List[str]) -> Dict:
    """
    Analyze mathematical properties of the projection matrix (if it exists).
    """
    if not encoder.apply_projection or encoder.projection_matrix is None:
        return {'projection_applied': False}
    
    W = encoder.projection_matrix
    D = W.shape[1]
    d = W.shape[0]
    
    # Compute rank
    rank = np.linalg.matrix_rank(W)
    nullity = D - rank
    
    # Compute singular values
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    condition_number = s[0] / s[-1] if s[-1] > 1e-10 else float('inf')
    
    # Check if it separates vocabulary
    # Sample some tokens and check if they're separated after projection
    sample_size = min(1000, len(vocab))
    sample_tokens = vocab[:sample_size]
    
    # Encode with and without projection
    encoder_no_proj = KroneckerEncoderStaged(
        char_dim=encoder.char_dim,
        pos_dim=encoder.pos_dim,
        apply_length_norm=encoder.apply_length_norm,
        apply_z_norm=encoder.apply_z_norm,
        apply_projection=False,
    )
    
    min_dist_before = float('inf')
    min_dist_after = float('inf')
    
    for i in range(min(100, len(sample_tokens))):
        for j in range(i+1, min(i+10, len(sample_tokens))):
            t1, t2 = sample_tokens[i], sample_tokens[j]
            
            # Before projection
            r1_before = encoder_no_proj.encode(t1)
            r2_before = encoder_no_proj.encode(t2)
            dist_before = np.linalg.norm(r1_before - r2_before)
            min_dist_before = min(min_dist_before, dist_before)
            
            # After projection
            r1_after = encoder.encode(t1)
            r2_after = encoder.encode(t2)
            dist_after = np.linalg.norm(r1_after - r2_after)
            min_dist_after = min(min_dist_after, dist_after)
    
    return {
        'projection_applied': True,
        'input_dim': D,
        'output_dim': d,
        'rank': int(rank),
        'nullity': int(nullity),
        'condition_number': float(condition_number),
        'min_singular_value': float(s[-1]),
        'max_singular_value': float(s[0]),
        'min_distance_before_projection': float(min_dist_before),
        'min_distance_after_projection': float(min_dist_after),
        'distance_ratio': float(min_dist_after / min_dist_before) if min_dist_before > 0 else 0,
    }


def determine_verdict(results: Dict) -> str:
    """
    Produce a clear scientific verdict based on results.
    
    Returns one of:
    - VERDICT A: Collision Found
    - VERDICT B: No Collision Found
    - VERDICT C: Representation Is Provably Injective
    - VERDICT D: Experiment Inconclusive
    """
    # Check if any stage has collisions
    has_collision = False
    collision_stage = None
    
    for stage_name, stage_result in results['stages'].items():
        if stage_result['collision_count'] > 0:
            has_collision = True
            collision_stage = stage_name
            break
    
    if has_collision:
        verdict = "VERDICT A — Collision Found"
        explanation = f"Collisions detected at {collision_stage}"
        return verdict, explanation
    
    # No collisions found — but distinguish between empirical observation and proof
    vocab_size = results['vocab_size']
    
    # Check all stages tested
    all_unique = all(
        stage_result['unique_count'] == vocab_size
        for stage_result in results['stages'].values()
    )
    
    if all_unique:
        verdict = "VERDICT B — No Collision Found (Empirical)"
        explanation = (
            f"No collisions detected across {vocab_size} tokens at any pipeline stage. "
            f"This is evidence of injectivity over this finite vocabulary, but NOT a mathematical proof. "
            f"Collisions may exist for tokens outside this test set."
        )
        return verdict, explanation
    
    verdict = "VERDICT D — Experiment Inconclusive"
    explanation = "Results are ambiguous; manual inspection required."
    return verdict, explanation


def main():
    """Main experiment entry point."""
    print("\n" + "="*70)
    print("KRONECKER V2 — PHASE 0: INJECTIVITY FALSIFICATION EXPERIMENT")
    print("="*70)
    print("\nObjective: Determine if Kronecker representation is injective")
    print("           and identify where information loss occurs.\n")
    
    # Create results directory
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Configuration
    VOCAB_NAME = "gpt2"
    MAX_TOKENS = 1000  # Start with 1000 tokens for fast initial test
    
    print(f"Configuration:")
    print(f"  Vocabulary: {VOCAB_NAME}")
    print(f"  Max tokens: {MAX_TOKENS or 'all'}")
    print(f"  char_dim: 256")
    print(f"  pos_dim: 32")
    print()
    
    # Load vocabulary
    print("Loading vocabulary...")
    vocab = load_vocabulary(VOCAB_NAME, MAX_TOKENS)
    
    # Add adversarial test cases
    print("Adding adversarial test cases...")
    adversarial = generate_adversarial_vocabulary()
    vocab.extend(adversarial)
    vocab = list(set(vocab))  # Remove duplicates
    print(f"Total vocabulary size: {len(vocab)}")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Stage 2: Raw Kronecker (no normalization)',
            'apply_length_norm': True,
            'apply_z_norm': False,
            'apply_projection': False,
            'stages': ['stage2_kronecker'],
        },
        {
            'name': 'Stage 3: With Z-normalization',
            'apply_length_norm': True,
            'apply_z_norm': True,
            'apply_projection': False,
            'stages': ['stage2_kronecker', 'stage3_znorm'],
        },
        {
            'name': 'Stage 4: With Projection (D=8192 → d=768)',
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
        )
        
        # Run collision analysis
        start_time = time.time()
        results = run_collision_analysis(vocab, encoder, config['stages'])
        elapsed = time.time() - start_time
        
        results['config'] = config
        results['elapsed_seconds'] = elapsed
        
        # Analyze projection if applicable
        if config['apply_projection']:
            proj_analysis = analyze_projection_properties(encoder, vocab)
            results['projection_analysis'] = proj_analysis
            
            print(f"\nProjection Analysis:")
            print(f"  Input dim (D): {proj_analysis.get('input_dim', 'N/A')}")
            print(f"  Output dim (d): {proj_analysis.get('output_dim', 'N/A')}")
            print(f"  Rank: {proj_analysis.get('rank', 'N/A')}")
            print(f"  Nullity: {proj_analysis.get('nullity', 'N/A')}")
            print(f"  Condition number: {proj_analysis.get('condition_number', 'N/A'):.2e}")
            print(f"  Distance preserved: {proj_analysis.get('distance_ratio', 0)*100:.1f}%")
        
        all_results[config['name']] = results
        
        # Save intermediate results
        output_file = results_dir / f"collision_report_{config['name'].replace(' ', '_').replace(':', '')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")
    
    # Determine final verdict
    print(f"\n{'='*70}")
    print("FINAL VERDICT")
    print(f"{'='*70}\n")
    
    # Examine most important configuration (raw Kronecker)
    raw_kronecker_results = all_results['Stage 2: Raw Kronecker (no normalization)']
    verdict, explanation = determine_verdict(raw_kronecker_results)
    
    print(f"{verdict}\n")
    print(f"{explanation}\n")
    
    # Summary table
    print(f"\nSummary Table:")
    print(f"{'Stage':<40} {'Dimension':<12} {'Unique':<10} {'Collisions':<12} {'Min Distance':<15}")
    print(f"{'-'*90}")
    
    for config_name, results in all_results.items():
        for stage_name, stage_result in results['stages'].items():
            # Get dimension
            if stage_name == 'stage2_kronecker' or stage_name == 'stage3_znorm':
                dim = 256 * 32
            elif stage_name == 'stage4_projection':
                dim = results['config'].get('d_model', 'N/A')
            else:
                dim = 'N/A'
            
            unique = stage_result['unique_count']
            collisions = stage_result['collision_count']
            min_dist = stage_result['min_distance']
            
            print(f"{stage_name:<40} {str(dim):<12} {unique:<10} {collisions:<12} {min_dist:<15.2e}")
    
    # Save final summary
    summary = {
        'verdict': verdict,
        'explanation': explanation,
        'vocab_size': len(vocab),
        'test_configs': list(all_results.keys()),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    summary_file = results_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to {summary_file}")
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    from typing import Optional
    main()
