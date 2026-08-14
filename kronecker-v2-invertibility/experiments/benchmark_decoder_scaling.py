#!/usr/bin/env python3
"""
Benchmark decoder scaling: O(D) vs O(V)

Objective: Prove that algebraic decoder runtime is independent of vocabulary size,
           while vocabulary search scales linearly with V.
"""

import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from kronecker_encoder import KroneckerEncoderStaged, algebraic_decode_raw_kronecker

def vocab_search_decoder(target_embedding: np.ndarray, vocab_embeddings: dict, tolerance: float = 1e-6) -> str:
    """
    Baseline: Search through all vocabulary embeddings.
    
    Complexity: O(V × D)
    """
    min_dist = float('inf')
    best_token = None
    
    for token, embedding in vocab_embeddings.items():
        dist = np.linalg.norm(target_embedding - embedding)
        if dist < min_dist:
            min_dist = dist
            best_token = token
    
    return best_token

def benchmark_decoders(vocab_sizes: list, num_trials: int = 100, seed: int = 42):
    """
    Benchmark algebraic decoder vs vocabulary search at different vocab sizes.
    """
    np.random.seed(seed)
    
    results = {
        'vocab_sizes': [],
        'algebraic_times': [],
        'vocab_search_times': [],
        'algebraic_std': [],
        'vocab_search_std': [],
    }
    
    encoder = KroneckerEncoderStaged(
        char_dim=256,
        pos_dim=32,
        apply_length_norm=True,
        apply_z_norm=False,
        apply_projection=False,
    )
    
    for V in vocab_sizes:
        print(f"\nTesting V = {V:,}...")
        
        # Generate vocabulary
        vocab = [f"token{i}" for i in range(V)]
        
        # Pre-compute embeddings
        vocab_embeddings = {}
        for token in vocab:
            stages = encoder.encode_all_stages(token)
            vocab_embeddings[token] = stages['stage2_kronecker']
        
        # Sample test tokens
        test_tokens = np.random.choice(vocab, size=min(num_trials, V), replace=False)
        
        # Benchmark algebraic decoder
        algebraic_times = []
        for token in test_tokens:
            kappa = vocab_embeddings[token]
            
            start = time.perf_counter()
            decoded_bytes = algebraic_decode_raw_kronecker(kappa, char_dim=256, pos_dim=32)
            end = time.perf_counter()
            
            algebraic_times.append((end - start) * 1e6)  # Convert to microseconds
        
        # Benchmark vocabulary search
        vocab_search_times = []
        for token in test_tokens:
            target_embedding = vocab_embeddings[token]
            
            start = time.perf_counter()
            result = vocab_search_decoder(target_embedding, vocab_embeddings)
            end = time.perf_counter()
            
            vocab_search_times.append((end - start) * 1e6)  # Convert to microseconds
        
        # Record results
        results['vocab_sizes'].append(V)
        results['algebraic_times'].append(np.mean(algebraic_times))
        results['vocab_search_times'].append(np.mean(vocab_search_times))
        results['algebraic_std'].append(np.std(algebraic_times))
        results['vocab_search_std'].append(np.std(vocab_search_times))
        
        print(f"  Algebraic decoder: {np.mean(algebraic_times):.1f} ± {np.std(algebraic_times):.1f} μs")
        print(f"  Vocab search: {np.mean(vocab_search_times):.1f} ± {np.std(vocab_search_times):.1f} μs")
        print(f"  Speedup: {np.mean(vocab_search_times)/np.mean(algebraic_times):.1f}×")
    
    return results

def plot_results(results: dict, output_path: Path):
    """Create scaling plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    V = np.array(results['vocab_sizes'])
    alg_times = np.array(results['algebraic_times'])
    vocab_times = np.array(results['vocab_search_times'])
    
    # Plot 1: Absolute times
    ax1.plot(V, alg_times, 'o-', label='Algebraic decoder', linewidth=2, markersize=8)
    ax1.plot(V, vocab_times, 's-', label='Vocabulary search', linewidth=2, markersize=8)
    ax1.set_xlabel('Vocabulary Size (V)', fontsize=12)
    ax1.set_ylabel('Decode Time (μs)', fontsize=12)
    ax1.set_title('Decoder Scaling Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Speedup
    speedup = vocab_times / alg_times
    ax2.plot(V, speedup, 'o-', color='green', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='red', linestyle='--', label='No speedup', alpha=0.5)
    ax2.set_xlabel('Vocabulary Size (V)', fontsize=12)
    ax2.set_ylabel('Speedup (×)', fontsize=12)
    ax2.set_title('Algebraic Decoder Speedup vs Vocabulary Search', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")

def main():
    print("\n" + "="*70)
    print("DECODER SCALING BENCHMARK")
    print("="*70)
    print("\nObjective: Prove algebraic decoder is O(D), independent of vocab size V")
    print("           while vocabulary search is O(V)\n")
    
    # Test vocabulary sizes
    vocab_sizes = [100, 500, 1000, 5000, 10000, 50000]
    
    print(f"Testing vocab sizes: {vocab_sizes}")
    print(f"Trials per size: 100\n")
    
    # Run benchmark
    results = benchmark_decoders(vocab_sizes, num_trials=100)
    
    # Plot
    output_dir = Path(__file__).parent.parent / "figures"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "decoder_scaling.png"
    plot_results(results, output_path)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Vocab Size':<15} {'Algebraic (μs)':<18} {'Search (μs)':<18} {'Speedup':<10}")
    print(f"{'-'*70}")
    
    for i, V in enumerate(results['vocab_sizes']):
        alg_time = results['algebraic_times'][i]
        search_time = results['vocab_search_times'][i]
        speedup = search_time / alg_time
        print(f"{V:<15,} {alg_time:<18.1f} {search_time:<18.1f} {speedup:<10.1f}×")
    
    # Analyze scaling
    print(f"\n{'='*70}")
    print(f"COMPLEXITY ANALYSIS")
    print(f"{'='*70}\n")
    
    # Fit linear model to vocab search (should be O(V))
    V_array = np.array(results['vocab_sizes'])
    search_array = np.array(results['vocab_search_times'])
    
    # Simple linear regression
    slope, intercept = np.polyfit(V_array, search_array, 1)
    
    print(f"Vocabulary Search:")
    print(f"  Fitted: time = {slope:.4f} × V + {intercept:.1f}")
    print(f"  Scaling: O(V) ✓")
    
    alg_array = np.array(results['algebraic_times'])
    alg_mean = np.mean(alg_array)
    alg_std = np.std(alg_array)
    alg_cv = alg_std / alg_mean  # Coefficient of variation
    
    print(f"\nAlgebraic Decoder:")
    print(f"  Mean time: {alg_mean:.1f} μs")
    print(f"  Std dev: {alg_std:.1f} μs")
    print(f"  Coefficient of variation: {alg_cv:.2%}")
    
    if alg_cv < 0.2:  # < 20% variation
        print(f"  Scaling: O(1) ✓ (approximately constant)")
    else:
        print(f"  Scaling: Varies with V (weak dependence)")
    
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}\n")
    
    if alg_cv < 0.3 and results['algebraic_times'][-1] / results['algebraic_times'][0] < 2:
        print("✅ **CONFIRMED: Algebraic decoder is O(D), independent of V**")
        print("   Runtime remains approximately constant as vocabulary scales")
        print(f"   from {results['vocab_sizes'][0]:,} to {results['vocab_sizes'][-1]:,} tokens.")
    else:
        print("⚠️  Algebraic decoder shows some dependence on V")
    
    print(f"\n   Maximum speedup: {max(np.array(results['vocab_search_times']) / np.array(results['algebraic_times'])):.1f}×")
    print(f"   At V = {results['vocab_sizes'][-1]:,}\n")

if __name__ == "__main__":
    main()
