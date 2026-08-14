"""
Diagnose why Kronecker decoder is failing.

Check:
1. Are predictions mostly zeros?
2. Are targets mostly zeros?
3. Is the decoder working on ground-truth κ?
4. What does the predicted κ look like?
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from kronecker_encoder import KroneckerEncoderStaged, algebraic_decode_raw_kronecker


def test_decoder_on_ground_truth():
    """Test if the decoder works on ground-truth Kronecker representations."""
    print("=" * 80)
    print("TEST 1: Decoder on Ground-Truth Kronecker")
    print("=" * 80)
    
    encoder = KroneckerEncoderStaged(
        char_dim=256,
        pos_dim=32,
        apply_length_norm=True,
        apply_z_norm=False,
        apply_projection=False,
    )
    
    test_tokens = ["the", "a", "test", "hello", "world", "word0", "word1", "word2"]
    
    correct = 0
    for token in test_tokens:
        # Encode
        stages = encoder.encode_all_stages(token)
        kappa = stages['stage2_kronecker']
        
        # Decode
        token_bytes = algebraic_decode_raw_kronecker(
            kappa,
            char_dim=256,
            pos_dim=32,
            length_normalized=True,
        )
        
        if token_bytes is not None:
            try:
                decoded = token_bytes.decode('utf-8', errors='replace')
                match = (decoded == token)
                print(f"  {token:10s} → κ → {decoded:10s}  {'✓' if match else '✗'}")
                if match:
                    correct += 1
            except:
                print(f"  {token:10s} → κ → DECODE ERROR")
        else:
            print(f"  {token:10s} → κ → None")
    
    print(f"\nAccuracy: {correct}/{len(test_tokens)} = {correct/len(test_tokens)*100:.1f}%")
    
    return correct == len(test_tokens)


def analyze_kappa_properties():
    """Analyze properties of Kronecker representations."""
    print("\n" + "=" * 80)
    print("TEST 2: Analyze Kronecker Properties")
    print("=" * 80)
    
    encoder = KroneckerEncoderStaged(
        char_dim=256,
        pos_dim=32,
        apply_length_norm=True,
        apply_z_norm=False,
        apply_projection=False,
    )
    
    test_tokens = ["the", "a", "test", "hello", "world", "x", "testing"]
    
    for token in test_tokens:
        stages = encoder.encode_all_stages(token)
        kappa = stages['stage2_kronecker']
        
        nnz = np.sum(np.abs(kappa) > 1e-6)
        magnitude = np.linalg.norm(kappa)
        max_val = np.max(np.abs(kappa))
        
        print(f"  {token:10s}: nnz={nnz:3d}, ||κ||={magnitude:.4f}, max={max_val:.4f}")


def test_decoder_on_noisy_kappa():
    """Test decoder robustness to noise."""
    print("\n" + "=" * 80)
    print("TEST 3: Decoder Robustness to Noise")
    print("=" * 80)
    
    encoder = KroneckerEncoderStaged(
        char_dim=256,
        pos_dim=32,
        apply_length_norm=True,
        apply_z_norm=False,
        apply_projection=False,
    )
    
    test_tokens = ["the", "hello", "test"]
    noise_levels = [0.0, 0.001, 0.01, 0.1, 0.5]
    
    print(f"\n{'Token':10s} | ", end="")
    for noise in noise_levels:
        print(f"σ={noise:5.3f} | ", end="")
    print()
    print("-" * 80)
    
    for token in test_tokens:
        stages = encoder.encode_all_stages(token)
        kappa = stages['stage2_kronecker']
        
        print(f"{token:10s} | ", end="")
        
        for noise in noise_levels:
            # Add noise
            kappa_noisy = kappa + np.random.randn(len(kappa)) * noise
            
            # Decode
            token_bytes = algebraic_decode_raw_kronecker(
                kappa_noisy,
                char_dim=256,
                pos_dim=32,
                length_normalized=True,
            )
            
            if token_bytes is not None:
                try:
                    decoded = token_bytes.decode('utf-8', errors='replace')
                    match = (decoded == token)
                    print(f"{'✓' if match else '✗':^9s} | ", end="")
                except:
                    print(f"{'ERR':^9s} | ", end="")
            else:
                print(f"{'None':^9s} | ", end="")
        
        print()


def test_random_predictions():
    """Test what happens when we feed random vectors to the decoder."""
    print("\n" + "=" * 80)
    print("TEST 4: Decoder on Random Vectors")
    print("=" * 80)
    
    D = 256 * 32
    n_tests = 10
    
    success = 0
    for i in range(n_tests):
        # Random κ-like vector
        kappa_random = np.random.randn(D) * 0.1
        
        token_bytes = algebraic_decode_raw_kronecker(
            kappa_random,
            char_dim=256,
            pos_dim=32,
            length_normalized=True,
        )
        
        if token_bytes is not None:
            try:
                decoded = token_bytes.decode('utf-8', errors='replace')
                print(f"  Random {i}: decoded as '{decoded}' ({'success' if decoded else 'empty'})")
                if decoded:
                    success += 1
            except:
                print(f"  Random {i}: decode error")
        else:
            print(f"  Random {i}: None")
    
    print(f"\nRandom vectors decoded: {success}/{n_tests}")


def test_learned_predictions():
    """
    Simulate what a trained model might predict.
    
    If the model learns to predict mostly zeros (which gives low MSE since κ is sparse),
    the decoder will fail.
    """
    print("\n" + "=" * 80)
    print("TEST 5: Decoder on Near-Zero Predictions")
    print("=" * 80)
    
    encoder = KroneckerEncoderStaged(
        char_dim=256,
        pos_dim=32,
        apply_length_norm=True,
        apply_z_norm=False,
        apply_projection=False,
    )
    
    test_tokens = ["the", "hello", "test"]
    
    for token in test_tokens:
        stages = encoder.encode_all_stages(token)
        kappa_true = stages['stage2_kronecker']
        
        # Simulate "learned" prediction: mostly zeros with small random noise
        kappa_pred = np.random.randn(len(kappa_true)) * 0.01
        
        # MSE
        mse = np.mean((kappa_pred - kappa_true) ** 2)
        
        # Try to decode
        token_bytes = algebraic_decode_raw_kronecker(
            kappa_pred,
            char_dim=256,
            pos_dim=32,
            length_normalized=True,
        )
        
        if token_bytes is not None:
            try:
                decoded = token_bytes.decode('utf-8', errors='replace')
            except:
                decoded = "ERROR"
        else:
            decoded = "None"
        
        print(f"  {token:10s}: MSE={mse:.6f}, decoded='{decoded}'")
    
    print("\n  💡 This demonstrates the problem:")
    print("      Low MSE does NOT mean the decoder will work!")
    print("      The model might be learning to predict near-zero vectors.")


if __name__ == "__main__":
    print("DIAGNOSING KRONECKER DECODER FAILURE\n")
    
    # Test 1: Basic decoder sanity check
    decoder_works = test_decoder_on_ground_truth()
    
    if not decoder_works:
        print("\n⚠️  CRITICAL: Decoder fails even on ground-truth κ!")
        print("    This is a bug in the decoder or encoder.")
    else:
        print("\n✓ Decoder works correctly on ground-truth κ")
    
    # Test 2: Analyze κ properties
    analyze_kappa_properties()
    
    # Test 3: Noise robustness
    test_decoder_on_noisy_kappa()
    
    # Test 4: Random vectors
    test_random_predictions()
    
    # Test 5: Near-zero predictions
    test_learned_predictions()
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
