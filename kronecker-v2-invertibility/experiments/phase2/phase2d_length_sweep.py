"""
Phase 2D: Fixed-Length Scaling Sweep

Test L ∈ {1, 2, 4, 8, 16, 32}
Keep everything else fixed.
Measure byte accuracy, exact-sequence accuracy, mean error, convergence.
"""

import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from kronecker_encoder import algebraic_decode_raw_kronecker


def construct_kappa(byte_seq: bytes) -> np.ndarray:
    """Exact Kronecker construction."""
    L = len(byte_seq)
    kappa = np.zeros(8192, dtype=np.float64)
    for p in range(L):
        idx = byte_seq[p] * 32 + p
        kappa[idx] = 1.0
    if L > 0:
        kappa /= np.sqrt(L)
    return kappa


def create_deterministic_task(n_samples, L, n_tokens, seed=42):
    """
    Deterministic task: token_id → L bytes
    n_tokens = vocabulary size
    """
    np.random.seed(seed)
    
    X = np.zeros((n_samples, n_tokens))
    y = np.zeros((n_samples, L), dtype=np.uint8)
    
    for i in range(n_samples):
        token_id = i % n_tokens
        X[i, token_id] = 1.0
        
        # Deterministic bytes from token_id
        for pos in range(L):
            byte_val = (token_id * 7 + pos * 13) % 256
            y[i, pos] = byte_val
    
    return X, y


class SimpleBytePredictorLinear:
    """Simple linear model: X @ W → logits"""
    
    def __init__(self, n_tokens: int, L: int):
        self.n_tokens = n_tokens
        self.L = L
        
        # Linear: (n_tokens, L*256)
        scale = np.sqrt(2.0 / n_tokens)
        self.W = np.random.randn(n_tokens, L * 256) * scale
    
    def forward(self, X):
        """X: (B, n_tokens) → logits: (B, L, 256)"""
        logits = X @ self.W
        return logits.reshape(-1, self.L, 256)
    
    def predict(self, X):
        """X: (B, n_tokens) → pred: (B, L)"""
        logits = self.forward(X)
        return np.argmax(logits, axis=-1).astype(np.uint8)
    
    def train_step(self, X, y, lr):
        """
        X: (B, n_tokens)
        y: (B, L)
        Returns: loss
        """
        B = X.shape[0]
        
        # Forward
        logits = self.forward(X)  # (B, L, 256)
        
        # Softmax
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        
        # Loss
        total_loss = 0
        for pos in range(self.L):
            target_probs = probs[np.arange(B), pos, y[:, pos]]
            target_probs = np.clip(target_probs, 1e-10, 1.0)
            total_loss += -np.sum(np.log(target_probs))
        total_loss /= (B * self.L)
        
        # Gradient
        grad_logits = probs.copy()
        for b in range(B):
            for pos in range(self.L):
                grad_logits[b, pos, y[b, pos]] -= 1
        grad_logits /= (B * self.L)
        
        # Update
        grad_logits_flat = grad_logits.reshape(B, -1)
        grad_W = X.T @ grad_logits_flat
        grad_W = np.clip(grad_W, -1.0, 1.0)
        
        self.W -= lr * grad_W
        
        return total_loss


def train_and_evaluate(L, n_tokens=100, n_train=1000, n_test=200, max_epochs=2000, lr=0.1, verbose=False):
    """
    Train and evaluate for a specific length L.
    
    Returns: dict with metrics
    """
    # Create data
    X_train, y_train = create_deterministic_task(n_train, L, n_tokens, seed=42)
    X_test, y_test = create_deterministic_task(n_test, L, n_tokens, seed=999)
    
    # Create model
    model = SimpleBytePredictorLinear(n_tokens, L)
    
    # Training
    start_time = time.time()
    converged_epoch = None
    best_exact_acc = 0
    
    for epoch in range(max_epochs):
        # Train
        idx = np.random.permutation(n_train)
        for i in range(0, n_train, 32):
            batch_idx = idx[i:min(i+32, n_train)]
            loss = model.train_step(X_train[batch_idx], y_train[batch_idx], lr)
        
        # Evaluate every 100 epochs
        if epoch % 100 == 0 or epoch == max_epochs - 1:
            preds_test = model.predict(X_test)
            byte_acc = np.mean(preds_test == y_test)
            exact_acc = np.mean([np.array_equal(preds_test[i], y_test[i]) for i in range(n_test)])
            
            if verbose and epoch % 200 == 0:
                print(f"    Epoch {epoch:4d}: byte={byte_acc*100:5.1f}%, exact={exact_acc*100:5.1f}%")
            
            # Check convergence (95% exact accuracy)
            if exact_acc >= 0.95 and converged_epoch is None:
                converged_epoch = epoch
            
            if exact_acc > best_exact_acc:
                best_exact_acc = exact_acc
    
    train_time = time.time() - start_time
    
    # Final evaluation
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    
    train_byte_acc = np.mean(preds_train == y_train)
    test_byte_acc = np.mean(preds_test == y_test)
    
    train_exact_acc = np.mean([np.array_equal(preds_train[i], y_train[i]) for i in range(n_train)])
    test_exact_acc = np.mean([np.array_equal(preds_test[i], y_test[i]) for i in range(n_test)])
    
    # Mean error (mean number of wrong bytes per sequence)
    train_errors = np.sum(preds_train != y_train, axis=1)
    test_errors = np.sum(preds_test != y_test, axis=1)
    mean_train_error = np.mean(train_errors)
    mean_test_error = np.mean(test_errors)
    
    # Test codec on predictions
    codec_ok = 0
    for i in range(min(50, n_test)):
        try:
            pred_bytes = bytes(preds_test[i])
            kappa = construct_kappa(pred_bytes)
            decoded = algebraic_decode_raw_kronecker(kappa, 256, 32, True)
            if decoded == pred_bytes:
                codec_ok += 1
        except:
            pass
    codec_rate = codec_ok / min(50, n_test)
    
    return {
        'L': L,
        'train_byte_acc': train_byte_acc,
        'test_byte_acc': test_byte_acc,
        'train_exact_acc': train_exact_acc,
        'test_exact_acc': test_exact_acc,
        'mean_train_error': mean_train_error,
        'mean_test_error': mean_test_error,
        'converged_epoch': converged_epoch if converged_epoch else max_epochs,
        'train_time': train_time,
        'codec_rate': codec_rate,
        'p_L_prediction': test_byte_acc ** L,
    }


def main():
    print("="*80)
    print("PHASE 2D: FIXED-LENGTH SCALING SWEEP")
    print("="*80)
    print("\nObjective: Characterize structured byte prediction across lengths")
    print("Lengths: L ∈ {1, 2, 4, 8, 16, 32}")
    print("\nTask: 100 tokens → deterministic L bytes")
    print("Model: Linear (no hidden layer)")
    print("Budget: 2000 epochs per length")
    print()
    
    lengths = [1, 2, 4, 8, 16, 32]
    results = []
    
    for L in lengths:
        print(f"\n{'='*80}")
        print(f"Testing L={L}")
        print(f"{'='*80}")
        
        result = train_and_evaluate(
            L=L,
            n_tokens=100,
            n_train=1000,
            n_test=200,
            max_epochs=2000,
            lr=0.1,
            verbose=True
        )
        
        results.append(result)
        
        print(f"\n  Final Test Results:")
        print(f"    Byte accuracy:     {result['test_byte_acc']*100:6.2f}%")
        print(f"    Exact accuracy:    {result['test_exact_acc']*100:6.2f}%")
        print(f"    Mean error/seq:    {result['mean_test_error']:.3f}")
        print(f"    Converged at:      epoch {result['converged_epoch']}")
        print(f"    Codec works:       {result['codec_rate']*100:5.1f}%")
        print(f"    p^L prediction:    {result['p_L_prediction']*100:6.2f}%")
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print()
    print("| L  | Byte Acc | Exact Seq | Mean Error | Convergence | Codec | p^L Pred |")
    print("|----|----------|-----------|------------|-------------|-------|----------|")
    
    for r in results:
        print(f"| {r['L']:2d} | {r['test_byte_acc']*100:7.2f}% | "
              f"{r['test_exact_acc']*100:8.2f}% | "
              f"{r['mean_test_error']:10.3f} | "
              f"epoch {r['converged_epoch']:4d} | "
              f"{r['codec_rate']*100:4.0f}% | "
              f"{r['p_L_prediction']*100:7.2f}% |")
    
    print()
    
    # Analysis
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print()
    
    # Check if exact accuracy degrades with L
    exact_accs = [r['test_exact_acc'] for r in results]
    if all(acc >= 0.95 for acc in exact_accs):
        print("✅ EXCELLENT: All lengths achieve ≥95% exact accuracy")
        print("   The structured byte interface scales well.")
    elif all(acc >= 0.80 for acc in exact_accs):
        print("✅ GOOD: All lengths achieve ≥80% exact accuracy")
        print("   Minor degradation observed but interface is viable.")
    else:
        print("⚠️  DEGRADATION DETECTED")
        for r in results:
            if r['test_exact_acc'] < 0.80:
                print(f"   L={r['L']}: {r['test_exact_acc']*100:.1f}% exact")
    
    print()
    
    # Check p^L prediction
    print("p^L Degradation Analysis:")
    print("(Does exact accuracy match independent-error prediction?)")
    print()
    for r in results:
        observed = r['test_exact_acc']
        predicted = r['p_L_prediction']
        ratio = observed / predicted if predicted > 0 else 0
        print(f"  L={r['L']:2d}: observed={observed*100:5.1f}%, "
              f"p^L={predicted*100:5.1f}%, ratio={ratio:.2f}")
    
    print()
    print("Ratio ≈ 1.0 → errors are independent")
    print("Ratio < 1.0 → errors are correlated (worse than random)")
    print("Ratio > 1.0 → errors are anti-correlated (better than random)")
    
    print()
    
    # Convergence
    print("Convergence Speed:")
    for r in results:
        print(f"  L={r['L']:2d}: {r['converged_epoch']:4d} epochs")
    
    print()
    
    # Failure modes
    print("Failure Modes:")
    failures = [r for r in results if r['test_exact_acc'] < 0.90]
    if not failures:
        print("  None detected (all ≥90% exact)")
    else:
        for r in failures:
            print(f"  L={r['L']}: {r['test_exact_acc']*100:.1f}% exact, "
                  f"{r['mean_test_error']:.2f} mean errors")
    
    print()
    print("="*80)
    print("PHASE 2D VERDICT")
    print("="*80)
    print()
    
    if all(r['test_exact_acc'] >= 0.95 for r in results):
        print("✅ STRUCTURED BYTE PREDICTION SCALES ACROSS LENGTHS")
        print()
        print("All tested lengths (L=1 to 32) achieve ≥95% exact reconstruction.")
        print("The discrete-factor interface is robust.")
        print()
        print("Ready to proceed:")
        print("  → Phase 2E: Variable length (EOS)")
        print("  → Phase 2F: Transformer integration")
        print("  → Phase 2G: Three-way comparison")
    else:
        print("⚠️  SCALING ISSUES DETECTED")
        print()
        print("Some lengths show degraded performance.")
        print("Investigate before proceeding:")
        print("  - Model capacity for longer sequences")
        print("  - Optimization issues")
        print("  - Task difficulty vs length")
    
    print("="*80)
    
    # Save results
    with open('phase2d_results.txt', 'w') as f:
        f.write("Phase 2D: Fixed-Length Scaling Sweep\n")
        f.write("="*80 + "\n\n")
        f.write("| L  | Byte Acc | Exact Seq | Mean Error | Convergence |\n")
        f.write("|----|----------|-----------|------------|-------------|\n")
        for r in results:
            f.write(f"| {r['L']:2d} | {r['test_byte_acc']*100:7.2f}% | "
                   f"{r['test_exact_acc']*100:8.2f}% | "
                   f"{r['mean_test_error']:10.3f} | "
                   f"epoch {r['converged_epoch']:4d} |\n")
    
    print("\nResults saved to: phase2d_results.txt")


if __name__ == "__main__":
    main()
