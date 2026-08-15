"""
Phase 2D: Fast Length Sweep

Optimized for speed while maintaining scientific validity.
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


def create_task(n_samples, L, n_tokens, seed=42):
    """Deterministic task: token_id → L bytes"""
    np.random.seed(seed)
    X = np.zeros((n_samples, n_tokens))
    y = np.zeros((n_samples, L), dtype=np.uint8)
    
    for i in range(n_samples):
        token_id = i % n_tokens
        X[i, token_id] = 1.0
        for pos in range(L):
            y[i, pos] = (token_id * 7 + pos * 13) % 256
    
    return X, y


class LinearModel:
    """Simple linear model"""
    
    def __init__(self, n_tokens, L):
        self.L = L
        self.W = np.random.randn(n_tokens, L * 256) * 0.01
    
    def forward(self, X):
        return (X @ self.W).reshape(-1, self.L, 256)
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=-1).astype(np.uint8)
    
    def train_step(self, X, y, lr):
        B = X.shape[0]
        logits = self.forward(X)
        
        # Softmax
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        
        # Loss
        loss = 0
        for pos in range(self.L):
            loss -= np.mean(np.log(probs[np.arange(B), pos, y[:, pos]] + 1e-10))
        
        # Gradient
        grad = probs.copy()
        for b in range(B):
            for pos in range(self.L):
                grad[b, pos, y[b, pos]] -= 1
        grad /= (B * self.L)
        
        # Update
        grad_W = X.T @ grad.reshape(B, -1)
        self.W -= lr * np.clip(grad_W, -1, 1)
        
        return loss / self.L


def run_length(L):
    """Run experiment for one length"""
    print(f"\n{'='*60}")
    print(f"L={L}")
    print(f"{'='*60}")
    
    n_tokens = 50
    n_train = 500
    n_test = 100
    max_epochs = 800
    
    # Data
    X_train, y_train = create_task(n_train, L, n_tokens, seed=42)
    X_test, y_test = create_task(n_test, L, n_tokens, seed=999)
    
    # Model
    model = LinearModel(n_tokens, L)
    
    # Train
    start = time.time()
    converged = None
    
    for epoch in range(max_epochs):
        # Training
        idx = np.random.permutation(n_train)
        for i in range(0, n_train, 25):
            batch = idx[i:i+25]
            model.train_step(X_train[batch], y_train[batch], lr=0.1)
        
        # Eval
        if epoch % 100 == 0 or epoch == max_epochs - 1:
            pred = model.predict(X_test)
            byte_acc = np.mean(pred == y_test)
            exact_acc = np.mean([np.array_equal(pred[i], y_test[i]) for i in range(n_test)])
            
            print(f"  Epoch {epoch:3d}: byte={byte_acc*100:5.1f}%, exact={exact_acc*100:5.1f}%")
            
            if exact_acc >= 0.95 and converged is None:
                converged = epoch
    
    elapsed = time.time() - start
    
    # Final metrics
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    byte_acc = np.mean(pred_test == y_test)
    exact_acc = np.mean([np.array_equal(pred_test[i], y_test[i]) for i in range(n_test)])
    mean_err = np.mean(np.sum(pred_test != y_test, axis=1))
    
    # Codec test
    codec_ok = sum(
        1 for i in range(min(50, n_test))
        if (lambda p: (
            kappa := construct_kappa(bytes(p)),
            algebraic_decode_raw_kronecker(kappa, 256, 32, True) == bytes(p)
        )[1])(pred_test[i])
    )
    codec_rate = codec_ok / min(50, n_test)
    
    print(f"\n  Results:")
    print(f"    Byte acc:   {byte_acc*100:6.2f}%")
    print(f"    Exact acc:  {exact_acc*100:6.2f}%")
    print(f"    Mean error: {mean_err:.3f}")
    print(f"    Converged:  epoch {converged if converged else 'N/A'}")
    print(f"    Codec:      {codec_rate*100:5.1f}%")
    print(f"    Time:       {elapsed:.1f}s")
    
    return {
        'L': L,
        'byte_acc': byte_acc,
        'exact_acc': exact_acc,
        'mean_err': mean_err,
        'converged': converged if converged else max_epochs,
        'p_L': byte_acc ** L,
        'codec_rate': codec_rate,
    }


def main():
    print("="*60)
    print("PHASE 2D: FIXED-LENGTH SCALING SWEEP (FAST)")
    print("="*60)
    print("\nTask: 50 tokens → L bytes (deterministic)")
    print("Model: Linear")
    print("Budget: 800 epochs/length")
    
    lengths = [1, 2, 4, 8, 16, 32]
    results = []
    
    for L in lengths:
        r = run_length(L)
        results.append(r)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print()
    print("| L  | Byte Acc | Exact Acc | Mean Err | Conv  | p^L   |")
    print("|----|----------|-----------|----------|-------|-------|")
    for r in results:
        print(f"| {r['L']:2d} | {r['byte_acc']*100:7.2f}% | "
              f"{r['exact_acc']*100:8.2f}% | "
              f"{r['mean_err']:8.3f} | "
              f"E{r['converged']:3d} | "
              f"{r['p_L']*100:5.1f}% |")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    all_good = all(r['exact_acc'] >= 0.95 for r in results)
    
    if all_good:
        print("\n✅ ALL LENGTHS: ≥95% exact accuracy")
        print("\nThe structured byte interface scales well from L=1 to L=32.")
    else:
        print("\n⚠️  Some lengths show degradation:")
        for r in results:
            if r['exact_acc'] < 0.95:
                print(f"   L={r['L']}: {r['exact_acc']*100:.1f}%")
    
    print("\np^L Analysis (independence test):")
    for r in results:
        ratio = r['exact_acc'] / r['p_L'] if r['p_L'] > 0 else 0
        print(f"  L={r['L']:2d}: exact={r['exact_acc']*100:5.1f}%, "
              f"p^L={r['p_L']*100:5.1f}%, ratio={ratio:.2f}")
    
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    
    if all_good:
        print("\n✅ STRUCTURED BYTE PREDICTION SCALES")
        print("\nReady for:")
        print("  • Variable length (EOS)")
        print("  • Transformer integration")
        print("  • Three-way comparison")
    else:
        print("\n⚠️  INVESTIGATE SCALING ISSUES")
    
    print("="*60)
    
    # Save
    with open('phase2d_results.txt', 'w') as f:
        f.write("Phase 2D Results\n\n")
        f.write("| L  | Byte Acc | Exact Acc | Mean Err | Conv  |\n")
        f.write("|----|----------|-----------|----------|-------|\n")
        for r in results:
            f.write(f"| {r['L']:2d} | {r['byte_acc']*100:7.2f}% | "
                   f"{r['exact_acc']*100:8.2f}% | "
                   f"{r['mean_err']:8.3f} | E{r['converged']:3d} |\n")
    
    print("\nSaved: phase2d_results.txt")


if __name__ == "__main__":
    main()
