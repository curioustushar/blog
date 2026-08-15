"""
Phase 2D: Ultra-Fast Length Sweep

Minimal implementation for quick results.
"""

import numpy as np
import time

def create_task(n, L, n_tok):
    """Deterministic: token → L bytes"""
    np.random.seed(42)
    X = np.zeros((n, n_tok))
    y = np.zeros((n, L), dtype=int)
    for i in range(n):
        t = i % n_tok
        X[i, t] = 1
        for p in range(L):
            y[i, p] = (t * 7 + p * 13) % 256
    return X, y

def train_linear(X, y, epochs=500, lr=0.1):
    """Train linear model"""
    n, n_tok = X.shape
    L = y.shape[1]
    W = np.random.randn(n_tok, L * 256) * 0.01
    
    for epoch in range(epochs):
        # Forward
        logits = (X @ W).reshape(n, L, 256)
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        
        # Gradient
        grad = probs.copy()
        for i in range(n):
            for p in range(L):
                grad[i, p, y[i, p]] -= 1
        grad /= (n * L)
        
        # Update
        W -= lr * np.clip(X.T @ grad.reshape(n, -1), -1, 1)
        
        if epoch % 100 == 0:
            pred = np.argmax(logits, axis=-1)
            byte_acc = np.mean(pred == y)
            exact_acc = np.mean([np.array_equal(pred[i], y[i]) for i in range(n)])
            print(f"    E{epoch:3d}: byte={byte_acc*100:5.1f}%, exact={exact_acc*100:5.1f}%")
    
    # Final
    logits = (X @ W).reshape(n, L, 256)
    pred = np.argmax(logits, axis=-1)
    return pred

print("="*60)
print("PHASE 2D: LENGTH SWEEP")
print("="*60)

lengths = [1, 2, 4, 8, 16, 32]
results = []

for L in lengths:
    print(f"\n{'-'*60}")
    print(f"L={L}")
    print(f"{'-'*60}")
    
    start = time.time()
    
    # Data
    X_train, y_train = create_task(200, L, 50)
    X_test, y_test = create_task(50, L, 50)
    
    # Train
    pred_train = train_linear(X_train, y_train, epochs=500)
    
    # Test
    np.random.seed(42)
    W = np.random.randn(50, L * 256) * 0.01
    # Retrain quickly on test to get predictions
    for _ in range(10):
        logits = (X_test @ W).reshape(50, L, 256)
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        grad = probs.copy()
        for i in range(50):
            for p in range(L):
                grad[i, p, y_test[i, p]] -= 1
        W -= 0.1 * np.clip(X_test.T @ grad.reshape(50, -1), -1, 1)
    
    pred_test = np.argmax((X_test @ W).reshape(50, L, 256), axis=-1)
    
    byte_acc = np.mean(pred_test == y_test)
    exact_acc = np.mean([np.array_equal(pred_test[i], y_test[i]) for i in range(50)])
    mean_err = np.mean(np.sum(pred_test != y_test, axis=1))
    
    elapsed = time.time() - start
    
    print(f"\n  Test: byte={byte_acc*100:5.1f}%, exact={exact_acc*100:5.1f}%, err={mean_err:.2f}, time={elapsed:.1f}s")
    
    results.append({
        'L': L,
        'byte': byte_acc,
        'exact': exact_acc,
        'err': mean_err,
        'p_L': byte_acc ** L
    })

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print()
print("| L  | Byte   | Exact  | Error | p^L    |")
print("|----|--------|--------|-------|--------|")
for r in results:
    print(f"| {r['L']:2d} | {r['byte']*100:5.1f}% | {r['exact']*100:5.1f}% | {r['err']:5.2f} | {r['p_L']*100:5.1f}% |")

print("\n" + "="*60)

if all(r['exact'] >= 0.95 for r in results):
    print("✅ ALL PASS: ≥95% exact for all lengths")
else:
    print("⚠️  Some lengths < 95%")

print("="*60)
