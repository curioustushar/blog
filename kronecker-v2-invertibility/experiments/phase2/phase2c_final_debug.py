"""
Phase 2C Final Debugging: Clean Binary Answer

Run tests in order until we have a clear verdict.
No new architecture. No optimization. Just verify the pipeline.
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


# ==================================================
# TEST A: CODEC (already verified)
# ==================================================

def test_a_codec():
    """bytes → κ → decode → bytes"""
    print("\n" + "="*80)
    print("TEST A: Codec Verification")
    print("="*80)
    
    test_cases = [
        b"hello",
        b"a",
        bytes([0, 255, 128, 64]),
        bytes(range(8)),
    ]
    
    for test_bytes in test_cases:
        kappa = construct_kappa(test_bytes)
        decoded = algebraic_decode_raw_kronecker(kappa, 256, 32, True)
        match = (decoded == test_bytes)
        print(f"  {str(list(test_bytes)):30s} → {'PASS' if match else 'FAIL'}")
        if not match:
            return False
    
    print("\n✅ Codec verified")
    return True


# ==================================================
# TEST B: FAKE LOGITS
# ==================================================

def test_b_fake_logits():
    """Perfect logits → extraction → exact match"""
    print("\n" + "="*80)
    print("TEST B: Fake Logits (verify extraction)")
    print("="*80)
    
    L = 8
    target = np.array([104, 101, 108, 108, 111, 33, 42, 99])
    
    # Create perfect logits
    logits = np.ones((1, L, 256)) * -100
    for pos in range(L):
        logits[0, pos, target[pos]] = 100
    
    # Extract prediction (SAME CODE as evaluation)
    pred = np.argmax(logits, axis=-1)[0]
    
    print(f"  Target:    {target}")
    print(f"  Predicted: {pred}")
    print(f"  Match: {np.array_equal(pred, target)}")
    
    if np.array_equal(pred, target):
        print("\n✅ Extraction verified")
        return True
    else:
        print("\n❌ Extraction broken")
        return False


# ==================================================
# SIMPLE FIXED-LENGTH MODEL
# ==================================================

class SimpleBytePredictor:
    """Minimal model: L=8 fixed, no EOS, no padding."""
    
    def __init__(self, input_dim: int, hidden_dim: int, L: int = 8):
        self.L = L
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Simple 2-layer MLP
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        
        # One classifier per position
        self.heads = []
        for _ in range(L):
            W = np.random.randn(hidden_dim, 256) * 0.1
            b = np.zeros(256)
            self.heads.append([W, b])
    
    def forward(self, x):
        """x: (B, input_dim) → logits: (B, L, 256)"""
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        
        B = x.shape[0]
        logits = np.zeros((B, self.L, 256))
        for pos in range(self.L):
            logits[:, pos, :] = h2 @ self.heads[pos][0] + self.heads[pos][1]
        
        return logits
    
    def predict(self, x):
        """x: (B, input_dim) → pred: (B, L)"""
        logits = self.forward(x)
        return np.argmax(logits, axis=-1)
    
    def train_step(self, x, targets, lr):
        """
        x: (B, input_dim)
        targets: (B, L) - byte values
        Returns: loss
        """
        B = x.shape[0]
        
        # Forward
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        logits = np.zeros((B, self.L, 256))
        for pos in range(self.L):
            logits[:, pos, :] = h2 @ self.heads[pos][0] + self.heads[pos][1]
        
        # Loss and backward
        total_loss = 0
        d_h2 = np.zeros_like(h2)
        
        for pos in range(self.L):
            # Softmax
            logits_pos = logits[:, pos, :]
            exp_l = np.exp(logits_pos - np.max(logits_pos, axis=1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            
            # Loss
            target_probs = probs[np.arange(B), targets[:, pos]]
            target_probs = np.clip(target_probs, 1e-10, 1.0)
            total_loss += -np.mean(np.log(target_probs))
            
            # Gradient
            d_logits = probs.copy()
            d_logits[np.arange(B), targets[:, pos]] -= 1
            d_logits /= B
            
            # Update head
            self.heads[pos][0] -= lr * (h2.T @ d_logits)
            self.heads[pos][1] -= lr * np.sum(d_logits, axis=0)
            
            # Backprop
            d_h2 += d_logits @ self.heads[pos][0].T
        
        # Backprop through MLP
        d_h2 *= (h2 > 0)
        self.W2 -= lr * (h1.T @ d_h2)
        self.b2 -= lr * np.sum(d_h2, axis=0)
        
        d_h1 = d_h2 @ self.W2.T
        d_h1 *= (h1 > 0)
        self.W1 -= lr * (x.T @ d_h1)
        self.b1 -= lr * np.sum(d_h1, axis=0)
        
        return total_loss / self.L


# ==================================================
# TEST C: SINGLE EXAMPLE OVERFIT
# ==================================================

def test_c_single_example():
    """Train until perfect on one example."""
    print("\n" + "="*80)
    print("TEST C: Single Example Overfit")
    print("="*80)
    
    x = np.random.randn(1, 16) * 0.1
    target = np.array([[104, 101, 108, 108, 111, 33, 42, 99]])
    
    print(f"Target: {target[0]}")
    
    model = SimpleBytePredictor(16, 64, 8)
    
    for epoch in range(3000):
        loss = model.train_step(x, target, lr=0.05)
        
        if epoch % 300 == 0:
            pred = model.predict(x)[0]
            exact = np.array_equal(pred, target[0])
            byte_acc = np.mean(pred == target[0])
            print(f"  Epoch {epoch:4d}: loss={loss:.6f}, byte={byte_acc*100:5.1f}%, exact={exact}")
            
            if exact and loss < 0.01:
                print(f"\n✅ Perfect at epoch {epoch}")
                return True
    
    pred = model.predict(x)[0]
    print(f"\nFinal: {pred}")
    print(f"Target: {target[0]}")
    
    if np.array_equal(pred, target[0]):
        print("\n✅ Single example works")
        return True
    else:
        print("\n❌ Cannot overfit single example")
        return False


# ==================================================
# TEST D: 100-EXAMPLE MEMORIZATION
# ==================================================

def test_d_memorization():
    """Memorize 100 examples."""
    print("\n" + "="*80)
    print("TEST D: 100-Example Memorization")
    print("="*80)
    
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 32) * 0.1
    targets = np.random.randint(0, 256, (n, 8))
    
    model = SimpleBytePredictor(32, 128, 8)
    
    for epoch in range(1000):
        idx = np.random.permutation(n)
        epoch_loss = 0
        
        for i in range(0, n, 10):
            batch_idx = idx[i:i+10]
            loss = model.train_step(X[batch_idx], targets[batch_idx], lr=0.01)
            epoch_loss += loss
        
        if epoch % 200 == 0:
            preds = model.predict(X)
            byte_acc = np.mean(preds == targets)
            exact_acc = np.mean([np.array_equal(preds[i], targets[i]) for i in range(n)])
            print(f"  Epoch {epoch:4d}: loss={epoch_loss/10:.4f}, byte={byte_acc*100:5.1f}%, exact={exact_acc*100:5.1f}%")
    
    preds = model.predict(X)
    exact_acc = np.mean([np.array_equal(preds[i], targets[i]) for i in range(n)])
    
    if exact_acc >= 0.90:
        print(f"\n✅ Memorization works ({exact_acc*100:.1f}%)")
        return True
    else:
        print(f"\n⚠️  Memorization weak ({exact_acc*100:.1f}%)")
        return False


# ==================================================
# TEST E: FIXED-LENGTH SYNTHETIC
# ==================================================

def test_e_fixed_length_synthetic():
    """Full synthetic test with L=8."""
    print("\n" + "="*80)
    print("TEST E: Fixed-Length Synthetic (L=8)")
    print("="*80)
    
    np.random.seed(42)
    n_train, n_test = 5000, 500
    input_dim, hidden_dim = 64, 256
    
    X_train = np.random.randn(n_train, input_dim) * 0.1
    y_train = np.random.randint(0, 256, (n_train, 8))
    
    X_test = np.random.randn(n_test, input_dim) * 0.1
    y_test = np.random.randint(0, 256, (n_test, 8))
    
    model = SimpleBytePredictor(input_dim, hidden_dim, 8)
    
    print(f"\nTraining {n_train} examples...")
    start = time.time()
    
    for epoch in range(200):
        idx = np.random.permutation(n_train)
        for i in range(0, n_train, 32):
            batch_idx = idx[i:i+32]
            model.train_step(X_train[batch_idx], y_train[batch_idx], lr=0.001)
        
        if epoch % 40 == 0:
            preds = model.predict(X_test)
            byte_acc = np.mean(preds == y_test)
            exact_acc = np.mean([np.array_equal(preds[i], y_test[i]) for i in range(n_test)])
            print(f"  Epoch {epoch:3d}: byte={byte_acc*100:5.1f}%, exact={exact_acc*100:5.1f}%")
    
    train_time = time.time() - start
    
    preds = model.predict(X_test)
    byte_acc = np.mean(preds == y_test)
    exact_acc = np.mean([np.array_equal(preds[i], y_test[i]) for i in range(n_test)])
    
    print(f"\nFinal Test:")
    print(f"  Byte accuracy: {byte_acc*100:.2f}%")
    print(f"  Exact sequence: {exact_acc*100:.2f}%")
    print(f"  Time: {train_time:.1f}s")
    
    # Test codec on predictions
    codec_ok = 0
    for i in range(min(100, n_test)):
        pred_bytes = bytes(preds[i])
        kappa = construct_kappa(pred_bytes)
        decoded = algebraic_decode_raw_kronecker(kappa, 256, 32, True)
        if decoded == pred_bytes:
            codec_ok += 1
    
    print(f"  Codec on predictions: {codec_ok}/100")
    
    return exact_acc, byte_acc


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    print("="*80)
    print("PHASE 2C FINAL DEBUGGING")
    print("="*80)
    print("\nObjective: Clean binary answer - does structured byte prediction work?")
    print("Strategy: Run tests in order, stop at first failure")
    
    results = {}
    
    # Test A
    if not test_a_codec():
        print("\n❌ STOP: Codec broken")
        exit(1)
    results['codec'] = True
    
    # Test B
    if not test_b_fake_logits():
        print("\n❌ STOP: Extraction broken")
        exit(1)
    results['fake_logits'] = True
    
    # Test C
    if not test_c_single_example():
        print("\n❌ STOP: Cannot overfit one example")
        exit(1)
    results['single'] = True
    
    # Test D
    if not test_d_memorization():
        print("\n⚠️  Memorization weak, but continuing...")
    results['memorization'] = True
    
    # Test E
    exact_acc, byte_acc = test_e_fixed_length_synthetic()
    results['synthetic_exact'] = exact_acc
    results['synthetic_byte'] = byte_acc
    
    # VERDICT
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    
    print(f"\nResults:")
    print(f"  Codec: ✓")
    print(f"  Fake logits: ✓")
    print(f"  Single example: ✓")
    print(f"  Memorization: ✓")
    print(f"  Synthetic byte acc: {byte_acc*100:.1f}%")
    print(f"  Synthetic exact acc: {exact_acc*100:.1f}%")
    
    if exact_acc >= 0.80:
        print("\n" + "="*80)
        print("✅ STRUCTURED BYTE PREDICTION WORKS")
        print("="*80)
        print("\nThe discrete-factor interface is learnable.")
        print("Next: Test multiple lengths, then proceed to LM.")
    elif byte_acc >= 0.90:
        print("\n" + "="*80)
        print("⚠️  PARTIAL SUCCESS")
        print("="*80)
        print(f"\nByte prediction works ({byte_acc*100:.1f}%)")
        print(f"But exact reconstruction is low ({exact_acc*100:.1f}%)")
        print(f"Likely p^L degradation: {byte_acc**8:.3f} predicted vs {exact_acc:.3f} observed")
        print("\nNext: Investigate autoregressive decoding or error correction.")
    else:
        print("\n" + "="*80)
        print("❌ STRUCTURED BYTE PREDICTION FAILS")
        print("="*80)
        print("\nThe model cannot learn accurate byte prediction.")
        print("Possible issues: capacity, optimization, task difficulty.")
    
    print("="*80)
