"""
Phase 2C Debugging: Fixed-Length Byte Prediction

Simplify to L=8, no length prediction, no EOS, no padding.
Verify the basic pipeline works before adding complexity.
"""

import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from kronecker_encoder import algebraic_decode_raw_kronecker


def construct_kronecker_from_bytes(byte_seq: bytes) -> np.ndarray:
    """Construct exact Kronecker from bytes."""
    char_dim, pos_dim = 256, 32
    L = min(len(byte_seq), pos_dim)
    D = char_dim * pos_dim
    kappa = np.zeros(D, dtype=np.float64)
    
    for p in range(L):
        byte_val = byte_seq[p]
        idx = byte_val * pos_dim + p
        kappa[idx] = 1.0
    
    if L > 0:
        kappa /= np.sqrt(L)
    
    return kappa


# ============================================================================
# STEP 1: TEST CODEC SEPARATELY
# ============================================================================

def test_codec():
    """Test that bytes → κ → decode → bytes works perfectly."""
    print("="*80)
    print("STEP 1: Test Codec (bytes → κ → decode)")
    print("="*80)
    
    n_tests = 1000
    L = 8
    success = 0
    
    for i in range(n_tests):
        # Random byte sequence
        bytes_in = bytes(np.random.randint(0, 256, L))
        
        # Encode
        kappa = construct_kronecker_from_bytes(bytes_in)
        
        # Decode
        bytes_out = algebraic_decode_raw_kronecker(kappa, 256, 32, True)
        
        if bytes_out == bytes_in:
            success += 1
        elif i < 5:  # Show first few failures
            print(f"  Mismatch {i}: {list(bytes_in)} → {list(bytes_out) if bytes_out else None}")
    
    print(f"\nCodec round-trip: {success}/{n_tests} = {success/n_tests*100:.1f}%")
    
    if success == n_tests:
        print("✅ Codec works perfectly")
        return True
    else:
        print("❌ Codec has bugs - fix before proceeding")
        return False


# ============================================================================
# STEP 2: SIMPLE FIXED-LENGTH MODEL
# ============================================================================

class FixedLengthBytePredictor:
    """
    Simplest possible model: predict exactly L bytes.
    No length prediction, no EOS, no padding.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, L: int = 8):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.L = L
        self.char_dim = 256
        
        # MLP
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        
        # Byte classifiers: one per position
        self.byte_heads = []
        for _ in range(L):
            W = np.random.randn(hidden_dim, self.char_dim) * np.sqrt(2.0 / hidden_dim)
            b = np.zeros(self.char_dim)
            self.byte_heads.append((W, b))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            logits: (batch, L, 256)
        """
        # MLP
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        
        # Byte predictions
        batch_size = x.shape[0]
        logits = np.zeros((batch_size, self.L, self.char_dim))
        
        for pos in range(self.L):
            W, b = self.byte_heads[pos]
            logits[:, pos, :] = h2 @ W + b
        
        return logits
    
    def predict_bytes(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            bytes: (batch, L) array of byte values
        """
        logits = self.forward(x)  # (batch, L, 256)
        predicted = np.argmax(logits, axis=-1)  # (batch, L)
        return predicted
    
    def compute_loss_and_update(self, x: np.ndarray, targets: np.ndarray, lr: float) -> float:
        """
        Args:
            x: (batch, input_dim)
            targets: (batch, L) array of byte values
        Returns:
            loss: scalar
        """
        batch_size = x.shape[0]
        
        # Forward (save activations)
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        
        logits = np.zeros((batch_size, self.L, self.char_dim))
        for pos in range(self.L):
            W, b = self.byte_heads[pos]
            logits[:, pos, :] = h2 @ W + b
        
        # Loss
        total_loss = 0.0
        d_h2 = np.zeros_like(h2)
        
        for pos in range(self.L):
            logits_pos = logits[:, pos, :]  # (batch, 256)
            targets_pos = targets[:, pos]    # (batch,)
            
            # Softmax
            exp_logits = np.exp(logits_pos - np.max(logits_pos, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Cross-entropy
            correct_probs = probs[np.arange(batch_size), targets_pos]
            correct_probs = np.clip(correct_probs, 1e-10, 1.0)
            loss_pos = -np.mean(np.log(correct_probs))
            total_loss += loss_pos
            
            # Gradient
            d_logits = probs.copy()
            d_logits[np.arange(batch_size), targets_pos] -= 1
            d_logits /= batch_size
            
            # Update byte head
            W, b = self.byte_heads[pos]
            W -= lr * (h2.T @ d_logits)
            b -= lr * np.sum(d_logits, axis=0)
            self.byte_heads[pos] = (W, b)
            
            # Backprop to h2
            d_h2 += d_logits @ W.T
        
        # Backprop through ReLU
        d_h2 = d_h2 * (h2 > 0)
        
        # Update MLP
        self.W2 -= lr * (h1.T @ d_h2)
        self.b2 -= lr * np.sum(d_h2, axis=0)
        
        d_h1 = d_h2 @ self.W2.T
        d_h1 = d_h1 * (h1 > 0)
        
        self.W1 -= lr * (x.T @ d_h1)
        self.b1 -= lr * np.sum(d_h1, axis=0)
        
        return total_loss / self.L


# ============================================================================
# STEP 3: TEST WITH FAKE LOGITS
# ============================================================================

def test_fake_logits():
    """Test extraction with perfect logits."""
    print("\n" + "="*80)
    print("STEP 3: Test Fake Logits (perfect predictions)")
    print("="*80)
    
    L = 8
    n_tests = 100
    
    success = 0
    for i in range(n_tests):
        # Target
        target = np.random.randint(0, 256, L)
        
        # Create "perfect" logits: huge value for correct class, low elsewhere
        logits = np.ones((1, L, 256)) * -100
        for pos in range(L):
            logits[0, pos, target[pos]] = 100
        
        # Extract prediction
        predicted = np.argmax(logits, axis=-1)[0]  # (L,)
        
        # Check
        if np.array_equal(predicted, target):
            success += 1
        elif i < 3:
            print(f"  Mismatch {i}: target={target}, pred={predicted}")
    
    print(f"\nFake logit extraction: {success}/{n_tests} = {success/n_tests*100:.1f}%")
    
    if success == n_tests:
        print("✅ Extraction logic is correct")
        return True
    else:
        print("❌ Extraction has bugs")
        return False


# ============================================================================
# STEP 4: SINGLE EXAMPLE OVERFIT
# ============================================================================

def test_single_example_overfit():
    """Train until perfect on single example."""
    print("\n" + "="*80)
    print("STEP 4: Single Example Overfit")
    print("="*80)
    
    L = 8
    input_dim = 16
    hidden_dim = 64
    
    # Fixed input and target
    x = np.random.randn(1, input_dim) * 0.1
    target = np.array([[104, 101, 108, 108, 111, 33, 42, 99]])
    
    print(f"Target: {target[0]}")
    
    model = FixedLengthBytePredictor(input_dim, hidden_dim, L)
    
    # Train
    for epoch in range(2000):
        loss = model.compute_loss_and_update(x, target, lr=0.1)
        
        if epoch % 200 == 0:
            pred = model.predict_bytes(x)[0]
            exact_match = np.array_equal(pred, target[0])
            byte_acc = np.mean(pred == target[0])
            
            print(f"  Epoch {epoch:4d}: loss={loss:.6f}, byte_acc={byte_acc*100:5.1f}%, exact={exact_match}")
            
            if exact_match:
                print(f"\n✅ Perfect match at epoch {epoch}!")
                break
    
    # Final check
    pred = model.predict_bytes(x)[0]
    exact_match = np.array_equal(pred, target[0])
    
    print(f"\nFinal:")
    print(f"  Target:    {target[0]}")
    print(f"  Predicted: {pred}")
    print(f"  Match: {exact_match}")
    
    if exact_match:
        print("✅ Single example overfit works")
        return True
    else:
        print("❌ Cannot overfit single example - architecture or optimization issue")
        return False


# ============================================================================
# STEP 5: SMALL MEMORIZATION TEST
# ============================================================================

def test_small_memorization():
    """Can model memorize 100 examples?"""
    print("\n" + "="*80)
    print("STEP 5: Small Memorization (100 examples)")
    print("="*80)
    
    L = 8
    input_dim = 32
    hidden_dim = 128
    n_samples = 100
    
    # Generate unique input-output pairs
    np.random.seed(42)
    X = np.random.randn(n_samples, input_dim) * 0.1
    targets = np.random.randint(0, 256, (n_samples, L))
    
    model = FixedLengthBytePredictor(input_dim, hidden_dim, L)
    
    # Train
    batch_size = 10
    n_epochs = 500
    
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        epoch_loss = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = X[batch_idx]
            target_batch = targets[batch_idx]
            
            loss = model.compute_loss_and_update(x_batch, target_batch, lr=0.01)
            epoch_loss += loss
        
        if epoch % 100 == 0:
            # Evaluate
            preds = model.predict_bytes(X)
            exact_match = np.mean([np.array_equal(preds[i], targets[i]) for i in range(n_samples)])
            byte_acc = np.mean(preds == targets)
            
            print(f"  Epoch {epoch:3d}: loss={epoch_loss/(n_samples/batch_size):.4f}, byte_acc={byte_acc*100:5.1f}%, exact={exact_match*100:5.1f}%")
    
    # Final evaluation
    preds = model.predict_bytes(X)
    exact_match = np.mean([np.array_equal(preds[i], targets[i]) for i in range(n_samples)])
    
    print(f"\nFinal exact sequence accuracy: {exact_match*100:.1f}%")
    
    if exact_match >= 0.95:
        print("✅ Memorization works")
        return True
    else:
        print("⚠️  Memorization accuracy lower than expected")
        return False


# ============================================================================
# STEP 6: FIXED-LENGTH SYNTHETIC EXPERIMENT
# ============================================================================

def test_fixed_length_synthetic():
    """Full synthetic experiment with L=8."""
    print("\n" + "="*80)
    print("STEP 6: Fixed-Length Synthetic (L=8)")
    print("="*80)
    
    L = 8
    input_dim = 64
    hidden_dim = 256
    n_train = 5000
    n_test = 500
    n_epochs = 100
    batch_size = 32
    
    # Generate data
    np.random.seed(42)
    X_train = np.random.randn(n_train, input_dim) * 0.1
    targets_train = np.random.randint(0, 256, (n_train, L))
    
    X_test = np.random.randn(n_test, input_dim) * 0.1
    targets_test = np.random.randint(0, 256, (n_test, L))
    
    model = FixedLengthBytePredictor(input_dim, hidden_dim, L)
    
    print(f"\nTraining {n_train} samples, {n_epochs} epochs...")
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_train)
        epoch_loss = 0
        
        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = X_train[batch_idx]
            target_batch = targets_train[batch_idx]
            
            loss = model.compute_loss_and_update(x_batch, target_batch, lr=0.001)
            epoch_loss += loss
        
        if epoch % 20 == 0:
            # Test evaluation
            preds_test = model.predict_bytes(X_test)
            exact_match = np.mean([np.array_equal(preds_test[i], targets_test[i]) for i in range(n_test)])
            byte_acc = np.mean(preds_test == targets_test)
            
            print(f"  Epoch {epoch:3d}: loss={epoch_loss*batch_size/n_train:.4f}, byte_acc={byte_acc*100:5.1f}%, exact={exact_match*100:5.1f}%")
    
    train_time = time.time() - start_time
    
    # Final evaluation
    preds_test = model.predict_bytes(X_test)
    exact_match = np.mean([np.array_equal(preds_test[i], targets_test[i]) for i in range(n_test)])
    byte_acc = np.mean(preds_test == targets_test)
    
    print(f"\nFinal Test Results:")
    print(f"  Byte accuracy: {byte_acc*100:.2f}%")
    print(f"  Exact sequence accuracy: {exact_match*100:.2f}%")
    print(f"  Training time: {train_time:.1f}s")
    
    # Test codec on predictions
    codec_success = 0
    for i in range(min(100, n_test)):
        pred_bytes = bytes(preds_test[i])
        kappa = construct_kronecker_from_bytes(pred_bytes)
        decoded = algebraic_decode_raw_kronecker(kappa, 256, 32, True)
        if decoded == pred_bytes:
            codec_success += 1
    
    print(f"  Codec round-trip on predictions: {codec_success}/100 = {codec_success}%")
    
    return exact_match


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("="*80)
    print("PHASE 2C DEBUGGING: FIXED-LENGTH PIPELINE")
    print("="*80)
    print("\nObjective: Verify basic pipeline works before adding complexity")
    print("Strategy: L=8 fixed length, no EOS, no padding, no length prediction")
    
    # Run tests in order
    tests_passed = []
    
    # Step 1: Codec
    if test_codec():
        tests_passed.append("codec")
    else:
        print("\n❌ STOP: Fix codec before proceeding")
        exit(1)
    
    # Step 2: Skip (simple model class is defined above)
    
    # Step 3: Fake logits
    if test_fake_logits():
        tests_passed.append("fake_logits")
    else:
        print("\n❌ STOP: Fix extraction logic")
        exit(1)
    
    # Step 4: Single example
    if test_single_example_overfit():
        tests_passed.append("single_example")
    else:
        print("\n❌ STOP: Cannot overfit single example")
        exit(1)
    
    # Step 5: Memorization
    if test_small_memorization():
        tests_passed.append("memorization")
    
    # Step 6: Synthetic
    exact_acc = test_fixed_length_synthetic()
    if exact_acc >= 0.8:
        tests_passed.append("synthetic")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTests passed: {', '.join(tests_passed)}")
    
    if "synthetic" in tests_passed:
        print("\n✅ Fixed-length pipeline works!")
        print("   → Can proceed to variable-length next")
    elif "memorization" in tests_passed:
        print("\n⚠️  Memorization works but generalization is weak")
        print("   → May need more capacity or better optimization")
    else:
        print("\n❌ Basic pipeline has issues")
        print("   → Fix before proceeding")
    
    print("="*80)
