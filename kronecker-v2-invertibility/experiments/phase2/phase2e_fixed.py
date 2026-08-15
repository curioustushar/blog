"""
Phase 2E Fixed: Variable-length with corrected evaluation
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from kronecker_encoder import algebraic_decode_raw_kronecker

EOS = 256
MAX_LEN = 8  # Start smaller

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

def create_task(n, max_len, n_tok, seed=42):
    """Variable-length task"""
    np.random.seed(seed)
    X = np.zeros((n, n_tok))
    y = np.full((n, max_len), EOS, dtype=int)
    lengths = np.zeros(n, dtype=int)
    
    for i in range(n):
        t = i % n_tok
        X[i, t] = 1
        length = (t % max_len) + 1
        lengths[i] = length
        for p in range(length):
            y[i, p] = (t * 7 + p * 13) % 256
    
    return X, y, lengths

def extract_seq(logits_row):
    """Extract sequence, stop at EOS"""
    preds = np.argmax(logits_row, axis=-1)
    eos_pos = np.where(preds == EOS)[0]
    end = eos_pos[0] if len(eos_pos) > 0 else len(preds)
    byte_seq = preds[:end]
    return np.clip(byte_seq, 0, 255).astype(np.uint8)

class Model:
    def __init__(self, n_tok, max_len):
        self.W = np.random.randn(n_tok, max_len * 257) * 0.01
        self.max_len = max_len
    
    def forward(self, X):
        return (X @ self.W).reshape(-1, self.max_len, 257)
    
    def train_step(self, X, y, lr):
        B = X.shape[0]
        logits = self.forward(X)
        
        # Softmax
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        
        # Loss
        loss = 0
        for p in range(self.max_len):
            loss -= np.mean(np.log(probs[np.arange(B), p, y[:, p]] + 1e-10))
        
        # Gradient
        grad = probs.copy()
        for b in range(B):
            for p in range(self.max_len):
                grad[b, p, y[b, p]] -= 1
        grad /= (B * self.max_len)
        
        # Update
        self.W -= lr * np.clip(X.T @ grad.reshape(B, -1), -1, 1)
        return loss / self.max_len

def eval_model(model, X, y, lengths):
    """Evaluate - FIXED VERSION"""
    n = X.shape[0]
    logits = model.forward(X)
    
    correct_len = 0
    correct_exact = 0
    byte_errors = 0
    total_bytes = 0
    
    for i in range(n):
        true_len = lengths[i]
        true_bytes = y[i, :true_len]  # numpy array
        pred_bytes = extract_seq(logits[i])  # numpy array
        
        # Length check
        if len(pred_bytes) == true_len:
            correct_len += 1
        
        # Exact match check (KEY FIX: proper array comparison)
        if np.array_equal(pred_bytes, true_bytes):
            correct_exact += 1
        
        # Byte accuracy
        min_len = min(len(pred_bytes), true_len)
        if min_len > 0:
            byte_errors += np.sum(pred_bytes[:min_len] != true_bytes[:min_len])
        byte_errors += abs(len(pred_bytes) - true_len)
        total_bytes += true_len
    
    return {
        'len_acc': correct_len / n,
        'exact_acc': correct_exact / n,
        'byte_acc': 1 - (byte_errors / total_bytes) if total_bytes > 0 else 0,
    }

print("="*60)
print("PHASE 2E: VARIABLE-LENGTH (FIXED)")
print("="*60)
print()

# Parameters
n_tok = 25
max_len = 8
n_train = 200
n_test = 50
epochs = 1000

print(f"Task: {n_tok} tokens → L=1 to {max_len}")
print(f"Training: {n_train} examples, {epochs} epochs")
print()

# Data
X_train, y_train, len_train = create_task(n_train, max_len, n_tok, seed=42)
X_test, y_test, len_test = create_task(n_test, max_len, n_tok, seed=999)

print("Length distribution (train):")
for L in range(1, max_len + 1):
    count = np.sum(len_train == L)
    if count > 0:
        print(f"  L={L}: {count} examples")
print()

# Model
model = Model(n_tok, max_len)

# Training
print("Training...")
for epoch in range(epochs + 1):
    if epoch > 0:
        idx = np.random.permutation(n_train)
        for i in range(0, n_train, 20):
            batch = idx[i:min(i+20, n_train)]
            model.train_step(X_train[batch], y_train[batch], lr=0.1)
    
    if epoch % 200 == 0:
        metrics = eval_model(model, X_test, y_test, len_test)
        print(f"  E{epoch:4d}: len={metrics['len_acc']*100:5.1f}%, "
              f"byte={metrics['byte_acc']*100:5.1f}%, "
              f"exact={metrics['exact_acc']*100:5.1f}%")

print()

# Final results
print("="*60)
print("FINAL RESULTS")
print("="*60)
print()

train_metrics = eval_model(model, X_train, y_train, len_train)
test_metrics = eval_model(model, X_test, y_test, len_test)

print("Train:")
print(f"  Length acc: {train_metrics['len_acc']*100:6.2f}%")
print(f"  Byte acc:   {train_metrics['byte_acc']*100:6.2f}%")
print(f"  Exact acc:  {train_metrics['exact_acc']*100:6.2f}%")
print()

print("Test:")
print(f"  Length acc: {test_metrics['len_acc']*100:6.2f}%")
print(f"  Byte acc:   {test_metrics['byte_acc']*100:6.2f}%")
print(f"  Exact acc:  {test_metrics['exact_acc']*100:6.2f}%")
print()

# Codec test
print("Codec test:")
logits = model.forward(X_test)
codec_ok = 0
for i in range(min(50, n_test)):
    try:
        pred_bytes = bytes(extract_seq(logits[i]))
        if len(pred_bytes) > 0:
            kappa = construct_kappa(pred_bytes)
            decoded = algebraic_decode_raw_kronecker(kappa, 256, 32, True)
            if decoded == pred_bytes:
                codec_ok += 1
    except:
        pass

print(f"  {codec_ok}/{min(50, n_test)} predictions decodable")
print()

# Analysis by length
print("="*60)
print("BY LENGTH")
print("="*60)
print()
print("| L | Count | Exact |")
print("|---|-------|-------|")

for L in range(1, max_len + 1):
    mask = (len_test == L)
    if np.sum(mask) == 0:
        continue
    
    X_L = X_test[mask]
    y_L = y_test[mask]
    len_L = len_test[mask]
    
    m = eval_model(model, X_L, y_L, len_L)
    print(f"| {L} | {np.sum(mask):5d} | {m['exact_acc']*100:5.1f}% |")

print()

# Verdict
print("="*60)
print("VERDICT")
print("="*60)
print()

if test_metrics['exact_acc'] >= 0.95:
    print("✅ VARIABLE-LENGTH WORKS")
    print("\nReady for Transformer integration.")
elif test_metrics['exact_acc'] >= 0.80:
    print("⚠️  MOSTLY WORKS")
    print(f"\n{test_metrics['exact_acc']*100:.1f}% exact accuracy.")
    print("Close to working, may need more training or capacity.")
elif test_metrics['len_acc'] >= 0.90 and test_metrics['byte_acc'] >= 0.80:
    print("⚠️  PARTIAL SUCCESS")
    print("\nLength and byte prediction work independently.")
    print("Joint optimization needs improvement.")
else:
    print("❌ NEEDS WORK")

print("="*60)
