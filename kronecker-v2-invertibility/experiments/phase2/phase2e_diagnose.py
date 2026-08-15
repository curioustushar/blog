"""
Phase 2E Diagnosis: Can model overfit small variable-length dataset?
"""

import numpy as np

EOS = 256
MAX_LEN = 8  # Reduced from 16

def create_task(n, max_len, n_tok):
    """Variable-length task"""
    np.random.seed(42)
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
    return bytes(np.clip(preds[:end], 0, 255).astype(np.uint8))

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
    """Evaluate"""
    n = X.shape[0]
    logits = model.forward(X)
    
    correct_len = 0
    correct_exact = 0
    byte_errors = 0
    total_bytes = 0
    
    for i in range(n):
        true_len = lengths[i]
        true_bytes = bytes(y[i, :true_len])
        pred_bytes = extract_seq(logits[i])
        
        if len(pred_bytes) == true_len:
            correct_len += 1
        
        if pred_bytes == true_bytes:
            correct_exact += 1
        
        min_len = min(len(pred_bytes), true_len)
        if min_len > 0:
            byte_errors += sum(pred_bytes[j] != true_bytes[j] for j in range(min_len))
        byte_errors += abs(len(pred_bytes) - true_len)
        total_bytes += true_len
    
    return {
        'len_acc': correct_len / n,
        'exact_acc': correct_exact / n,
        'byte_acc': 1 - (byte_errors / total_bytes) if total_bytes > 0 else 0,
    }

print("="*60)
print("PHASE 2E DIAGNOSIS")
print("="*60)
print()

# Test 1: Single example
print("Test 1: Can it overfit 1 example?")
print("-"*60)

X1, y1, len1 = create_task(1, 8, 10)
model1 = Model(10, 8)

print(f"Target: length={len1[0]}, bytes={list(y1[0, :len1[0]])}")

for epoch in [0, 200, 500, 1000]:
    if epoch > 0:
        for _ in range(200):
            model1.train_step(X1, y1, lr=0.1)
    
    metrics = eval_model(model1, X1, y1, len1)
    pred = extract_seq(model1.forward(X1)[0])
    print(f"E{epoch:4d}: len={metrics['len_acc']*100:5.1f}%, exact={metrics['exact_acc']*100:5.1f}%, pred={list(pred)}")

print()

# Test 2: 10 examples
print("Test 2: Can it memorize 10 examples?")
print("-"*60)

X10, y10, len10 = create_task(10, 8, 10)
model10 = Model(10, 8)

print("Lengths:", len10)

for epoch in [0, 500, 1000, 2000]:
    if epoch > 0:
        for _ in range(500):
            model10.train_step(X10, y10, lr=0.05)
    
    metrics = eval_model(model10, X10, y10, len10)
    print(f"E{epoch:4d}: len={metrics['len_acc']*100:5.1f}%, exact={metrics['exact_acc']*100:5.1f}%, byte={metrics['byte_acc']*100:5.1f}%")

print()

# Test 3: Compare with fixed-length L=4
print("Test 3: Fixed-length L=4 (baseline)")
print("-"*60)

X_fix, y_fix, len_fix = create_task(50, 4, 50)
# Make all length 4
for i in range(50):
    t = i
    for p in range(4):
        y_fix[i, p] = (t * 7 + p * 13) % 256

model_fix = Model(50, 4)

for epoch in [0, 200, 500]:
    if epoch > 0:
        for _ in range(200):
            idx = np.random.permutation(50)
            for j in range(0, 50, 10):
                batch = idx[j:j+10]
                model_fix.train_step(X_fix[batch], y_fix[batch], lr=0.1)
    
    preds = np.argmax(model_fix.forward(X_fix), axis=-1)
    byte_acc = np.mean(preds[:, :4] == y_fix[:, :4])
    exact_acc = np.mean([np.array_equal(preds[i, :4], y_fix[i, :4]) for i in range(50)])
    print(f"E{epoch:3d}: byte={byte_acc*100:5.1f}%, exact={exact_acc*100:5.1f}%")

print()
print("="*60)
print("DIAGNOSIS")
print("="*60)
print()
print("Comparing Test 1 (single example) and Test 3 (fixed-length):")
print("  • If Test 1 reaches 100% exact: model CAN learn variable-length")
print("  • If Test 3 works but Test 1 fails: variable-length is harder")
print("  • If Test 2 works partially: capacity/optimization issue")
print()
print("="*60)
