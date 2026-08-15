"""
Phase 2E: Variable-Length with EOS

Test whether the model can learn:
1. Content (byte values)
2. Length (when to emit EOS)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from kronecker_encoder import algebraic_decode_raw_kronecker

EOS = 256  # End-of-sequence token
MAX_LEN = 16  # Maximum sequence length


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


def create_variable_length_task(n_samples, max_len, n_tokens, seed=42):
    """
    Create variable-length task:
    - Token ID determines both content and length
    - Sequences padded to max_len with EOS followed by padding
    
    Returns:
        X: (n_samples, n_tokens) - one-hot input
        y: (n_samples, max_len) - targets with EOS
        lengths: (n_samples,) - actual sequence lengths (before EOS)
    """
    np.random.seed(seed)
    
    X = np.zeros((n_samples, n_tokens))
    y = np.full((n_samples, max_len), EOS, dtype=int)  # Fill with EOS
    lengths = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        token_id = i % n_tokens
        X[i, token_id] = 1.0
        
        # Length determined by token (vary from 1 to max_len)
        length = (token_id % max_len) + 1
        lengths[i] = length
        
        # Generate content bytes
        for pos in range(length):
            byte_val = (token_id * 7 + pos * 13) % 256
            y[i, pos] = byte_val
        
        # EOS at position length (already filled with EOS by default)
    
    return X, y, lengths


def extract_sequence(logits_row):
    """
    Extract sequence from logits, stopping at first EOS.
    
    Args:
        logits_row: (max_len, 257) - logits for one example
    
    Returns:
        bytes: predicted byte sequence (without EOS)
    """
    preds = np.argmax(logits_row, axis=-1)  # (max_len,)
    
    # Find first EOS
    eos_positions = np.where(preds == EOS)[0]
    if len(eos_positions) > 0:
        end_pos = eos_positions[0]
    else:
        end_pos = len(preds)  # No EOS found, use full length
    
    # Extract bytes before EOS
    byte_seq = preds[:end_pos]
    
    # Clip to valid byte range
    byte_seq = np.clip(byte_seq, 0, 255).astype(np.uint8)
    
    return bytes(byte_seq)


class VariableLengthModel:
    """Linear model for variable-length prediction with EOS."""
    
    def __init__(self, n_tokens, max_len):
        self.n_tokens = n_tokens
        self.max_len = max_len
        
        # Linear: (n_tokens, max_len * 257)
        # 257 classes: 0-255 (bytes) + 256 (EOS)
        self.W = np.random.randn(n_tokens, max_len * 257) * 0.01
    
    def forward(self, X):
        """X: (B, n_tokens) → logits: (B, max_len, 257)"""
        logits = X @ self.W
        return logits.reshape(-1, self.max_len, 257)
    
    def predict_sequences(self, X):
        """
        X: (B, n_tokens) → list of byte sequences
        """
        logits = self.forward(X)
        sequences = []
        for i in range(X.shape[0]):
            seq = extract_sequence(logits[i])
            sequences.append(seq)
        return sequences
    
    def train_step(self, X, y, lr):
        """
        X: (B, n_tokens)
        y: (B, max_len) - targets (bytes + EOS)
        """
        B = X.shape[0]
        
        # Forward
        logits = self.forward(X)  # (B, max_len, 257)
        
        # Softmax
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        
        # Loss (all positions, including EOS)
        loss = 0
        for pos in range(self.max_len):
            target_probs = probs[np.arange(B), pos, y[:, pos]]
            target_probs = np.clip(target_probs, 1e-10, 1.0)
            loss -= np.sum(np.log(target_probs))
        loss /= (B * self.max_len)
        
        # Gradient
        grad = probs.copy()
        for b in range(B):
            for pos in range(self.max_len):
                grad[b, pos, y[b, pos]] -= 1
        grad /= (B * self.max_len)
        
        # Update
        grad_W = X.T @ grad.reshape(B, -1)
        self.W -= lr * np.clip(grad_W, -1, 1)
        
        return loss


def evaluate_variable_length(model, X, y_targets, true_lengths):
    """
    Evaluate model on variable-length data.
    
    Returns dict with metrics.
    """
    n = X.shape[0]
    
    # Get predictions
    pred_sequences = model.predict_sequences(X)
    
    # Metrics
    correct_length = 0
    correct_content = 0
    correct_exact = 0
    total_byte_errors = 0
    total_bytes = 0
    
    for i in range(n):
        true_len = true_lengths[i]
        true_bytes = bytes(y_targets[i, :true_len])
        pred_bytes = pred_sequences[i]
        
        # Length accuracy
        if len(pred_bytes) == true_len:
            correct_length += 1
        
        # Content accuracy (for positions that exist in both)
        min_len = min(len(pred_bytes), true_len)
        if min_len > 0:
            matches = sum(pred_bytes[j] == true_bytes[j] for j in range(min_len))
            total_byte_errors += (true_len - matches)
            total_bytes += true_len
            
            if matches == min_len and len(pred_bytes) == true_len:
                correct_content += 1
        
        # Exact match (length and content)
        if pred_bytes == true_bytes:
            correct_exact += 1
    
    return {
        'length_acc': correct_length / n,
        'content_acc': correct_content / n,
        'exact_acc': correct_exact / n,
        'byte_acc': 1.0 - (total_byte_errors / total_bytes) if total_bytes > 0 else 0,
        'mean_byte_error': total_byte_errors / n,
    }


def run_experiment():
    """Run variable-length experiment."""
    print("="*60)
    print("PHASE 2E: VARIABLE-LENGTH WITH EOS")
    print("="*60)
    print()
    
    # Setup
    n_tokens = 50
    max_len = 16
    n_train = 500
    n_test = 100
    epochs = 1000
    
    print(f"Task: {n_tokens} tokens → variable-length sequences (L=1 to {max_len})")
    print(f"Classes: 0-255 (bytes) + 256 (EOS)")
    print(f"Model: Linear")
    print(f"Training: {n_train} examples, {epochs} epochs")
    print()
    
    # Create data
    X_train, y_train, len_train = create_variable_length_task(n_train, max_len, n_tokens, seed=42)
    X_test, y_test, len_test = create_variable_length_task(n_test, max_len, n_tokens, seed=999)
    
    print(f"Length distribution (train):")
    for L in range(1, max_len + 1):
        count = np.sum(len_train == L)
        if count > 0:
            print(f"  L={L:2d}: {count:3d} examples")
    print()
    
    # Model
    model = VariableLengthModel(n_tokens, max_len)
    
    # Training
    print("Training...")
    for epoch in range(epochs):
        # Shuffle
        idx = np.random.permutation(n_train)
        
        # Mini-batches
        for i in range(0, n_train, 25):
            batch = idx[i:min(i+25, n_train)]
            model.train_step(X_train[batch], y_train[batch], lr=0.1)
        
        # Eval
        if epoch % 200 == 0 or epoch == epochs - 1:
            metrics = evaluate_variable_length(model, X_test, y_test, len_test)
            print(f"  E{epoch:4d}: length={metrics['length_acc']*100:5.1f}%, "
                  f"content={metrics['content_acc']*100:5.1f}%, "
                  f"exact={metrics['exact_acc']*100:5.1f}%")
    
    print()
    
    # Final evaluation
    print("="*60)
    print("FINAL RESULTS")
    print("="*60)
    print()
    
    train_metrics = evaluate_variable_length(model, X_train, y_train, len_train)
    test_metrics = evaluate_variable_length(model, X_test, y_test, len_test)
    
    print("Train:")
    print(f"  Length accuracy:  {train_metrics['length_acc']*100:6.2f}%")
    print(f"  Content accuracy: {train_metrics['content_acc']*100:6.2f}%")
    print(f"  Exact accuracy:   {train_metrics['exact_acc']*100:6.2f}%")
    print(f"  Byte accuracy:    {train_metrics['byte_acc']*100:6.2f}%")
    print()
    
    print("Test:")
    print(f"  Length accuracy:  {test_metrics['length_acc']*100:6.2f}%")
    print(f"  Content accuracy: {test_metrics['content_acc']*100:6.2f}%")
    print(f"  Exact accuracy:   {test_metrics['exact_acc']*100:6.2f}%")
    print(f"  Byte accuracy:    {test_metrics['byte_acc']*100:6.2f}%")
    print()
    
    # Test codec on predictions
    print("Codec test (on test predictions):")
    pred_seqs = model.predict_sequences(X_test)
    codec_ok = 0
    for i in range(min(50, n_test)):
        try:
            pred_bytes = pred_seqs[i]
            if len(pred_bytes) > 0:
                kappa = construct_kappa(pred_bytes)
                decoded = algebraic_decode_raw_kronecker(kappa, 256, 32, True)
                if decoded == pred_bytes:
                    codec_ok += 1
        except:
            pass
    
    print(f"  Codec works: {codec_ok}/{min(50, n_test)} ({codec_ok/min(50, n_test)*100:.1f}%)")
    print()
    
    # Analysis by length
    print("="*60)
    print("ANALYSIS BY LENGTH")
    print("="*60)
    print()
    print("| Length | Count | Exact Acc |")
    print("|--------|-------|-----------|")
    
    for L in range(1, max_len + 1):
        mask = (len_test == L)
        if np.sum(mask) == 0:
            continue
        
        X_L = X_test[mask]
        y_L = y_test[mask]
        len_L = len_test[mask]
        
        metrics_L = evaluate_variable_length(model, X_L, y_L, len_L)
        print(f"| {L:6d} | {np.sum(mask):5d} | {metrics_L['exact_acc']*100:8.1f}% |")
    
    print()
    
    # Verdict
    print("="*60)
    print("VERDICT")
    print("="*60)
    print()
    
    if test_metrics['exact_acc'] >= 0.95:
        print("✅ VARIABLE-LENGTH PREDICTION WORKS")
        print()
        print("The model learned both content and length.")
        print("EOS mechanism functions correctly.")
        print()
        print("Ready for:")
        print("  • Phase 2F: Transformer integration")
        print("  • Phase 2G: Three-way comparison")
    elif test_metrics['length_acc'] >= 0.90 and test_metrics['content_acc'] >= 0.90:
        print("⚠️  PARTIAL SUCCESS")
        print()
        print(f"Length prediction works ({test_metrics['length_acc']*100:.1f}%)")
        print(f"Content prediction works ({test_metrics['content_acc']*100:.1f}%)")
        print(f"But exact match is lower ({test_metrics['exact_acc']*100:.1f}%)")
        print()
        print("Both components work independently but need better joint training.")
    else:
        print("❌ VARIABLE-LENGTH NEEDS WORK")
        print()
        if test_metrics['length_acc'] < 0.80:
            print("Issue: Length prediction")
        if test_metrics['content_acc'] < 0.80:
            print("Issue: Content prediction")
    
    print("="*60)
    
    return test_metrics


if __name__ == "__main__":
    metrics = run_experiment()
