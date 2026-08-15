"""
Phase 2F: Transformer Integration

Test structured byte prediction with actual Transformer architecture.
Task: Synthetic next-token prediction (copy task)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from kronecker_encoder import algebraic_decode_raw_kronecker

EOS = 256
MAX_BYTE_LEN = 8

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


# Simplified Transformer components
def softmax(x, axis=-1):
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class LayerNorm:
    """Layer normalization"""
    def __init__(self, dim):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = 1e-5
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        self.x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * self.x_norm + self.beta
    
    def backward(self, grad_out):
        # Simplified - skip for speed
        return grad_out


class MiniTransformer:
    """
    Minimal Transformer for testing structured byte head.
    Simplified to focus on the output interface, not full Transformer training.
    """
    
    def __init__(self, vocab_size, d_model, n_layers, max_seq_len, max_byte_len):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.max_byte_len = max_byte_len
        
        # Embedding
        self.token_embed = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, d_model) * 0.02
        
        # Simple MLP layers (simplified - no attention for speed)
        self.layers = []
        for _ in range(n_layers):
            layer = {
                'W1': np.random.randn(d_model, d_model * 2) * 0.02,
                'W2': np.random.randn(d_model * 2, d_model) * 0.02,
                'ln1': LayerNorm(d_model),
                'ln2': LayerNorm(d_model),
            }
            self.layers.append(layer)
        
        # Output: structured byte head
        # Predicts: max_byte_len positions × 257 classes (0-255 bytes + EOS)
        self.byte_head = np.random.randn(d_model, max_byte_len * 257) * 0.02
    
    def forward(self, tokens):
        """
        tokens: (batch, seq_len) - input token IDs
        Returns: (batch, max_byte_len, 257) - logits for next token's bytes
        """
        batch, seq_len = tokens.shape
        
        # Embed
        x = self.token_embed[tokens]  # (batch, seq_len, d_model)
        x = x + self.pos_embed[:seq_len]
        
        # Layers
        for layer in self.layers:
            # Residual MLP
            h = layer['ln1'].forward(x)
            h = np.maximum(0, h @ layer['W1'])  # ReLU
            h = h @ layer['W2']
            x = x + h
            
            x = layer['ln2'].forward(x)
        
        # Take last position (next-token prediction)
        h_last = x[:, -1, :]  # (batch, d_model)
        
        # Byte head
        logits = h_last @ self.byte_head  # (batch, max_byte_len * 257)
        logits = logits.reshape(batch, self.max_byte_len, 257)
        
        return logits
    
    def predict_bytes(self, tokens):
        """Predict byte sequences for next tokens"""
        logits = self.forward(tokens)
        
        sequences = []
        for i in range(tokens.shape[0]):
            preds = np.argmax(logits[i], axis=-1)
            
            # Find EOS
            eos_pos = np.where(preds == EOS)[0]
            end = eos_pos[0] if len(eos_pos) > 0 else len(preds)
            
            # Extract bytes
            byte_seq = np.clip(preds[:end], 0, 255).astype(np.uint8)
            sequences.append(bytes(byte_seq))
        
        return sequences
    
    def train_step(self, tokens, target_bytes, lr):
        """
        tokens: (batch, seq_len) - context
        target_bytes: (batch, max_byte_len) - next token's byte representation
        """
        batch = tokens.shape[0]
        
        # Forward
        logits = self.forward(tokens)  # (batch, max_byte_len, 257)
        
        # Softmax
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        
        # Loss
        loss = 0
        for pos in range(self.max_byte_len):
            target_probs = probs[np.arange(batch), pos, target_bytes[:, pos]]
            target_probs = np.clip(target_probs, 1e-10, 1.0)
            loss -= np.sum(np.log(target_probs))
        loss /= (batch * self.max_byte_len)
        
        # Gradient (simplified - only update byte head)
        grad_logits = probs.copy()
        for b in range(batch):
            for pos in range(self.max_byte_len):
                grad_logits[b, pos, target_bytes[b, pos]] -= 1
        grad_logits /= (batch * self.max_byte_len)
        
        # Backprop to byte head only (simplified training)
        h_last = self.forward(tokens)  # Re-compute (inefficient but simple)
        tokens_embed = self.token_embed[tokens]
        x = tokens_embed + self.pos_embed[:tokens.shape[1]]
        for layer in self.layers:
            h = layer['ln1'].forward(x)
            h = np.maximum(0, h @ layer['W1'])
            h = h @ layer['W2']
            x = x + h
            x = layer['ln2'].forward(x)
        h_last = x[:, -1, :]
        
        # Update byte head
        grad_head = h_last.T @ grad_logits.reshape(batch, -1)
        self.byte_head -= lr * np.clip(grad_head, -1, 1)
        
        return loss


def create_copy_task(n_examples, vocab_size, seq_len, max_byte_len, seed=42):
    """
    Simple copy task: given sequence [A, B], predict bytes of C where C comes next.
    
    For simplicity: C is deterministically derived from (A, B)
    """
    np.random.seed(seed)
    
    contexts = np.random.randint(1, vocab_size, (n_examples, seq_len))
    target_bytes = np.full((n_examples, max_byte_len), EOS, dtype=int)
    target_lengths = np.zeros(n_examples, dtype=int)
    
    for i in range(n_examples):
        # Next token derived from context
        next_token = (contexts[i, -1] + contexts[i, -2]) % vocab_size
        
        # Token's byte representation (deterministic)
        byte_len = (next_token % max_byte_len) + 1
        target_lengths[i] = byte_len
        
        for pos in range(byte_len):
            target_bytes[i, pos] = (next_token * 7 + pos * 13) % 256
    
    return contexts, target_bytes, target_lengths


def evaluate(model, contexts, target_bytes, target_lengths):
    """Evaluate model"""
    n = contexts.shape[0]
    
    pred_sequences = model.predict_bytes(contexts)
    
    correct_len = 0
    correct_exact = 0
    byte_errors = 0
    total_bytes = 0
    
    for i in range(n):
        true_len = target_lengths[i]
        true_bytes = bytes(target_bytes[i, :true_len])
        pred_bytes = pred_sequences[i]
        
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


def main():
    print("="*60)
    print("PHASE 2F: TRANSFORMER + STRUCTURED BYTE HEAD")
    print("="*60)
    print()
    
    # Config
    vocab_size = 50
    d_model = 128  # Small for speed
    n_layers = 2   # Small for speed
    seq_len = 4
    max_byte_len = 8
    
    n_train = 300
    n_test = 60
    epochs = 600
    
    print(f"Transformer: {n_layers} layers, {d_model} dim")
    print(f"Task: Copy (predict next token's bytes from context)")
    print(f"Vocab: {vocab_size}, Context: {seq_len} tokens")
    print(f"Output: Structured bytes (max {max_byte_len})")
    print()
    
    # Data
    X_train, y_train, len_train = create_copy_task(n_train, vocab_size, seq_len, max_byte_len, seed=42)
    X_test, y_test, len_test = create_copy_task(n_test, vocab_size, seq_len, max_byte_len, seed=999)
    
    # Model
    model = MiniTransformer(vocab_size, d_model, n_layers, seq_len, max_byte_len)
    
    print("Training...")
    for epoch in range(epochs + 1):
        if epoch > 0:
            idx = np.random.permutation(n_train)
            for i in range(0, n_train, 30):
                batch = idx[i:min(i+30, n_train)]
                model.train_step(X_train[batch], y_train[batch], lr=0.01)
        
        if epoch % 150 == 0:
            metrics = evaluate(model, X_test, y_test, len_test)
            print(f"  E{epoch:4d}: len={metrics['len_acc']*100:5.1f}%, "
                  f"byte={metrics['byte_acc']*100:5.1f}%, "
                  f"exact={metrics['exact_acc']*100:5.1f}%")
    
    print()
    
    # Final results
    print("="*60)
    print("RESULTS")
    print("="*60)
    print()
    
    train_metrics = evaluate(model, X_train, y_train, len_train)
    test_metrics = evaluate(model, X_test, y_test, len_test)
    
    print("Test:")
    print(f"  Length acc: {test_metrics['len_acc']*100:6.2f}%")
    print(f"  Byte acc:   {test_metrics['byte_acc']*100:6.2f}%")
    print(f"  Exact acc:  {test_metrics['exact_acc']*100:6.2f}%")
    print()
    
    # Codec test
    pred_seqs = model.predict_bytes(X_test)
    codec_ok = 0
    for i in range(min(50, n_test)):
        try:
            if len(pred_seqs[i]) > 0:
                kappa = construct_kappa(pred_seqs[i])
                decoded = algebraic_decode_raw_kronecker(kappa, 256, 32, True)
                if decoded == pred_seqs[i]:
                    codec_ok += 1
        except:
            pass
    
    print(f"Codec: {codec_ok}/{min(50, n_test)} predictions decodable")
    print()
    
    # Verdict
    print("="*60)
    print("VERDICT")
    print("="*60)
    print()
    
    if test_metrics['exact_acc'] >= 0.90:
        print("✅ TRANSFORMER + STRUCTURED BYTES WORKS")
        print("\nThe interface integrates successfully with Transformer architecture.")
        print("\nReady for Phase 2G: Three-way comparison")
    elif test_metrics['exact_acc'] >= 0.70:
        print("⚠️  MOSTLY WORKS")
        print(f"\n{test_metrics['exact_acc']*100:.1f}% exact accuracy.")
        print("Interface works but needs tuning.")
    else:
        print("⚠️  NEEDS MORE TRAINING")
        print("\nSimplified training (byte head only) may be limiting.")
        print("Full backprop through Transformer would likely improve results.")
    
    print("="*60)


if __name__ == "__main__":
    main()
