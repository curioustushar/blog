"""
Phase 2C: Synthetic Learnability Experiment

Test whether structured byte prediction is learnable BEFORE language modeling.

Critical question: Can a neural model predict discrete bytes accurately enough
for EXACT whole-token reconstruction?
"""

import sys
import os
import numpy as np
import time
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from kronecker_encoder import algebraic_decode_raw_kronecker


def construct_kronecker_from_bytes(
    byte_seq: bytes,
    char_dim: int = 256,
    pos_dim: int = 32,
    apply_length_norm: bool = True,
) -> np.ndarray:
    """Construct exact Kronecker from bytes."""
    L = min(len(byte_seq), pos_dim)
    D = char_dim * pos_dim
    kappa = np.zeros(D, dtype=np.float64)
    
    for p in range(L):
        byte_val = byte_seq[p]
        idx = byte_val * pos_dim + p
        kappa[idx] = 1.0
    
    if apply_length_norm and L > 0:
        kappa /= np.sqrt(L)
    
    return kappa


# ============================================================================
# SYNTHETIC DATASET GENERATION
# ============================================================================

def generate_synthetic_dataset(
    n_samples: int,
    min_length: int = 1,
    max_length: int = 32,
    mode: str = 'uniform',
    seed: int = 42,
) -> List[bytes]:
    """
    Generate synthetic byte sequences.
    
    Args:
        n_samples: Number of sequences to generate
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        mode: 'uniform', 'structured', or 'mixed'
        seed: Random seed
        
    Returns:
        List of byte sequences
    """
    np.random.seed(seed)
    sequences = []
    
    for _ in range(n_samples):
        # Random length
        L = np.random.randint(min_length, max_length + 1)
        
        if mode == 'uniform':
            # Uniform random bytes
            byte_seq = bytes(np.random.randint(0, 256, L))
        
        elif mode == 'structured':
            # Create structured patterns
            pattern_type = np.random.randint(0, 5)
            
            if pattern_type == 0:
                # Repeated byte
                byte_val = np.random.randint(0, 256)
                byte_seq = bytes([byte_val] * L)
            
            elif pattern_type == 1:
                # Alternating pattern
                byte_a = np.random.randint(0, 256)
                byte_b = np.random.randint(0, 256)
                byte_seq = bytes([byte_a if i % 2 == 0 else byte_b for i in range(L)])
            
            elif pattern_type == 2:
                # Sequential
                start = np.random.randint(0, 256 - L)
                byte_seq = bytes(range(start, start + L))
            
            elif pattern_type == 3:
                # Prefix + repeated suffix
                prefix_len = min(L // 2, 4)
                prefix = bytes(np.random.randint(0, 256, prefix_len))
                suffix_byte = np.random.randint(0, 256)
                suffix = bytes([suffix_byte] * (L - prefix_len))
                byte_seq = prefix + suffix
            
            else:
                # Random with bias toward ASCII
                byte_seq = bytes(np.random.randint(32, 127, L))
        
        elif mode == 'mixed':
            # Mix of uniform and structured
            if np.random.random() < 0.5:
                byte_seq = bytes(np.random.randint(0, 256, L))
            else:
                # Use structured generation
                sequences_temp = generate_synthetic_dataset(1, L, L, mode='structured', seed=None)
                byte_seq = sequences_temp[0]
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        sequences.append(byte_seq)
    
    return sequences


def create_length_controlled_datasets(
    n_per_length: int = 1000,
    lengths: List[int] = [1, 2, 4, 8, 16, 32],
    seed: int = 42,
) -> Dict[int, List[bytes]]:
    """Create test sets with controlled lengths."""
    datasets = {}
    
    for L in lengths:
        datasets[L] = generate_synthetic_dataset(
            n_samples=n_per_length,
            min_length=L,
            max_length=L,
            mode='uniform',
            seed=seed + L,
        )
    
    return datasets


# ============================================================================
# SIMPLE MLP MODEL
# ============================================================================

class SimpleMLP:
    """
    Simple MLP for testing structured byte prediction.
    
    Input: Random embedding (simulates hidden state)
    Output: Byte predictions at each position
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        max_length: int = 32,
        char_dim: int = 256,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.char_dim = char_dim
        
        # MLP layers
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        
        # Output heads: one per position
        self.byte_heads = []
        for _ in range(max_length):
            W = np.random.randn(hidden_dim, char_dim) * np.sqrt(2.0 / hidden_dim)
            b = np.zeros(char_dim)
            self.byte_heads.append((W, b))
        
        # Length predictor
        self.length_W = np.random.randn(hidden_dim, max_length + 1) * np.sqrt(2.0 / hidden_dim)
        self.length_b = np.zeros(max_length + 1)
        
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.
        
        Args:
            x: (batch, input_dim)
            
        Returns:
            byte_logits: (batch, max_length, char_dim)
            length_logits: (batch, max_length + 1)
        """
        # MLP
        h1 = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU
        
        # Byte predictions
        batch_size = x.shape[0]
        byte_logits = np.zeros((batch_size, self.max_length, self.char_dim))
        
        for pos in range(self.max_length):
            W, b = self.byte_heads[pos]
            byte_logits[:, pos, :] = h2 @ W + b
        
        # Length prediction
        length_logits = h2 @ self.length_W + self.length_b
        
        return byte_logits, length_logits
    
    def predict(self, x: np.ndarray) -> Tuple[List[bytes], List[int]]:
        """Predict byte sequences."""
        byte_logits, length_logits = self.forward(x)
        batch_size = x.shape[0]
        
        bytes_pred = np.argmax(byte_logits, axis=-1)
        lengths_pred = np.argmax(length_logits, axis=-1)
        
        sequences = []
        lengths = []
        
        for i in range(batch_size):
            L = lengths_pred[i]
            if L >= self.max_length:
                sequences.append(b'')
                lengths.append(0)
            else:
                byte_seq = bytes(bytes_pred[i, :L+1])
                sequences.append(byte_seq)
                lengths.append(L + 1)
        
        return sequences, lengths
    
    def backward_and_update(
        self,
        x: np.ndarray,
        targets: List[bytes],
        lr: float,
    ) -> float:
        """
        Backpropagation and SGD update.
        
        Returns loss value.
        """
        batch_size = x.shape[0]
        
        # Forward pass (save activations)
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        
        byte_logits = np.zeros((batch_size, self.max_length, self.char_dim))
        for pos in range(self.max_length):
            W, b = self.byte_heads[pos]
            byte_logits[:, pos, :] = h2 @ W + b
        
        length_logits = h2 @ self.length_W + self.length_b
        
        # Compute loss
        total_loss = 0.0
        
        # Initialize gradients
        d_h2 = np.zeros_like(h2)
        
        # Byte loss and gradients
        for pos in range(self.max_length):
            logits_pos = byte_logits[:, pos, :]
            targets_pos = np.zeros(batch_size, dtype=np.int64)
            
            for j, target in enumerate(targets):
                if pos < len(target):
                    targets_pos[j] = target[pos]
            
            probs = self.softmax(logits_pos)
            correct_probs = probs[np.arange(batch_size), targets_pos]
            correct_probs = np.clip(correct_probs, 1e-10, 1.0)
            total_loss += -np.mean(np.log(correct_probs))
            
            # Gradient
            d_logits = probs.copy()
            d_logits[np.arange(batch_size), targets_pos] -= 1
            d_logits /= (batch_size * self.max_length)
            
            # Update byte head
            W, b = self.byte_heads[pos]
            W -= lr * (h2.T @ d_logits)
            b -= lr * np.sum(d_logits, axis=0)
            self.byte_heads[pos] = (W, b)
            
            # Backprop to h2
            d_h2 += d_logits @ W.T
        
        # Length loss and gradients
        target_lengths = np.array([min(len(t), self.max_length) - 1 if len(t) > 0 else self.max_length for t in targets])
        
        probs_length = self.softmax(length_logits)
        correct_probs_length = probs_length[np.arange(batch_size), target_lengths]
        correct_probs_length = np.clip(correct_probs_length, 1e-10, 1.0)
        total_loss += -np.mean(np.log(correct_probs_length))
        
        d_length_logits = probs_length.copy()
        d_length_logits[np.arange(batch_size), target_lengths] -= 1
        d_length_logits /= batch_size
        
        # Update length head
        self.length_W -= lr * (h2.T @ d_length_logits)
        self.length_b -= lr * np.sum(d_length_logits, axis=0)
        
        # Backprop to h2
        d_h2 += d_length_logits @ self.length_W.T
        
        # Backprop through ReLU
        d_h2 = d_h2 * (h2 > 0)
        
        # Update W2, b2
        self.W2 -= lr * (h1.T @ d_h2)
        self.b2 -= lr * np.sum(d_h2, axis=0)
        
        # Backprop to h1
        d_h1 = d_h2 @ self.W2.T
        d_h1 = d_h1 * (h1 > 0)
        
        # Update W1, b1
        self.W1 -= lr * (x.T @ d_h1)
        self.b1 -= lr * np.sum(d_h1, axis=0)
        
        return total_loss / (self.max_length + 1)
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def evaluate_exact_reconstruction(
    model: SimpleMLP,
    inputs: np.ndarray,
    targets: List[bytes],
) -> Dict[str, float]:
    """
    Evaluate model with EXACT metrics.
    
    Critical: Report exact sequence accuracy, not just byte accuracy.
    """
    predictions, pred_lengths = model.predict(inputs)
    
    n_samples = len(targets)
    
    # Byte-level metrics
    total_bytes = 0
    correct_bytes = 0
    
    # Sequence-level metrics
    exact_match = 0
    length_correct = 0
    
    # Kronecker decode verification
    decode_success = 0
    decode_match = 0
    
    for i in range(n_samples):
        target = targets[i]
        pred = predictions[i]
        
        # Length accuracy
        if len(pred) == len(target):
            length_correct += 1
        
        # Byte accuracy
        min_len = min(len(pred), len(target))
        for j in range(min_len):
            total_bytes += 1
            if pred[j] == target[j]:
                correct_bytes += 1
        
        # Pad shorter sequence for fair comparison
        if len(pred) < len(target):
            total_bytes += len(target) - len(pred)
        elif len(target) < len(pred):
            total_bytes += len(pred) - len(target)
        
        # Exact match
        if pred == target:
            exact_match += 1
        
        # Verify Kronecker decode
        if len(pred) > 0:
            kappa = construct_kronecker_from_bytes(pred)
            decoded = algebraic_decode_raw_kronecker(kappa, 256, 32, True)
            
            if decoded is not None:
                decode_success += 1
                if decoded == pred:
                    decode_match += 1
    
    return {
        'byte_accuracy': correct_bytes / max(total_bytes, 1),
        'length_accuracy': length_correct / n_samples,
        'exact_sequence_accuracy': exact_match / n_samples,
        'decode_success_rate': decode_success / n_samples,
        'decode_match_rate': decode_match / max(decode_success, 1),
    }


def train_synthetic_experiment(
    n_train: int = 10000,
    n_val: int = 1000,
    n_test: int = 1000,
    input_dim: int = 64,
    hidden_dim: int = 128,
    n_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    verbose: bool = True,
) -> Dict:
    """
    Main synthetic learnability experiment.
    
    Tests: Can we learn to predict bytes accurately enough for exact reconstruction?
    """
    if verbose:
        print("=" * 80)
        print("PHASE 2C: SYNTHETIC LEARNABILITY EXPERIMENT")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Train: {n_train:,} sequences")
        print(f"  Val:   {n_val:,} sequences")
        print(f"  Test:  {n_test:,} sequences")
        print(f"  Model: MLP({input_dim} → {hidden_dim} → byte predictions)")
        print(f"  Training: {n_epochs} epochs, batch={batch_size}, lr={lr}")
    
    # Generate datasets
    if verbose:
        print(f"\nGenerating synthetic data...")
    
    train_targets = generate_synthetic_dataset(n_train, mode='mixed', seed=42)
    val_targets = generate_synthetic_dataset(n_val, mode='mixed', seed=43)
    test_targets = generate_synthetic_dataset(n_test, mode='mixed', seed=44)
    
    # Create unique input embeddings for each sequence
    # (In real LM, this would be Transformer hidden states)
    train_inputs = np.random.randn(n_train, input_dim) * 0.1
    val_inputs = np.random.randn(n_val, input_dim) * 0.1
    test_inputs = np.random.randn(n_test, input_dim) * 0.1
    
    # Create model
    model = SimpleMLP(input_dim, hidden_dim)
    
    if verbose:
        print(f"\nInitial evaluation (random model):")
    
    init_metrics = evaluate_exact_reconstruction(model, val_inputs[:100], val_targets[:100])
    if verbose:
        print(f"  Byte accuracy:     {init_metrics['byte_accuracy']*100:5.2f}%")
        print(f"  Exact sequence:    {init_metrics['exact_sequence_accuracy']*100:5.2f}%")
        print(f"  Decode success:    {init_metrics['decode_success_rate']*100:5.2f}%")
    
    # Training loop
    if verbose:
        print(f"\nTraining...")
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Shuffle
        indices = np.random.permutation(n_train)
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_inputs = train_inputs[batch_idx]
            batch_targets = [train_targets[j] for j in batch_idx]
            
            # Forward + backward + update
            loss = model.backward_and_update(batch_inputs, batch_targets, lr)
            
            epoch_loss += loss
            n_batches += 1
        
        epoch_loss /= n_batches
        
        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            # Evaluate on validation
            val_metrics = evaluate_exact_reconstruction(model, val_inputs, val_targets)
            
            print(f"  Epoch {epoch:2d}: loss={epoch_loss:.4f}, "
                  f"byte_acc={val_metrics['byte_accuracy']*100:5.2f}%, "
                  f"exact={val_metrics['exact_sequence_accuracy']*100:5.2f}%")
    
    train_time = time.time() - start_time
    
    # Final evaluation
    if verbose:
        print(f"\nFinal evaluation:")
    
    test_metrics = evaluate_exact_reconstruction(model, test_inputs, test_targets)
    
    if verbose:
        print(f"  Byte accuracy:         {test_metrics['byte_accuracy']*100:6.2f}%")
        print(f"  Length accuracy:       {test_metrics['length_accuracy']*100:6.2f}%")
        print(f"  Exact sequence:        {test_metrics['exact_sequence_accuracy']*100:6.2f}%")
        print(f"  Decode success:        {test_metrics['decode_success_rate']*100:6.2f}%")
        print(f"  Decode match:          {test_metrics['decode_match_rate']*100:6.2f}%")
        print(f"  Training time:         {train_time:.1f}s")
    
    return {
        'model': model,
        'test_metrics': test_metrics,
        'train_time': train_time,
    }


if __name__ == "__main__":
    # Run experiment
    results = train_synthetic_experiment(
        n_train=10000,
        n_val=1000,
        n_test=1000,
        n_epochs=50,
        verbose=True,
    )
    
    test_metrics = results['test_metrics']
    
    # Decision point
    print("\n" + "=" * 80)
    print("DECISION POINT")
    print("=" * 80)
    
    exact_acc = test_metrics['exact_sequence_accuracy']
    byte_acc = test_metrics['byte_accuracy']
    
    print(f"\nExact sequence accuracy: {exact_acc*100:.2f}%")
    print(f"Byte accuracy: {byte_acc*100:.2f}%")
    
    if exact_acc >= 0.999:
        print("\n✅ SUCCESS: Exact reconstruction ≥99.9%")
        print("   Structured byte prediction is LEARNABLE")
        print("   → Proceed to tiny Transformer experiments")
    
    elif exact_acc >= 0.90 and byte_acc >= 0.98:
        print("\n⚠️  PROMISING: High byte accuracy but lower exact reconstruction")
        print(f"   Likely cause: p^L degradation (0.98^16 = {0.98**16:.3f})")
        print("   → Investigate: autoregressive decoding, error accumulation")
        print("   → May still proceed to LM with caution")
    
    elif byte_acc >= 0.80:
        print("\n⚠️  PARTIAL: Byte prediction works but not accurate enough")
        print("   → Investigate: model capacity, loss function, training dynamics")
        print("   → Do NOT proceed to LM yet")
    
    else:
        print("\n❌ FAILURE: Structured byte prediction is not learnable")
        print("   → Reconsider architecture")
        print("   → May need different output representation")
    
    print("=" * 80)
