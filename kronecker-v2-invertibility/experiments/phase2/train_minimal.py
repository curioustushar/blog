"""
Minimal training script for Phase 2 experiments.

This is a proof-of-concept to test whether h→κ regression works.
Uses a tiny synthetic dataset for fast iteration.
"""

import sys
import os
import time
from typing import Dict, List, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from kronecker_encoder import KroneckerEncoderStaged, algebraic_decode_raw_kronecker


class SimpleLM:
    """
    Extremely simplified language model for proof-of-concept.
    
    This is NOT a real Transformer - it's a minimal test to see if
    we can learn h→κ mapping without vocabulary-sized parameters.
    """
    
    def __init__(self, vocab_size: int, d_model: int, output_type: str = 'softmax'):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Hidden dimension
            output_type: 'softmax' or 'kronecker'
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.output_type = output_type
        
        # Embedding layer
        self.embed = np.random.randn(vocab_size, d_model) * 0.01
        
        # Simple "Transformer" (just one linear layer for speed)
        self.W_hidden = np.random.randn(d_model, d_model) * 0.01
        self.b_hidden = np.zeros(d_model)
        
        # Output head
        if output_type == 'softmax':
            self.W_out = np.random.randn(d_model, vocab_size) * 0.01
            self.b_out = np.zeros(vocab_size)
        elif output_type == 'kronecker':
            self.char_dim = 256
            self.pos_dim = 32
            self.D = self.char_dim * self.pos_dim
            self.W_out = np.random.randn(d_model, self.D) * 0.01
            self.b_out = np.zeros(self.D)
            
            # Kronecker encoder for targets
            self.encoder = KroneckerEncoderStaged(
                char_dim=self.char_dim,
                pos_dim=self.pos_dim,
                apply_length_norm=True,
                apply_z_norm=False,
                apply_projection=False,
            )
        
        # Store all parameters for gradient descent
        if output_type == 'softmax':
            self.params = {
                'embed': self.embed,
                'W_hidden': self.W_hidden,
                'b_hidden': self.b_hidden,
                'W_out': self.W_out,
                'b_out': self.b_out,
            }
        else:
            self.params = {
                'embed': self.embed,
                'W_hidden': self.W_hidden,
                'b_hidden': self.b_hidden,
                'W_out': self.W_out,
                'b_out': self.b_out,
            }
        
        # Gradients
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        
    def forward(self, token_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.
        
        Args:
            token_ids: (batch_size,) array of token IDs
            
        Returns:
            output: (batch_size, vocab_size) or (batch_size, D)
            hidden: (batch_size, d_model) hidden states
        """
        # Embedding
        x = self.embed[token_ids]  # (batch, d_model)
        
        # Simple "Transformer" (just one layer)
        h = np.tanh(x @ self.W_hidden + self.b_hidden)  # (batch, d_model)
        
        # Output projection
        out = h @ self.W_out + self.b_out  # (batch, vocab_size) or (batch, D)
        
        return out, h
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def cross_entropy_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            logits: (batch, vocab_size)
            targets: (batch,) target token IDs
            
        Returns:
            loss: scalar
        """
        batch_size = logits.shape[0]
        probs = self.softmax(logits)
        
        # Get probabilities of correct tokens
        correct_probs = probs[np.arange(batch_size), targets]
        
        # Avoid log(0)
        correct_probs = np.clip(correct_probs, 1e-10, 1.0)
        
        loss = -np.mean(np.log(correct_probs))
        return loss
    
    def mse_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        """MSE loss for Kronecker regression."""
        return np.mean((pred - target) ** 2)
    
    def backward_softmax(self, logits: np.ndarray, targets: np.ndarray, hidden: np.ndarray, token_ids: np.ndarray):
        """Backward pass for softmax output."""
        batch_size = logits.shape[0]
        
        # Gradient through softmax + cross-entropy
        probs = self.softmax(logits)
        d_logits = probs.copy()
        d_logits[np.arange(batch_size), targets] -= 1
        d_logits /= batch_size
        
        # Gradient of output layer
        self.grads['W_out'] = hidden.T @ d_logits
        self.grads['b_out'] = np.sum(d_logits, axis=0)
        
        # Backprop to hidden
        d_hidden = d_logits @ self.W_out.T
        
        # Gradient through tanh
        d_hidden = d_hidden * (1 - hidden ** 2)
        
        # Gradient of hidden layer
        x = self.embed[token_ids]
        self.grads['W_hidden'] = x.T @ d_hidden
        self.grads['b_hidden'] = np.sum(d_hidden, axis=0)
        
        # Gradient of embedding
        d_embed = d_hidden @ self.W_hidden.T
        self.grads['embed'] = np.zeros_like(self.embed)
        for i, tid in enumerate(token_ids):
            self.grads['embed'][tid] += d_embed[i]
    
    def backward_kronecker(self, pred_kappa: np.ndarray, target_kappa: np.ndarray, hidden: np.ndarray, token_ids: np.ndarray):
        """Backward pass for Kronecker regression."""
        batch_size = pred_kappa.shape[0]
        
        # Gradient of MSE
        d_pred = 2 * (pred_kappa - target_kappa) / batch_size
        
        # Gradient of output layer
        self.grads['W_out'] = hidden.T @ d_pred
        self.grads['b_out'] = np.sum(d_pred, axis=0)
        
        # Backprop to hidden
        d_hidden = d_pred @ self.W_out.T
        
        # Gradient through tanh
        d_hidden = d_hidden * (1 - hidden ** 2)
        
        # Gradient of hidden layer
        x = self.embed[token_ids]
        self.grads['W_hidden'] = x.T @ d_hidden
        self.grads['b_hidden'] = np.sum(d_hidden, axis=0)
        
        # Gradient of embedding
        d_embed = d_hidden @ self.W_hidden.T
        self.grads['embed'] = np.zeros_like(self.embed)
        for i, tid in enumerate(token_ids):
            self.grads['embed'][tid] += d_embed[i]
    
    def sgd_step(self, lr: float):
        """Simple SGD update."""
        for name, param in self.params.items():
            param -= lr * self.grads[name]
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.size for p in self.params.values())


def create_simple_vocab(size: int = 100) -> Tuple[List[str], Dict[str, int]]:
    """
    Create a simple synthetic vocabulary.
    
    Args:
        size: Number of tokens
        
    Returns:
        vocab: List of token strings
        token_to_id: Dictionary mapping tokens to IDs
    """
    # Common words + synthetic tokens
    common = ["the", "a", "is", "to", "of", "and", "in", "it", "you", "that",
              "was", "for", "on", "are", "with", "as", "be", "at", "one", "have"]
    
    vocab = common.copy()
    
    # Add more tokens
    for i in range(size - len(common)):
        vocab.append(f"word{i}")
    
    token_to_id = {token: i for i, token in enumerate(vocab)}
    
    return vocab, token_to_id


def train_and_evaluate(
    model_type: str,
    vocab_size: int = 100,
    d_model: int = 64,
    n_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.01,
    verbose: bool = True,
) -> Dict:
    """
    Train a simple model and evaluate.
    
    Args:
        model_type: 'softmax' or 'kronecker'
        vocab_size: Size of vocabulary
        d_model: Hidden dimension
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        verbose: Print progress
        
    Returns:
        results: Dictionary of metrics
    """
    # Create vocabulary
    vocab, token_to_id = create_simple_vocab(vocab_size)
    
    # Create model
    model = SimpleLM(vocab_size, d_model, output_type=model_type)
    
    if verbose:
        print(f"\nTraining {model_type} model:")
        print(f"  Vocabulary: {vocab_size}")
        print(f"  Hidden dim: {d_model}")
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Output head params: {model.W_out.size:,}")
    
    # Generate training data (next-token prediction)
    # Just use random sequences for proof-of-concept
    n_train = 1000
    train_inputs = np.random.randint(0, vocab_size, n_train)
    train_targets = np.random.randint(0, vocab_size, n_train)
    
    # Prepare Kronecker targets if needed
    if model_type == 'kronecker':
        target_kappas = []
        for tid in train_targets:
            token = vocab[tid]
            stages = model.encoder.encode_all_stages(token)
            kappa = stages['stage2_kronecker']
            target_kappas.append(kappa)
        target_kappas = np.array(target_kappas)
    
    # Training loop
    losses = []
    start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        # Mini-batch training
        indices = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_inputs = train_inputs[batch_idx]
            batch_targets = train_targets[batch_idx]
            
            # Forward pass
            outputs, hidden = model.forward(batch_inputs)
            
            # Compute loss and backward
            if model_type == 'softmax':
                loss = model.cross_entropy_loss(outputs, batch_targets)
                model.backward_softmax(outputs, batch_targets, hidden, batch_inputs)
            else:
                batch_target_kappas = target_kappas[batch_idx]
                loss = model.mse_loss(outputs, batch_target_kappas)
                model.backward_kronecker(outputs, batch_target_kappas, hidden, batch_inputs)
            
            # SGD step
            model.sgd_step(lr)
            
            epoch_loss += loss
        
        epoch_loss /= (n_train / batch_size)
        losses.append(epoch_loss)
        
        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:3d}: loss = {epoch_loss:.4f}")
    
    train_time = time.time() - start_time
    
    # Evaluation
    if model_type == 'softmax':
        # Accuracy: argmax prediction
        outputs, _ = model.forward(train_inputs)
        predictions = np.argmax(outputs, axis=-1)
        accuracy = np.mean(predictions == train_targets)
        
        results = {
            'model_type': model_type,
            'parameters': model.count_parameters(),
            'output_params': model.W_out.size,
            'final_loss': losses[-1],
            'accuracy': accuracy,
            'train_time': train_time,
        }
        
        if verbose:
            print(f"\n  Final loss: {losses[-1]:.4f}")
            print(f"  Accuracy: {accuracy*100:.1f}%")
            print(f"  Train time: {train_time:.2f}s")
    
    else:  # kronecker
        # Test algebraic decoder
        outputs, _ = model.forward(train_inputs[:100])  # Test on subset
        
        correct = 0
        for i, tid in enumerate(train_targets[:100]):
            kappa_pred = outputs[i]
            token_bytes = algebraic_decode_raw_kronecker(
                kappa_pred,
                char_dim=model.char_dim,
                pos_dim=model.pos_dim,
                length_normalized=True,
            )
            
            if token_bytes is not None:
                try:
                    token_pred = token_bytes.decode('utf-8', errors='replace')
                    if token_pred == vocab[tid]:
                        correct += 1
                except:
                    pass
        
        decode_accuracy = correct / 100.0
        
        results = {
            'model_type': model_type,
            'parameters': model.count_parameters(),
            'output_params': model.W_out.size,
            'final_loss': losses[-1],
            'decode_accuracy': decode_accuracy,
            'train_time': train_time,
        }
        
        if verbose:
            print(f"\n  Final MSE: {losses[-1]:.6f}")
            print(f"  Decode accuracy: {decode_accuracy*100:.1f}%")
            print(f"  Train time: {train_time:.2f}s")
    
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 2 MINIMAL TRAINING EXPERIMENT")
    print("=" * 80)
    print("\nObjective: Test if h→κ regression can learn without vocab-sized parameters\n")
    
    # Experiment configuration
    VOCAB_SIZE = 100
    D_MODEL = 128
    N_EPOCHS = 100
    BATCH_SIZE = 32
    LR = 0.05
    
    # Train both models
    results_softmax = train_and_evaluate(
        model_type='softmax',
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        verbose=True,
    )
    
    print("\n" + "=" * 80 + "\n")
    
    results_kronecker = train_and_evaluate(
        model_type='kronecker',
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR * 0.1,  # Lower LR for regression
        verbose=True,
    )
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Metric':<25} {'Softmax':>15} {'Kronecker':>15}")
    print("-" * 80)
    print(f"{'Parameters':<25} {results_softmax['parameters']:>15,} {results_kronecker['parameters']:>15,}")
    print(f"{'Output head params':<25} {results_softmax['output_params']:>15,} {results_kronecker['output_params']:>15,}")
    print(f"{'Final loss':<25} {results_softmax['final_loss']:>15.4f} {results_kronecker['final_loss']:>15.6f}")
    
    if 'accuracy' in results_softmax:
        print(f"{'Accuracy':<25} {results_softmax['accuracy']*100:>14.1f}% {'N/A':>15}")
    
    if 'decode_accuracy' in results_kronecker:
        print(f"{'Decode accuracy':<25} {'N/A':>15} {results_kronecker['decode_accuracy']*100:>14.1f}%")
    
    print(f"{'Training time':<25} {results_softmax['train_time']:>14.2f}s {results_kronecker['train_time']:>14.2f}s")
    
    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    if 'decode_accuracy' in results_kronecker:
        decode_acc = results_kronecker['decode_accuracy']
        param_ratio = results_softmax['output_params'] / results_kronecker['output_params']
        
        print(f"\nKronecker regression decode accuracy: {decode_acc*100:.1f}%")
        print(f"Parameter reduction: {param_ratio:.1f}× ({results_softmax['output_params']:,} → {results_kronecker['output_params']:,})")
        
        if decode_acc > 0.5:
            print("\n✅ PROOF-OF-CONCEPT SUCCESSFUL")
            print("   The model CAN learn h→κ mapping without vocabulary-sized parameters!")
            print("   This suggests the V2 hypothesis is promising.")
        elif decode_acc > 0.1:
            print("\n⚠️  PARTIAL SUCCESS")
            print("   Some learning occurred, but accuracy is low.")
            print("   May need: better architecture, more training, or different loss function.")
        else:
            print("\n❌ HYPOTHESIS REJECTED")
            print("   The model cannot learn meaningful h→κ mapping.")
            print("   The representation may be too complex for regression.")
    
    print("=" * 80)
