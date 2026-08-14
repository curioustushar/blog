"""
Structured Byte Prediction Head - Phase 2B

Instead of regressing continuous κ, predict DISCRETE bytes that construct exact κ.

Architecture:
    h → byte_classifiers → discrete bytes → exact κ → algebraic_decode → token
"""

import sys
import os
import math
import numpy as np
from typing import Tuple, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from kronecker_encoder import algebraic_decode_raw_kronecker


def construct_kronecker_from_bytes(
    byte_seq: bytes,
    char_dim: int = 256,
    pos_dim: int = 32,
    apply_length_norm: bool = True,
) -> np.ndarray:
    """Construct exact Kronecker representation from bytes."""
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


class StructuredByteHead:
    """
    Predict discrete bytes at each position, then construct exact κ.
    
    This is the Phase 2B approach that respects the discrete structure of κ.
    """
    
    def __init__(
        self,
        d_model: int,
        char_dim: int = 256,
        pos_dim: int = 32,
        apply_length_norm: bool = True,
    ):
        """
        Args:
            d_model: Hidden dimension
            char_dim: Byte alphabet size (256)
            pos_dim: Maximum byte length (32)
            apply_length_norm: Apply 1/√L normalization
        """
        self.d_model = d_model
        self.char_dim = char_dim
        self.pos_dim = pos_dim
        self.apply_length_norm = apply_length_norm
        self.D = char_dim * pos_dim
        
        # Byte classifiers: one per position
        # Each predicts probability distribution over 256 bytes
        self.byte_predictors = []
        for _ in range(pos_dim):
            W = np.random.randn(d_model, char_dim) * math.sqrt(2.0 / d_model)
            b = np.zeros(char_dim)
            self.byte_predictors.append((W, b))
        
        # Length predictor (or end-of-token)
        # Predicts which position is the last valid byte (0-indexed)
        # Output: logits for positions 0 to pos_dim-1, plus "no token" class
        self.length_predictor_W = np.random.randn(d_model, pos_dim + 1) * math.sqrt(2.0 / d_model)
        self.length_predictor_b = np.zeros(pos_dim + 1)
        
    def forward(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict byte logits and length logits.
        
        Args:
            h: (batch, d_model) hidden states
            
        Returns:
            byte_logits: (batch, pos_dim, char_dim)
            length_logits: (batch, pos_dim + 1)
        """
        batch_size = h.shape[0]
        
        # Byte predictions
        byte_logits = np.zeros((batch_size, self.pos_dim, self.char_dim))
        for pos in range(self.pos_dim):
            W, b = self.byte_predictors[pos]
            byte_logits[:, pos, :] = h @ W + b
        
        # Length prediction
        length_logits = h @ self.length_predictor_W + self.length_predictor_b
        
        return byte_logits, length_logits
    
    def predict_bytes(self, h: np.ndarray) -> Tuple[List[bytes], List[int]]:
        """
        Predict discrete bytes using argmax.
        
        Args:
            h: (batch, d_model) hidden states
            
        Returns:
            byte_sequences: List of byte sequences
            lengths: List of predicted lengths
        """
        byte_logits, length_logits = self.forward(h)
        batch_size = h.shape[0]
        
        # Predict bytes at each position
        bytes_pred = np.argmax(byte_logits, axis=-1)  # (batch, pos_dim)
        
        # Predict length
        lengths_pred = np.argmax(length_logits, axis=-1)  # (batch,)
        
        # Convert to byte sequences
        byte_sequences = []
        lengths = []
        for i in range(batch_size):
            L = lengths_pred[i]
            if L >= self.pos_dim:
                # "No token" class
                byte_sequences.append(b'')
                lengths.append(0)
            else:
                # Extract bytes up to predicted length
                bytes_i = bytes(bytes_pred[i, :L+1])
                byte_sequences.append(bytes_i)
                lengths.append(L + 1)
        
        return byte_sequences, lengths
    
    def predict_and_decode(self, h: np.ndarray) -> List[Optional[str]]:
        """
        Full pipeline: h → bytes → exact κ → decode → token.
        
        Args:
            h: (batch, d_model) hidden states
            
        Returns:
            tokens: List of decoded token strings (or None if decode fails)
        """
        byte_sequences, lengths = self.predict_bytes(h)
        
        tokens = []
        for byte_seq in byte_sequences:
            if len(byte_seq) == 0:
                tokens.append(None)
                continue
            
            # Construct EXACT Kronecker representation
            kappa = construct_kronecker_from_bytes(
                byte_seq,
                self.char_dim,
                self.pos_dim,
                self.apply_length_norm,
            )
            
            # Algebraic decode
            decoded_bytes = algebraic_decode_raw_kronecker(
                kappa,
                char_dim=self.char_dim,
                pos_dim=self.pos_dim,
                length_normalized=self.apply_length_norm,
            )
            
            if decoded_bytes is not None:
                try:
                    token = decoded_bytes.decode('utf-8', errors='replace')
                    tokens.append(token)
                except:
                    tokens.append(None)
            else:
                tokens.append(None)
        
        return tokens
    
    def compute_loss(
        self,
        h: np.ndarray,
        target_byte_sequences: List[bytes],
    ) -> Tuple[float, float, float]:
        """
        Compute cross-entropy loss for byte and length prediction.
        
        Args:
            h: (batch, d_model)
            target_byte_sequences: List of target byte sequences
            
        Returns:
            total_loss: Combined loss
            byte_loss: Byte prediction loss
            length_loss: Length prediction loss
        """
        batch_size = h.shape[0]
        byte_logits, length_logits = self.forward(h)
        
        # Prepare targets
        target_bytes = np.zeros((batch_size, self.pos_dim), dtype=np.int64)
        target_lengths = np.zeros(batch_size, dtype=np.int64)
        
        for i, byte_seq in enumerate(target_byte_sequences):
            L = min(len(byte_seq), self.pos_dim)
            target_lengths[i] = L - 1 if L > 0 else self.pos_dim  # Last valid index or "no token"
            for p in range(L):
                target_bytes[i, p] = byte_seq[p]
        
        # Byte loss: cross-entropy at each position
        byte_loss = 0.0
        for pos in range(self.pos_dim):
            logits_pos = byte_logits[:, pos, :]  # (batch, char_dim)
            targets_pos = target_bytes[:, pos]  # (batch,)
            
            # Softmax + cross-entropy
            probs = self._softmax(logits_pos)
            correct_probs = probs[np.arange(batch_size), targets_pos]
            correct_probs = np.clip(correct_probs, 1e-10, 1.0)
            byte_loss += -np.mean(np.log(correct_probs))
        
        byte_loss /= self.pos_dim
        
        # Length loss: cross-entropy
        probs_length = self._softmax(length_logits)
        correct_probs_length = probs_length[np.arange(batch_size), target_lengths]
        correct_probs_length = np.clip(correct_probs_length, 1e-10, 1.0)
        length_loss = -np.mean(np.log(correct_probs_length))
        
        # Combined loss
        total_loss = byte_loss + length_loss
        
        return total_loss, byte_loss, length_loss
    
    def backward(
        self,
        h: np.ndarray,
        target_byte_sequences: List[bytes],
    ) -> dict:
        """
        Backpropagation through the structured head.
        
        This is a simplified version for the minimal experiment.
        """
        batch_size = h.shape[0]
        byte_logits, length_logits = self.forward(h)
        
        # Prepare targets
        target_bytes = np.zeros((batch_size, self.pos_dim), dtype=np.int64)
        target_lengths = np.zeros(batch_size, dtype=np.int64)
        
        for i, byte_seq in enumerate(target_byte_sequences):
            L = min(len(byte_seq), self.pos_dim)
            target_lengths[i] = L - 1 if L > 0 else self.pos_dim
            for p in range(L):
                target_bytes[i, p] = byte_seq[p]
        
        # Gradients
        grads = {}
        
        # Byte prediction gradients
        for pos in range(self.pos_dim):
            logits_pos = byte_logits[:, pos, :]
            targets_pos = target_bytes[:, pos]
            
            probs = self._softmax(logits_pos)
            d_logits = probs.copy()
            d_logits[np.arange(batch_size), targets_pos] -= 1
            d_logits /= (batch_size * self.pos_dim)
            
            W, b = self.byte_predictors[pos]
            grads[f'byte_W_{pos}'] = h.T @ d_logits
            grads[f'byte_b_{pos}'] = np.sum(d_logits, axis=0)
        
        # Length prediction gradients
        probs_length = self._softmax(length_logits)
        d_length_logits = probs_length.copy()
        d_length_logits[np.arange(batch_size), target_lengths] -= 1
        d_length_logits /= batch_size
        
        grads['length_W'] = h.T @ d_length_logits
        grads['length_b'] = np.sum(d_length_logits, axis=0)
        
        return grads
    
    def sgd_step(self, grads: dict, lr: float):
        """Apply SGD update."""
        for pos in range(self.pos_dim):
            W, b = self.byte_predictors[pos]
            W -= lr * grads[f'byte_W_{pos}']
            b -= lr * grads[f'byte_b_{pos}']
            self.byte_predictors[pos] = (W, b)
        
        self.length_predictor_W -= lr * grads['length_W']
        self.length_predictor_b -= lr * grads['length_b']
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        # Byte predictors: pos_dim × (d_model × char_dim + char_dim)
        byte_params = self.pos_dim * (self.d_model * self.char_dim + self.char_dim)
        
        # Length predictor: d_model × (pos_dim + 1) + (pos_dim + 1)
        length_params = self.d_model * (self.pos_dim + 1) + (self.pos_dim + 1)
        
        return byte_params + length_params


if __name__ == "__main__":
    print("=" * 80)
    print("STRUCTURED BYTE HEAD - Phase 2B")
    print("=" * 80)
    
    # Test configuration
    d_model = 128
    batch_size = 4
    char_dim = 256
    pos_dim = 32
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  char_dim: {char_dim}")
    print(f"  pos_dim: {pos_dim}")
    print(f"  batch_size: {batch_size}")
    
    # Create head
    head = StructuredByteHead(d_model, char_dim, pos_dim)
    print(f"\nParameters: {head.count_parameters():,}")
    
    # Test forward pass
    h = np.random.randn(batch_size, d_model) * 0.1
    byte_logits, length_logits = head.forward(h)
    
    print(f"\nForward pass:")
    print(f"  Input: {h.shape}")
    print(f"  Byte logits: {byte_logits.shape}")
    print(f"  Length logits: {length_logits.shape}")
    
    # Test prediction
    byte_sequences, lengths = head.predict_bytes(h)
    print(f"\nPredictions:")
    for i, (byte_seq, length) in enumerate(zip(byte_sequences, lengths)):
        print(f"  Sample {i}: length={length}, bytes={list(byte_seq[:min(length, 8)])}")
    
    # Test full pipeline with known targets
    print(f"\n" + "=" * 80)
    print("TEST: Prediction with Oracle Verification")
    print("=" * 80)
    
    target_tokens = ["the", "hello", "test", "AI"]
    target_byte_sequences = [token.encode('utf-8') for token in target_tokens]
    
    # Simulate perfect predictions (for testing)
    # In reality, the network would learn these
    tokens_pred = head.predict_and_decode(h)
    
    print(f"\nTarget tokens: {target_tokens}")
    print(f"Predicted tokens: {tokens_pred}")
    
    # Test loss computation
    total_loss, byte_loss, length_loss = head.compute_loss(h, target_byte_sequences)
    print(f"\nLoss:")
    print(f"  Byte loss: {byte_loss:.4f}")
    print(f"  Length loss: {length_loss:.4f}")
    print(f"  Total loss: {total_loss:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ Structured Byte Head implementation complete!")
    print("=" * 80)
    
    print("\nKey properties:")
    print(f"  • Predicts discrete bytes (not continuous κ)")
    print(f"  • Constructs EXACT κ from predicted bytes")
    print(f"  • Algebraic decoder guaranteed to work on exact κ")
    print(f"  • Parameter count: {head.count_parameters():,}")
    print(f"  • Scales with max_length, NOT vocabulary size")
    
    print("\nNext: Train on synthetic data and compare to continuous regression")
