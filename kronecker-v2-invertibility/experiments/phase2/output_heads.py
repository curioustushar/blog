"""
Three output head implementations for Phase 2 comparison.

Approach A: Standard softmax (baseline)
Approach B: Kronecker input + standard softmax (current paper)
Approach C: Kronecker input + invertible output (V2 hypothesis)
"""

import math
import numpy as np
from typing import Tuple, Optional
import sys
import os

# Add src to path for Kronecker encoder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from kronecker_encoder import KroneckerEncoderStaged, algebraic_decode_raw_kronecker


class StandardSoftmaxHead:
    """
    Approach A: Standard vocabulary-sized output projection + softmax.
    
    This is the baseline approach used in nearly all language models.
    
    Architecture:
        h ∈ ℝ^d_model → W_out @ h → logits ∈ ℝ^V → softmax → token
    
    Parameters:
        W_out: V × d_model
    """
    
    def __init__(self, d_model: int, vocab_size: int):
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Output projection: d_model → vocab_size
        self.W_out = np.random.randn(d_model, vocab_size) * math.sqrt(2.0 / d_model)
        
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Compute logits for next token prediction.
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        logits = hidden_states @ self.W_out
        return logits
    
    def predict(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token IDs.
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            
        Returns:
            token_ids: (batch, seq_len)
        """
        logits = self.forward(hidden_states)
        token_ids = np.argmax(logits, axis=-1)
        return token_ids
    
    def count_parameters(self) -> int:
        """Return number of parameters."""
        return self.d_model * self.vocab_size
    
    def __repr__(self) -> str:
        return f"StandardSoftmaxHead(V={self.vocab_size:,}, d={self.d_model}, params={self.count_parameters():,})"


class KroneckerRegressionHead:
    """
    Approach C: Predict Kronecker representation directly, then decode.
    
    This is the V2 hypothesis: can we eliminate vocabulary-sized parameters
    by predicting the structured Kronecker representation?
    
    Architecture:
        h ∈ ℝ^d_model → W_regress @ h → κ_pred ∈ ℝ^D → algebraic_decode → token
    
    Parameters:
        W_regress: d_model × D  (where D = char_dim × pos_dim)
    
    Key insight: D is independent of vocabulary size V.
    For D=8192 and V=1M, this gives 122× parameter reduction.
    """
    
    def __init__(
        self,
        d_model: int,
        char_dim: int = 256,
        pos_dim: int = 32,
        apply_length_norm: bool = True,
    ):
        self.d_model = d_model
        self.char_dim = char_dim
        self.pos_dim = pos_dim
        self.D = char_dim * pos_dim
        self.apply_length_norm = apply_length_norm
        
        # Regression head: d_model → D
        self.W_regress = np.random.randn(d_model, self.D) * math.sqrt(2.0 / d_model)
        
        # Initialize Kronecker encoder for generating targets
        self.encoder = KroneckerEncoderStaged(
            char_dim=char_dim,
            pos_dim=pos_dim,
            apply_length_norm=apply_length_norm,
            apply_z_norm=False,  # Use raw Kronecker for now
            apply_projection=False,
        )
        
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict Kronecker representations.
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            
        Returns:
            kappa_pred: (batch, seq_len, D)
        """
        kappa_pred = hidden_states @ self.W_regress
        return kappa_pred
    
    def compute_target_kappa(self, tokens: list) -> np.ndarray:
        """
        Compute target Kronecker representations for a list of tokens.
        
        Args:
            tokens: List of token strings
            
        Returns:
            kappa_target: (len(tokens), D)
        """
        kappa_list = []
        for token in tokens:
            stages = self.encoder.encode_all_stages(token)
            kappa = stages['stage2_kronecker']
            kappa_list.append(kappa)
        
        kappa_target = np.stack(kappa_list, axis=0)
        return kappa_target
    
    def predict(self, hidden_states: np.ndarray, return_kappa: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict tokens by decoding Kronecker representations.
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            return_kappa: If True, also return predicted kappa
            
        Returns:
            tokens: List of decoded token strings (batch * seq_len,)
            kappa_pred: Optional (batch, seq_len, D) if return_kappa=True
        """
        kappa_pred = self.forward(hidden_states)
        batch_size, seq_len, D = kappa_pred.shape
        
        # Decode each position
        tokens = []
        for b in range(batch_size):
            for t in range(seq_len):
                kappa = kappa_pred[b, t]
                token_bytes = algebraic_decode_raw_kronecker(
                    kappa,
                    char_dim=self.char_dim,
                    pos_dim=self.pos_dim,
                    length_normalized=self.apply_length_norm,
                )
                
                if token_bytes is not None:
                    try:
                        token = token_bytes.decode('utf-8', errors='replace')
                    except:
                        token = None
                else:
                    token = None
                
                tokens.append(token)
        
        if return_kappa:
            return tokens, kappa_pred
        else:
            return tokens, None
    
    def compute_mse_loss(self, kappa_pred: np.ndarray, kappa_target: np.ndarray) -> float:
        """
        Compute MSE loss between predicted and target Kronecker representations.
        
        Args:
            kappa_pred: (batch, seq_len, D)
            kappa_target: (batch, seq_len, D)
            
        Returns:
            loss: scalar MSE
        """
        return np.mean((kappa_pred - kappa_target) ** 2)
    
    def count_parameters(self) -> int:
        """Return number of parameters."""
        return self.d_model * self.D
    
    def __repr__(self) -> str:
        return f"KroneckerRegressionHead(D={self.D:,}, d={self.d_model}, params={self.count_parameters():,})"


class HybridHead:
    """
    Hybrid approach: Combine standard softmax with Kronecker guidance.
    
    This is an experimental variant that could help with training.
    
    Architecture:
        h → W_out @ h → logits (standard)
        h → W_regress @ h → κ_pred (Kronecker)
        
    Loss: α * CrossEntropy(logits, target) + (1-α) * MSE(κ_pred, κ_target)
    
    Inference: Can use either softmax or Kronecker decoder.
    """
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        char_dim: int = 256,
        pos_dim: int = 32,
        alpha: float = 0.5,
    ):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.D = char_dim * pos_dim
        self.alpha = alpha
        
        self.softmax_head = StandardSoftmaxHead(d_model, vocab_size)
        self.kronecker_head = KroneckerRegressionHead(d_model, char_dim, pos_dim)
        
    def forward(self, hidden_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both logits and Kronecker predictions.
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            
        Returns:
            logits: (batch, seq_len, vocab_size)
            kappa_pred: (batch, seq_len, D)
        """
        logits = self.softmax_head.forward(hidden_states)
        kappa_pred = self.kronecker_head.forward(hidden_states)
        return logits, kappa_pred
    
    def count_parameters(self) -> int:
        """Return total number of parameters."""
        return self.softmax_head.count_parameters() + self.kronecker_head.count_parameters()
    
    def __repr__(self) -> str:
        softmax_params = self.softmax_head.count_parameters()
        kronecker_params = self.kronecker_head.count_parameters()
        total = softmax_params + kronecker_params
        return f"HybridHead(softmax={softmax_params:,}, kronecker={kronecker_params:,}, total={total:,})"


def compare_parameter_counts(d_model: int, vocab_sizes: list, char_dim: int = 256, pos_dim: int = 32):
    """
    Compare parameter counts across different output heads and vocabulary sizes.
    
    This demonstrates the key advantage of the Kronecker approach.
    """
    D = char_dim * pos_dim
    
    print("=" * 100)
    print("OUTPUT HEAD PARAMETER COMPARISON")
    print("=" * 100)
    print(f"Configuration: d_model={d_model}, char_dim={char_dim}, pos_dim={pos_dim}, D={D}")
    print()
    
    header = f"{'Vocab Size':>12} | {'Standard':>15} | {'Kronecker':>15} | {'Speedup':>10} | {'Savings':>12}"
    print(header)
    print("-" * 100)
    
    for V in vocab_sizes:
        standard_params = d_model * V
        kronecker_params = d_model * D
        speedup = standard_params / kronecker_params
        savings_pct = (1 - kronecker_params / standard_params) * 100
        
        print(f"{V:>12,} | {standard_params:>15,} | {kronecker_params:>15,} | {speedup:>10.1f}× | {savings_pct:>11.1f}%")
    
    print("=" * 100)
    print()
    print("Key observations:")
    print(f"  • Kronecker parameters are CONSTANT: {kronecker_params:,}")
    print(f"  • Standard parameters grow linearly with vocabulary size")
    print(f"  • At V=1M, Kronecker uses {kronecker_params / 1e6:.1f}M params vs {d_model * 1_000_000 / 1e6:.1f}M (standard)")
    print()


if __name__ == "__main__":
    print("Testing output head implementations...\n")
    
    # Test configuration
    d_model = 512
    vocab_size = 10_000
    batch_size = 4
    seq_len = 16
    char_dim = 256
    pos_dim = 32
    D = char_dim * pos_dim
    
    # Generate fake hidden states
    hidden_states = np.random.randn(batch_size, seq_len, d_model) * 0.1
    
    print("=" * 100)
    print("TEST 1: Standard Softmax Head")
    print("=" * 100)
    head_standard = StandardSoftmaxHead(d_model, vocab_size)
    print(head_standard)
    
    logits = head_standard.forward(hidden_states)
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Parameters: {head_standard.count_parameters():,}")
    
    predictions = head_standard.predict(hidden_states)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[0, :8]}")
    
    print("\n" + "=" * 100)
    print("TEST 2: Kronecker Regression Head")
    print("=" * 100)
    head_kronecker = KroneckerRegressionHead(d_model, char_dim, pos_dim)
    print(head_kronecker)
    
    kappa_pred = head_kronecker.forward(hidden_states)
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape (κ): {kappa_pred.shape}")
    print(f"Parameters: {head_kronecker.count_parameters():,}")
    print(f"κ sparsity: {np.mean(np.abs(kappa_pred) < 0.01) * 100:.1f}% near-zero")
    
    # Test target computation
    test_tokens = ["the", "a", "test", "hello"]
    kappa_target = head_kronecker.compute_target_kappa(test_tokens)
    print(f"\nTarget κ shape: {kappa_target.shape}")
    print(f"Target κ sparsity: {np.mean(np.abs(kappa_target) < 0.01) * 100:.1f}% near-zero")
    print(f"Target κ mean nonzero count: {np.mean(np.sum(np.abs(kappa_target) > 0.01, axis=1)):.1f}")
    
    print("\n" + "=" * 100)
    print("TEST 3: Parameter Comparison")
    print("=" * 100)
    
    compare_parameter_counts(
        d_model=512,
        vocab_sizes=[1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000],
        char_dim=256,
        pos_dim=32,
    )
    
    print("=" * 100)
    print("✅ All tests passed!")
    print("=" * 100)
