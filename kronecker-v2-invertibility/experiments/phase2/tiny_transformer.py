"""
Tiny Transformer implementation for Phase 2 experiments.

Clean, minimal implementation for testing Kronecker invertible output heads.
"""

import math
import numpy as np
from typing import Optional, Tuple


class MultiHeadAttention:
    """Multi-head self-attention."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        
        # Q, K, V projections (combined for efficiency)
        self.qkv_proj = np.random.randn(d_model, 3 * d_model) * math.sqrt(2.0 / d_model)
        self.out_proj = np.random.randn(d_model, d_model) * math.sqrt(2.0 / d_model)
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) attention mask
            
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        qkv = x @ self.qkv_proj  # (batch, seq_len, 3*d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq_len, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.d_head)
        
        if mask is not None:
            # Expand mask to match scores shape: (batch, n_heads, seq_len, seq_len)
            mask_expanded = np.expand_dims(mask, axis=1)  # (batch, 1, seq_len, seq_len)
            scores = scores + mask_expanded * -1e9
        
        attn_weights = self._softmax(scores, axis=-1)
        
        # Apply attention to values
        attn_output = attn_weights @ v  # (batch, n_heads, seq_len, d_head)
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3)  # (batch, seq_len, n_heads, d_head)
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)
        
        # Output projection
        output = attn_output @ self.out_proj
        
        return output
    
    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class FeedForward:
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Two-layer MLP with GELU activation
        self.w1 = np.random.randn(d_model, d_ff) * math.sqrt(2.0 / d_model)
        self.w2 = np.random.randn(d_ff, d_model) * math.sqrt(2.0 / d_ff)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            (batch, seq_len, d_model)
        """
        # FFN(x) = W2 * GELU(W1 * x)
        hidden = x @ self.w1
        hidden = self._gelu(hidden)
        output = hidden @ self.w2
        return output
    
    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit activation."""
        return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


class LayerNorm:
    """Layer normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            (batch, seq_len, d_model)
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class TransformerBlock:
    """Single Transformer block with attention and feed-forward."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.dropout = dropout
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) causal mask
            
        Returns:
            (batch, seq_len, d_model)
        """
        # Pre-norm architecture
        # x = x + Attention(LN(x))
        attn_out = self.attention.forward(self.ln1.forward(x), mask)
        x = x + attn_out
        
        # x = x + FFN(LN(x))
        ffn_out = self.ffn.forward(self.ln2.forward(x))
        x = x + ffn_out
        
        return x


class TinyTransformer:
    """
    Minimal Transformer language model for Phase 2 experiments.
    
    This implementation is intentionally simple and uses numpy for clarity.
    For actual training, this should be rewritten in PyTorch/JAX.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        use_kronecker_input: bool = False,
        kronecker_config: Optional[dict] = None,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.use_kronecker_input = use_kronecker_input
        
        # Input embedding
        if use_kronecker_input:
            # Kronecker embedding will be implemented separately
            assert kronecker_config is not None
            self.char_dim = kronecker_config.get('char_dim', 256)
            self.pos_dim = kronecker_config.get('pos_dim', 32)
            self.D = self.char_dim * self.pos_dim
            self.input_projection = np.random.randn(self.D, d_model) * math.sqrt(2.0 / self.D)
        else:
            # Standard learned embeddings
            self.token_embeddings = np.random.randn(vocab_size, d_model) * 0.01
        
        # Positional embeddings (learned)
        self.position_embeddings = np.random.randn(max_seq_len, d_model) * 0.01
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]
        
        # Final layer norm
        self.ln_f = LayerNorm(d_model)
        
        # Note: Output head is NOT included here
        # It will be implemented separately in output_heads.py
        
    def forward(
        self,
        input_ids: np.ndarray,
        kronecker_embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass through the Transformer backbone.
        
        Args:
            input_ids: (batch, seq_len) token IDs
            kronecker_embeddings: (batch, seq_len, D) optional Kronecker representations
            
        Returns:
            (batch, seq_len, d_model) hidden states
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
        
        # Input embeddings
        if self.use_kronecker_input:
            assert kronecker_embeddings is not None, "Kronecker embeddings required"
            # Project Kronecker to d_model
            x = kronecker_embeddings @ self.input_projection  # (batch, seq_len, d_model)
        else:
            # Standard token embeddings
            x = self.token_embeddings[input_ids]  # (batch, seq_len, d_model)
        
        # Add positional embeddings
        positions = np.arange(seq_len)
        x = x + self.position_embeddings[positions]
        
        # Create causal attention mask
        mask = self._create_causal_mask(seq_len)
        mask = np.broadcast_to(mask, (batch_size, seq_len, seq_len))
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)
        
        # Final layer norm
        x = self.ln_f.forward(x)
        
        return x
    
    @staticmethod
    def _create_causal_mask(seq_len: int) -> np.ndarray:
        """
        Create causal attention mask (upper triangular).
        
        Returns:
            (seq_len, seq_len) mask where future positions are masked
        """
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mask  # 1 = mask, 0 = attend
    
    def count_parameters(self) -> dict:
        """Count parameters in each component."""
        params = {}
        
        # Input embeddings
        if self.use_kronecker_input:
            params['input_projection'] = self.D * self.d_model
        else:
            params['token_embeddings'] = self.vocab_size * self.d_model
        
        params['position_embeddings'] = self.max_seq_len * self.d_model
        
        # Transformer blocks
        # Each block has:
        # - Attention: d_model * 3*d_model (QKV) + d_model * d_model (out)
        # - FFN: d_model * d_ff + d_ff * d_model
        # - LayerNorm: 2 * d_model (gamma, beta) x 2
        attn_params = self.d_model * 3 * self.d_model + self.d_model * self.d_model
        ffn_params = self.d_model * self.d_ff + self.d_ff * self.d_model
        ln_params = 2 * self.d_model * 2
        block_params = attn_params + ffn_params + ln_params
        
        params['transformer_blocks'] = self.n_layers * block_params
        params['final_ln'] = 2 * self.d_model
        
        params['total_backbone'] = sum(params.values())
        
        return params


def print_model_summary(model: TinyTransformer):
    """Print a summary of the model architecture and parameters."""
    print("=" * 80)
    print("TINY TRANSFORMER MODEL SUMMARY")
    print("=" * 80)
    print(f"Architecture:")
    print(f"  Vocabulary size: {model.vocab_size:,}")
    print(f"  Model dimension: {model.d_model}")
    print(f"  Number of layers: {model.n_layers}")
    print(f"  Number of heads: {model.n_heads}")
    print(f"  FFN dimension: {model.d_ff}")
    print(f"  Max sequence length: {model.max_seq_len}")
    print(f"  Dropout: {model.dropout}")
    print(f"  Kronecker input: {model.use_kronecker_input}")
    
    params = model.count_parameters()
    print(f"\nParameter Counts:")
    for name, count in params.items():
        if name != 'total_backbone':
            print(f"  {name}: {count:,}")
    print(f"  {'─' * 30}")
    print(f"  TOTAL (backbone): {params['total_backbone']:,}")
    print(f"\nNote: Output head parameters are not included here.")
    print("=" * 80)


if __name__ == "__main__":
    # Test the implementation
    print("Testing Tiny Transformer implementation...\n")
    
    # Configuration
    config = {
        'vocab_size': 10_000,
        'd_model': 512,
        'n_layers': 4,
        'n_heads': 8,
        'd_ff': 2048,
        'max_seq_len': 128,
        'dropout': 0.1,
    }
    
    # Test 1: Standard embeddings
    print("TEST 1: Standard Embeddings")
    model_standard = TinyTransformer(
        use_kronecker_input=False,
        **config
    )
    print_model_summary(model_standard)
    
    # Forward pass test
    batch_size, seq_len = 4, 32
    input_ids = np.random.randint(0, config['vocab_size'], (batch_size, seq_len))
    hidden_states = model_standard.forward(input_ids)
    print(f"\nForward pass test:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {hidden_states.shape}")
    print(f"  Output mean: {np.mean(hidden_states):.4f}")
    print(f"  Output std: {np.std(hidden_states):.4f}")
    
    print("\n" + "=" * 80 + "\n")
    
    # Test 2: Kronecker embeddings
    print("TEST 2: Kronecker Embeddings")
    model_kronecker = TinyTransformer(
        use_kronecker_input=True,
        kronecker_config={'char_dim': 256, 'pos_dim': 32},
        **config
    )
    print_model_summary(model_kronecker)
    
    # Forward pass test with Kronecker
    D = 256 * 32
    kronecker_embeddings = np.random.randn(batch_size, seq_len, D) * 0.1
    hidden_states_kron = model_kronecker.forward(input_ids, kronecker_embeddings)
    print(f"\nForward pass test:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Kronecker embeddings shape: {kronecker_embeddings.shape}")
    print(f"  Output shape: {hidden_states_kron.shape}")
    print(f"  Output mean: {np.mean(hidden_states_kron):.4f}")
    print(f"  Output std: {np.std(hidden_states_kron):.4f}")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
