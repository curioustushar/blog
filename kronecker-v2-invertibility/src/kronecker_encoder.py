"""
Minimal Kronecker Encoder for Injectivity Testing

Implements the exact pipeline from the paper (arXiv:2605.29459) with
explicit tracking of each transformation stage.
"""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class KroneckerEncoderStaged:
    """
    Kronecker encoder that exposes every transformation stage.
    
    Pipeline (from paper Section 3):
    
    Stage 0: token (string)
    Stage 1: UTF-8 bytes (b_1, ..., b_L)
    Stage 2: Kronecker sum: κ(b) = (1/√L) Σ (c_b ⊗ p_p)
    Stage 3: Z-normalization: κ̄ = (κ - μ) / σ
    Stage 4: Projection: e = W · κ̄  (if projection exists)
    
    Each stage is tested for collisions.
    """
    
    def __init__(
        self,
        char_dim: int = 256,
        pos_dim: int = 32,
        d_model: Optional[int] = None,
        apply_length_norm: bool = True,
        apply_z_norm: bool = True,
        apply_projection: bool = False,
        projection_seed: int = 42,
    ):
        """
        Parameters
        ----------
        char_dim : int
            Byte alphabet size (always 256 for UTF-8)
        pos_dim : int
            Maximum byte position (paper uses 16 or 32)
        d_model : int, optional
            Output dimension for projection (if apply_projection=True)
        apply_length_norm : bool
            Apply 1/√L normalization (paper default: True)
        apply_z_norm : bool
            Apply z-normalization (subtract mean, divide by std)
        apply_projection : bool
            Apply learned projection W: D → d_model
        projection_seed : int
            Random seed for projection initialization
        """
        self.char_dim = char_dim
        self.pos_dim = pos_dim
        self.D = char_dim * pos_dim
        self.d_model = d_model
        
        self.apply_length_norm = apply_length_norm
        self.apply_z_norm = apply_z_norm
        self.apply_projection = apply_projection
        
        # Initialize projection if requested
        self.projection_matrix = None
        if apply_projection:
            if d_model is None:
                raise ValueError("d_model required when apply_projection=True")
            np.random.seed(projection_seed)
            # Paper uses normal init with std=1/√D
            std = 1.0 / math.sqrt(self.D)
            self.projection_matrix = np.random.normal(0, std, (d_model, self.D)).astype(np.float64)
    
    def token_to_bytes(self, token: str, max_len: Optional[int] = None) -> bytes:
        """
        Convert token string to UTF-8 bytes.
        
        Stage 0 → Stage 1 transformation.
        """
        byte_seq = token.encode('utf-8')
        if max_len is not None and len(byte_seq) > max_len:
            # UTF-8 safe truncation: don't break multi-byte characters
            byte_seq = self._utf8_safe_truncate(byte_seq, max_len)
        return byte_seq
    
    def _utf8_safe_truncate(self, byte_seq: bytes, max_len: int) -> bytes:
        """Truncate bytes without breaking UTF-8 multi-byte sequences."""
        if len(byte_seq) <= max_len:
            return byte_seq
        
        # Walk backward from max_len to find a valid UTF-8 boundary
        for i in range(max_len, max(0, max_len - 4), -1):
            try:
                byte_seq[:i].decode('utf-8')
                return byte_seq[:i]
            except UnicodeDecodeError:
                continue
        return byte_seq[:max_len]  # Fallback: might break UTF-8
    
    def bytes_to_kronecker(self, byte_seq: bytes) -> Tuple[np.ndarray, Dict]:
        """
        Compute raw Kronecker representation κ(b).
        
        From paper Equation 1:
        κ(b) = (1/√L) Σ_{p=1}^L (c_{b_p} ⊗ p_p)
        
        Stage 1 → Stage 2 transformation.
        
        Returns
        -------
        kappa : np.ndarray of shape (D,)
            Raw Kronecker representation (before z-norm)
        metadata : dict
            Contains: L (length), bytes (the byte sequence)
        """
        L = min(len(byte_seq), self.pos_dim)
        
        # Initialize output: D-dimensional vector
        kappa = np.zeros(self.D, dtype=np.float64)
        
        # For each byte at position p
        for p in range(L):
            byte_val = byte_seq[p]
            # Linear index into flattened (char_dim, pos_dim) tensor
            # Storage: index = byte_value * pos_dim + position
            idx = byte_val * self.pos_dim + p
            kappa[idx] += 1.0
        
        # Length normalization: divide by √L
        if self.apply_length_norm and L > 0:
            kappa /= math.sqrt(L)
        
        metadata = {
            'L': L,
            'bytes': byte_seq[:L],
            'nnz': L,  # Number of non-zero entries
        }
        
        return kappa, metadata
    
    def apply_z_normalization(self, kappa: np.ndarray) -> np.ndarray:
        """
        Apply z-normalization: κ̄ = (κ - μ) / σ
        
        Stage 2 → Stage 3 transformation.
        
        WARNING: This is NOT invertible without storing μ and σ.
        """
        if not self.apply_z_norm:
            return kappa
        
        mean = kappa.mean()
        std = kappa.std()
        
        eps = 1e-6  # Numerical stability
        kappa_normalized = (kappa - mean) / (std + eps)
        
        return kappa_normalized
    
    def apply_projection_transform(self, kappa_bar: np.ndarray) -> np.ndarray:
        """
        Apply learned projection: e = W · κ̄
        
        Stage 3 → Stage 4 transformation.
        
        This is a dimensionality reduction when d_model < D.
        """
        if not self.apply_projection:
            return kappa_bar
        
        # e = W @ kappa_bar, where W is (d_model, D)
        e = self.projection_matrix @ kappa_bar
        return e
    
    def encode_all_stages(self, token: str) -> Dict[str, np.ndarray]:
        """
        Encode token through all pipeline stages and return each intermediate result.
        
        Returns
        -------
        stages : dict
            {
                'stage0_token': str,
                'stage1_bytes': bytes,
                'stage2_kronecker': np.ndarray,
                'stage3_znorm': np.ndarray,
                'stage4_projection': np.ndarray (if applicable)
            }
        """
        stages = {}
        
        # Stage 0: Original token
        stages['stage0_token'] = token
        
        # Stage 1: UTF-8 bytes
        byte_seq = self.token_to_bytes(token, max_len=self.pos_dim)
        stages['stage1_bytes'] = byte_seq
        
        # Stage 2: Raw Kronecker representation
        kappa, metadata = self.bytes_to_kronecker(byte_seq)
        stages['stage2_kronecker'] = kappa
        stages['metadata'] = metadata
        
        # Stage 3: Z-normalization (if enabled)
        kappa_bar = self.apply_z_normalization(kappa)
        stages['stage3_znorm'] = kappa_bar
        
        # Stage 4: Projection (if enabled)
        if self.apply_projection:
            e = self.apply_projection_transform(kappa_bar)
            stages['stage4_projection'] = e
        
        return stages
    
    def encode(self, token: str, return_stages: bool = False):
        """
        Encode a single token to its final representation.
        
        Parameters
        ----------
        token : str
            Input token
        return_stages : bool
            If True, return all intermediate stages
        
        Returns
        -------
        representation : np.ndarray
            Final representation (stage 2, 3, or 4 depending on config)
        stages : dict (if return_stages=True)
            All intermediate representations
        """
        all_stages = self.encode_all_stages(token)
        
        # Determine final stage
        if self.apply_projection:
            final = all_stages['stage4_projection']
        elif self.apply_z_norm:
            final = all_stages['stage3_znorm']
        else:
            final = all_stages['stage2_kronecker']
        
        if return_stages:
            return final, all_stages
        return final


def compute_representation_hash(arr: np.ndarray, precision: int = 8) -> str:
    """
    Compute a deterministic hash of a representation.
    
    Rounds to `precision` decimal places to handle floating-point noise.
    """
    rounded = np.round(arr, decimals=precision)
    return hash(rounded.tobytes())


def find_collisions(
    tokens: List[str],
    encoder: KroneckerEncoderStaged,
    stage_name: str,
    tolerance: float = 1e-10,
) -> Tuple[int, List[Tuple[str, str, float]]]:
    """
    Find collision pairs at a specific pipeline stage.
    
    Parameters
    ----------
    tokens : list of str
        Vocabulary to test
    encoder : KroneckerEncoderStaged
        Encoder instance
    stage_name : str
        Which stage to test: 'stage2_kronecker', 'stage3_znorm', 'stage4_projection'
    tolerance : float
        Distance threshold for considering two representations "colliding"
    
    Returns
    -------
    unique_count : int
        Number of unique representations found
    collisions : list of (token_a, token_b, distance)
        Pairs of tokens with distance < tolerance
    """
    print(f"\n{'='*60}")
    print(f"Testing stage: {stage_name}")
    print(f"{'='*60}")
    
    # Encode all tokens
    representations = {}
    for token in tokens:
        stages = encoder.encode_all_stages(token)
        if stage_name in stages:
            representations[token] = stages[stage_name]
    
    if not representations:
        print(f"Stage {stage_name} not found. Skipping.")
        return 0, []
    
    # Check for exact duplicates using hash
    hash_to_tokens = {}
    for token, rep in representations.items():
        h = compute_representation_hash(rep, precision=12)
        if h not in hash_to_tokens:
            hash_to_tokens[h] = []
        hash_to_tokens[h].append(token)
    
    # Find hash collisions (exact matches)
    exact_collisions = []
    for h, token_list in hash_to_tokens.items():
        if len(token_list) > 1:
            # Multiple tokens map to same hash
            for i in range(len(token_list)):
                for j in range(i+1, len(token_list)):
                    exact_collisions.append((token_list[i], token_list[j], 0.0))
    
    # Find near collisions (distance < tolerance)
    tokens_list = list(representations.keys())
    near_collisions = []
    
    for i in range(len(tokens_list)):
        for j in range(i+1, min(i+1000, len(tokens_list))):  # Only check nearby pairs for speed
            t1, t2 = tokens_list[i], tokens_list[j]
            r1, r2 = representations[t1], representations[t2]
            
            dist = np.linalg.norm(r1 - r2)
            if 0 < dist < tolerance:
                near_collisions.append((t1, t2, dist))
    
    unique_count = len(hash_to_tokens)
    all_collisions = exact_collisions + near_collisions
    
    # Report
    print(f"Total tokens: {len(tokens)}")
    print(f"Unique representations: {unique_count}")
    print(f"Exact collisions (distance = 0): {len(exact_collisions)}")
    print(f"Near collisions (0 < distance < {tolerance}): {len(near_collisions)}")
    
    if exact_collisions:
        print(f"\n⚠️  EXACT COLLISIONS FOUND:")
        for t1, t2, dist in exact_collisions[:5]:
            print(f"  '{t1}' ≡ '{t2}'")
            print(f"    bytes: {t1.encode('utf-8').hex()} vs {t2.encode('utf-8').hex()}")
    
    if near_collisions:
        print(f"\n⚠️  NEAR COLLISIONS FOUND:")
        for t1, t2, dist in near_collisions[:5]:
            print(f"  '{t1}' ≈ '{t2}'  (distance: {dist:.2e})")
    
    if not exact_collisions and not near_collisions:
        print(f"✅ No collisions found")
    
    return unique_count, all_collisions


def compute_min_pairwise_distance(
    tokens: List[str],
    encoder: KroneckerEncoderStaged,
    stage_name: str,
    sample_size: int = 1000,
) -> float:
    """
    Compute minimum pairwise distance (sampled for speed).
    """
    representations = {}
    for token in tokens[:sample_size]:
        stages = encoder.encode_all_stages(token)
        if stage_name in stages:
            representations[token] = stages[stage_name]
    
    tokens_list = list(representations.keys())
    min_dist = float('inf')
    
    for i in range(len(tokens_list)):
        for j in range(i+1, min(len(tokens_list), i+100)):
            r1 = representations[tokens_list[i]]
            r2 = representations[tokens_list[j]]
            dist = np.linalg.norm(r1 - r2)
            if dist > 0:
                min_dist = min(min_dist, dist)
    
    return min_dist if min_dist != float('inf') else 0.0


def algebraic_decode_raw_kronecker(
    kappa: np.ndarray,
    char_dim: int = 256,
    pos_dim: int = 32,
    length_normalized: bool = True,
    tolerance: float = 1e-6,
) -> Optional[bytes]:
    """
    Algebraic decoder: reconstruct byte sequence from raw Kronecker representation.
    
    Algorithm:
    1. Find non-zero positions in κ(b)
    2. Extract (byte_val, position) from each index
    3. Reconstruct byte sequence
    4. Verify consistency
    
    Complexity: O(D) = O(char_dim × pos_dim), independent of vocabulary size V.
    
    Parameters
    ----------
    kappa : np.ndarray
        Raw Kronecker representation
    char_dim : int
        Byte alphabet size (256 for UTF-8)
    pos_dim : int
        Maximum byte position
    length_normalized : bool
        Whether input was length-normalized
    tolerance : float
        Threshold for considering an entry non-zero
    
    Returns
    -------
    bytes or None
        Reconstructed byte sequence, or None if decoding fails
    """
    D = char_dim * pos_dim
    
    if len(kappa) != D:
        return None
    
    # Find non-zero entries (O(D) operation)
    nonzero_indices = np.where(np.abs(kappa) > tolerance)[0]
    
    if len(nonzero_indices) == 0:
        return b''  # Empty token
    
    # Extract byte-position pairs
    byte_position_pairs = []
    for idx in nonzero_indices:
        byte_val = idx // pos_dim
        position = idx % pos_dim
        magnitude = kappa[idx]
        
        byte_position_pairs.append((position, byte_val, magnitude))
    
    # Sort by position
    byte_position_pairs.sort()
    
    # Verify positions are consecutive (0, 1, 2, ...)
    positions = [pos for pos, _, _ in byte_position_pairs]
    expected_positions = list(range(len(positions)))
    if positions != expected_positions:
        # Non-consecutive positions - possibly corrupted
        return None
    
    # Extract bytes
    byte_sequence = bytes([byte_val for _, byte_val, _ in byte_position_pairs])
    
    return byte_sequence


if __name__ == "__main__":
    # Quick test
    encoder = KroneckerEncoderStaged(
        char_dim=256,
        pos_dim=32,
        apply_length_norm=True,
        apply_z_norm=False,
        apply_projection=False,
    )
    
    test_tokens = ["hello", "world", "test"]
    for token in test_tokens:
        final, stages = encoder.encode(token, return_stages=True)
        print(f"\nToken: '{token}'")
        print(f"  Bytes: {stages['stage1_bytes'].hex()}")
        print(f"  Kronecker shape: {stages['stage2_kronecker'].shape}")
        print(f"  Non-zero entries: {np.count_nonzero(stages['stage2_kronecker'])}")
        print(f"  Norm: {np.linalg.norm(stages['stage2_kronecker']):.4f}")
        
        # Test decoder
        decoded_bytes = algebraic_decode_raw_kronecker(stages['stage2_kronecker'])
        if decoded_bytes:
            decoded_token = decoded_bytes.decode('utf-8', errors='replace')
            match = "✓" if decoded_token == token else "✗"
            print(f"  Decoded: '{decoded_token}' {match}")
