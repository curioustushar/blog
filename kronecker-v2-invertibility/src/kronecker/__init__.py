"""
Base Kronecker Embedding implementation for invertibility research.

This module provides the reference Kronecker encoder with controllable
normalization and projection strategies.
"""

from .encoder import (
    KroneckerEncoder,
    kronecker_codec,
    encode_single_token,
)
from .utils import (
    bytes_to_tensor,
    utf8_safe_truncate,
    token_to_bytes,
)

__all__ = [
    "KroneckerEncoder",
    "kronecker_codec",
    "encode_single_token",
    "bytes_to_tensor",
    "utf8_safe_truncate",
    "token_to_bytes",
]
