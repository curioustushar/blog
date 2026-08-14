"""
Simplest possible codec test: one fixed sequence.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from kronecker_encoder import KroneckerEncoderStaged, algebraic_decode_raw_kronecker


def construct_kronecker_from_bytes(byte_seq: bytes) -> np.ndarray:
    char_dim, pos_dim = 256, 32
    L = len(byte_seq)
    D = char_dim * pos_dim
    kappa = np.zeros(D, dtype=np.float64)
    
    for p in range(L):
        byte_val = byte_seq[p]
        idx = byte_val * pos_dim + p
        kappa[idx] = 1.0
    
    if L > 0:
        kappa /= np.sqrt(L)
    
    return kappa


print("Simple Codec Test")
print("="*80)

# Use the official encoder as ground truth
encoder = KroneckerEncoderStaged(
    char_dim=256,
    pos_dim=32,
    apply_length_norm=True,
    apply_z_norm=False,
    apply_projection=False,
)

# Test with a known string
test_string = "hello"
print(f"\nTest string: '{test_string}'")

# Official encoding
stages = encoder.encode_all_stages(test_string)
kappa_official = stages['stage2_kronecker']

# My encoding
bytes_in = test_string.encode('utf-8')
kappa_mine = construct_kronecker_from_bytes(bytes_in)

print(f"\nBytes: {list(bytes_in)}")
print(f"Official κ nnz: {np.sum(np.abs(kappa_official) > 1e-6)}")
print(f"My κ nnz: {np.sum(np.abs(kappa_mine) > 1e-6)}")
print(f"κ match: {np.allclose(kappa_official, kappa_mine)}")

# Decode with official decoder
decoded_from_official = algebraic_decode_raw_kronecker(kappa_official, 256, 32, True)
decoded_from_mine = algebraic_decode_raw_kronecker(kappa_mine, 256, 32, True)

print(f"\nDecoded from official κ: {decoded_from_official}")
print(f"Decoded from my κ: {decoded_from_mine}")

print(f"\nRound-trip from official: {decoded_from_official == bytes_in}")
print(f"Round-trip from mine: {decoded_from_mine == bytes_in}")

if decoded_from_official == bytes_in and decoded_from_mine == bytes_in:
    print("\n✅ Codec works perfectly!")
else:
    print("\n❌ Codec broken")
    print(f"  Expected: {bytes_in}")
    print(f"  From official: {decoded_from_official}")
    print(f"  From mine: {decoded_from_mine}")

print("="*80)
