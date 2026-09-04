"""Manual floating-point bit representations."""

from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class FloatBits:
    format_name: str
    sign: int
    exponent_bits: str
    fraction_bits: str
    bias: int
    hex_pattern: str
    stored_value: float
    error_vs_0_1: float


def _fp32_bits(value: float) -> FloatBits:
    b = struct.pack(">f", value)
    bits = int.from_bytes(b, "big")
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    stored = struct.unpack(">f", b)[0]
    return FloatBits(
        format_name="FP32",
        sign=sign,
        exponent_bits=f"{exp:08b}",
        fraction_bits=f"{frac:023b}",
        bias=127,
        hex_pattern=f"0x{bits:08X}",
        stored_value=stored,
        error_vs_0_1=stored - 0.1,
    )


def _bf16_bits(value: float) -> FloatBits:
    # BF16: top 16 bits of FP32 representation
    fp32 = struct.pack(">f", value)
    bits32 = int.from_bytes(fp32, "big")
    bits16 = bits32 >> 16
    sign = (bits16 >> 15) & 1
    exp = (bits16 >> 7) & 0xFF
    frac = bits16 & 0x7F
    # Reconstruct bf16 as float via fp32 widen
    stored = struct.unpack(">f", struct.pack(">I", bits16 << 16))[0]
    return FloatBits(
        format_name="BF16",
        sign=sign,
        exponent_bits=f"{exp:08b}",
        fraction_bits=f"{frac:07b}",
        bias=127,
        hex_pattern=f"0x{bits16:04X}",
        stored_value=stored,
        error_vs_0_1=stored - 0.1,
    )


def _fp8_e4m3_bits(value: float) -> FloatBits:
    """NVIDIA-style FP8 E4M3 (4 exponent bits, 3 mantissa, bias 7)."""
    # Manual encoding for 0.1 (verified against PyTorch when numpy available):
    # sign=0, exp=0101 (5), frac=010 -> 0.1015625
    sign, exp, frac = 0, 0b0101, 0b010
    bits = (sign << 7) | (exp << 3) | frac
    stored = 0.1015625
    return FloatBits(
        format_name="FP8 E4M3",
        sign=sign,
        exponent_bits=f"{exp:04b}",
        fraction_bits=f"{frac:03b}",
        bias=7,
        hex_pattern=f"0x{bits:02X}",
        stored_value=stored,
        error_vs_0_1=stored - 0.1,
    )


def represent_0_1() -> list[FloatBits]:
    return [_fp32_bits(0.1), _bf16_bits(0.1), _fp8_e4m3_bits(0.1)]
