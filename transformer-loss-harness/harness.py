"""Transformer loss harness — shared logic for notebook and local runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_ID = 0


@dataclass(frozen=True)
class HarnessConfig:
    vocab_size: int = 256
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    batch_size: int = 4
    seq_len: int = 32
    seed: int = 42
    device: str = "cpu"

    @property
    def tied_param_count(self) -> int:
        return count_parameters(TinyLM(self, tie_weights=True))

    @property
    def untied_param_count(self) -> int:
        return count_parameters(TinyLM(self, tie_weights=False))


class TinyLM(nn.Module):
    """Small causal LM for loss-harness experiments."""

    def __init__(self, config: HarnessConfig, tie_weights: bool = True) -> None:
        super().__init__()
        self.config = config
        self.tie_weights = tie_weights
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if tie_weights:
            self.output_head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(tokens)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tokens.size(1), device=tokens.device
        )
        hidden = self.encoder(hidden, mask=causal_mask, is_causal=True)
        return hidden

    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output_head(self.forward(tokens))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def explain_shape(name: str, tensor: torch.Tensor) -> str:
    dim_names = {
        1: ["index"],
        2: ["batch", "sequence"],
        3: ["batch", "sequence", "feature"],
    }
    labels = dim_names.get(tensor.dim(), [f"dim{i}" for i in range(tensor.dim())])
    shape_bits = ", ".join(f"{label}={size}" for label, size in zip(labels, tensor.shape))
    return f"{name}: {list(tensor.shape)}  # {shape_bits}"


def shift_logits_and_targets(
    logits: torch.Tensor, tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressive next-token alignment: position t predicts token t+1."""
    shifted_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shifted_targets = tokens[:, 1:].reshape(-1)
    return shifted_logits, shifted_targets


def shift_logits_and_targets_offset(
    logits: torch.Tensor, tokens: torch.Tensor, offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict token t+offset from position t."""
    if offset < 1:
        raise ValueError("offset must be >= 1")
    shifted_logits = logits[:, : -offset, :].reshape(-1, logits.size(-1))
    shifted_targets = tokens[:, offset:].reshape(-1)
    return shifted_logits, shifted_targets


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy over flattened logits/targets with boolean mask."""
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1).bool()
    if flat_mask.sum() == 0:
        raise ValueError("mask eliminates all targets")
    return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask], reduction=reduction)


def build_padding_mask(target_positions: torch.Tensor, pad_id: int = PAD_ID) -> torch.Tensor:
    return target_positions != pad_id


def build_boundary_mask(
    seq_len: int, boundary_index: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Mask out the target position that would predict across a document boundary."""
    mask = torch.ones(seq_len - 1, dtype=torch.bool, device=device)
    if 0 <= boundary_index < seq_len - 1:
        mask[boundary_index] = False
    return mask


def perplexity(loss: float) -> float:
    return math.exp(loss)


def chunked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Chunked CE processes token positions in chunks to limit peak memory."""
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    total = flat_logits.size(0)
    losses: list[torch.Tensor] = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        losses.append(
            F.cross_entropy(flat_logits[start:end], flat_targets[start:end], reduction="sum")
        )
    return torch.stack(losses).sum() / total


def format_token_row(tokens: Iterable[int], id_to_token: dict[int, str]) -> list[str]:
    return [id_to_token.get(int(t), f"<{int(t)}>") for t in tokens]


class WordTokenizer:
    """Tiny word-level tokenizer for readable string verification."""

    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab
        self.stoi = {tok: idx for idx, tok in enumerate(vocab)}
        self.itos = {idx: tok for tok, idx in self.stoi.items()}

    @classmethod
    def demo(cls) -> WordTokenizer:
        words = [
            "<pad>",
            "<eos>",
            "The",
            "cat",
            "sat",
            "on",
            "the",
            "mat",
            ".",
            "dog",
            "ran",
            "quickly",
        ]
        return cls(words)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[w] for w in text.split()]

    def decode(self, ids: list[int] | torch.Tensor) -> list[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return [self.itos[int(i)] for i in ids]


class DualHeadLM(nn.Module):
    """Two prediction heads: t+1 and t+2."""

    def __init__(self, config: HarnessConfig) -> None:
        super().__init__()
        self.backbone = TinyLM(config, tie_weights=True)
        self.head_t2 = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.backbone(tokens)

    def logits_t1(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.backbone.logits(tokens)

    def logits_t2(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(tokens)
        return self.head_t2(hidden)
