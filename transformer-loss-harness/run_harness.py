#!/usr/bin/env python3
"""Run the full loss-harness experiment and print submission numbers."""

from __future__ import annotations

import gc
import json
import math
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from harness import (
    PAD_ID,
    DualHeadLM,
    HarnessConfig,
    TinyLM,
    WordTokenizer,
    build_boundary_mask,
    build_padding_mask,
    chunked_cross_entropy,
    count_parameters,
    explain_shape,
    format_token_row,
    perplexity,
    shift_logits_and_targets,
    shift_logits_and_targets_offset,
)


@dataclass
class RunResults:
    config: dict
    valid_tokens_after_padding_mask: int
    packed_loss_before_boundary_mask: float
    packed_loss_after_boundary_mask: float
    initial_loss: float
    initial_perplexity: float
    tied_params: int
    untied_params: int
    param_difference: int
    ordinary_ce_peak_bytes: int
    chunked_ce_peak_bytes: int
    memory_ratio: float
    next_token_loss_final: float
    two_ahead_loss_final: float
    combined_loss_final: float


def peak_bytes(fn) -> tuple[int, object]:
    gc.collect()
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, result


def main() -> RunResults:
    config = HarnessConfig()
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    tokenizer = WordTokenizer.demo()
    model = TinyLM(config, tie_weights=False).to(device)
    model.eval()

    # --- baseline batch ---
    sample_texts = [
        "The cat sat on the mat .",
        "The dog ran quickly .",
        "The cat sat on the mat .",
        "The dog ran quickly .",
    ]
    encoded = [tokenizer.encode(t) for t in sample_texts]
    max_len = max(len(row) for row in encoded)
    padded_rows = [row + [PAD_ID] * (max_len - len(row)) for row in encoded]
    tokens = torch.tensor(padded_rows, device=device)
    if tokens.size(1) < config.seq_len:
        pad = torch.full(
            (tokens.size(0), config.seq_len - tokens.size(1)), PAD_ID, device=device
        )
        tokens = torch.cat([tokens, pad], dim=1)

    hidden = model(tokens)
    logits = model.output_head(hidden)
    shifted_logits, shifted_targets = shift_logits_and_targets(logits, tokens)

    assert hidden.shape == (config.batch_size, config.seq_len, config.hidden_dim)
    assert logits.shape == (config.batch_size, config.seq_len, config.vocab_size)
    assert shifted_logits.shape[0] == config.batch_size * (config.seq_len - 1)

    # string shift on first sequence
    row_tokens = tokens[0, :7].tolist()
    input_strings = format_token_row(row_tokens[:-1], tokenizer.itos)
    target_strings = format_token_row(row_tokens[1:], tokenizer.itos)
    assert input_strings == ["The", "cat", "sat", "on", "the", "mat"]
    assert target_strings == ["cat", "sat", "on", "the", "mat", "."]

    # padding mask
    target_positions = tokens[:, 1:]
    pad_mask = build_padding_mask(target_positions, pad_id=PAD_ID).reshape(-1)
    loss_unmasked = F.cross_entropy(shifted_logits, shifted_targets)
    loss_masked = F.cross_entropy(shifted_logits[pad_mask], shifted_targets[pad_mask])
    valid_after_pad = int(pad_mask.sum().item())

    # packed documents
    doc_a = tokenizer.encode("The cat sat .")
    doc_b = tokenizer.encode("The dog ran .")
    packed = doc_a + doc_b
    packed_tensor = torch.tensor([packed], device=device)
    packed_logits = model.logits(packed_tensor)
    packed_shift_logits, packed_shift_targets = shift_logits_and_targets(
        packed_logits, packed_tensor
    )
    boundary_index = len(doc_a) - 1
    boundary_mask = build_boundary_mask(
        packed_tensor.size(1), boundary_index, device=device
    )
    packed_loss_before = F.cross_entropy(packed_shift_logits, packed_shift_targets).item()
    packed_loss_after = F.cross_entropy(
        packed_shift_logits[boundary_mask], packed_shift_targets[boundary_mask]
    ).item()

    # perplexity sanity
    initial_loss = loss_masked.item()
    initial_ppl = perplexity(initial_loss)
    assert abs(initial_ppl - config.vocab_size) / config.vocab_size < 0.15

    tied = count_parameters(TinyLM(config, tie_weights=True))
    untied = count_parameters(TinyLM(config, tie_weights=False))

    # memory comparison on a large synthetic batch
    big_batch = 8
    big_seq = 128
    big_vocab = 8192
    big_hidden = 256
    big_logits = torch.randn(big_batch, big_seq - 1, big_vocab)
    big_targets = torch.randint(0, big_vocab, (big_batch * (big_seq - 1),))

    def ordinary():
        flat = big_logits.reshape(-1, big_vocab)
        log_probs = F.log_softmax(flat, dim=-1)
        idx = torch.arange(big_targets.size(0))
        return (-log_probs[idx, big_targets]).mean()

    def chunked():
        return chunked_cross_entropy(big_logits, big_targets, chunk_size=256)

    ordinary_peak, _ = peak_bytes(ordinary)
    chunked_peak, _ = peak_bytes(chunked)

    # Part 2 training
    dual = DualHeadLM(config).to(device)
    opt = torch.optim.Adam(dual.parameters(), lr=3e-4)
    train_tokens = tokens[:2].clone()
    final_t1 = final_t2 = final_sum = 0.0
    for _ in range(30):
        dual.train()
        opt.zero_grad()
        l1_logits, l1_targets = shift_logits_and_targets_offset(
            dual.logits_t1(train_tokens), train_tokens, offset=1
        )
        l2_logits, l2_targets = shift_logits_and_targets_offset(
            dual.logits_t2(train_tokens), train_tokens, offset=2
        )
        loss1 = F.cross_entropy(l1_logits, l1_targets)
        loss2 = F.cross_entropy(l2_logits, l2_targets)
        total = loss1 + loss2
        total.backward()
        opt.step()
        final_t1 = loss1.item()
        final_t2 = loss2.item()
        final_sum = total.item()

    results = RunResults(
        config={
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "vocab_size": config.vocab_size,
            "hidden_dim": config.hidden_dim,
        },
        valid_tokens_after_padding_mask=valid_after_pad,
        packed_loss_before_boundary_mask=packed_loss_before,
        packed_loss_after_boundary_mask=packed_loss_after,
        initial_loss=initial_loss,
        initial_perplexity=initial_ppl,
        tied_params=tied,
        untied_params=untied,
        param_difference=untied - tied,
        ordinary_ce_peak_bytes=ordinary_peak,
        chunked_ce_peak_bytes=chunked_peak,
        memory_ratio=ordinary_peak / max(chunked_peak, 1),
        next_token_loss_final=final_t1,
        two_ahead_loss_final=final_t2,
        combined_loss_final=final_sum,
    )

    print(json.dumps(asdict(results), indent=2))
    Path(__file__).parent.joinpath("results.json").write_text(json.dumps(asdict(results), indent=2))
    return results


if __name__ == "__main__":
    main()
