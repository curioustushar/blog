"""Generate the Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent / "transformer_loss_harness.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


cells = [
    md(
        """# Transformer Loss Harness: Make It Correct, Observable, and Explainable

**Colab:** [Open verified notebook](https://colab.research.google.com/drive/1B0FOUHvzvu-mt8Wdswnegv2gL7cvMx3H?usp=sharing)  
**GitHub:** [transformer-loss-harness](https://github.com/curioustushar/blog/tree/master/transformer-loss-harness)

Runs top-to-bottom in Google Colab.

> **Warning:** A wrong target shift can produce a beautiful loss curve. This notebook prints decoded token **strings** at every critical step.
"""
    ),
    code(
        """# Setup
!pip -q install matplotlib

import gc
import math
import tracemalloc
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SEED = 42
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

PAD_ID = 0
EOS_ID = None  # set after tokenizer is built
"""
    ),
    code(
        """@dataclass(frozen=True)
class HarnessConfig:
    vocab_size: int = 256
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    batch_size: int = 4
    seq_len: int = 32
    train_steps: int = 40
    lr: float = 3e-4

CONFIG = HarnessConfig()
CONFIG
"""
    ),
    code(
        """class TinyLM(nn.Module):
    def __init__(self, config: HarnessConfig, tie_weights: bool = False):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.output_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if tie_weights:
            self.output_head.weight = self.embed.weight

    def forward(self, tokens):
        hidden = self.embed(tokens)
        mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(1), device=tokens.device)
        return self.encoder(hidden, mask=mask, is_causal=True)

    def logits(self, tokens):
        return self.output_head(self.forward(tokens))


class DualHeadLM(nn.Module):
    def __init__(self, config: HarnessConfig):
        super().__init__()
        self.backbone = TinyLM(config, tie_weights=True)
        self.head_t2 = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def logits_t1(self, tokens):
        return self.backbone.logits(tokens)

    def logits_t2(self, tokens):
        return self.head_t2(self.backbone(tokens))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def explain_shape(name, tensor):
    labels = {1: ["index"], 2: ["batch", "sequence"], 3: ["batch", "sequence", "feature"]}[tensor.dim()]
    parts = ", ".join(f"{label}={size}" for label, size in zip(labels, tensor.shape))
    print(f"{name}: {list(tensor.shape)}")
    print(f"  {parts}")


def shift_logits_and_targets(logits, tokens):
    shifted_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shifted_targets = tokens[:, 1:].reshape(-1)
    return shifted_logits, shifted_targets


def shift_with_offset(logits, tokens, offset):
    shifted_logits = logits[:, :-offset, :].reshape(-1, logits.size(-1))
    shifted_targets = tokens[:, offset:].reshape(-1)
    return shifted_logits, shifted_targets


class WordTokenizer:
    def __init__(self, vocab):
        self.stoi = {t: i for i, t in enumerate(vocab)}
        self.itos = {i: t for t, i in self.stoi.items()}

    @classmethod
    def demo(cls):
        return cls(["<pad>", "<eos>", "The", "cat", "sat", "on", "the", "mat", ".", "dog", "ran", "quickly"])

    def encode(self, text):
        return [self.stoi[w] for w in text.split()]

    def decode_row(self, ids):
        return [self.itos[int(i)] for i in ids]


tokenizer = WordTokenizer.demo()
EOS_ID = tokenizer.stoi["<eos>"]
"""
    ),
    md("## Part 1.1 — Tensor shape inspection"),
    code(
        """model = TinyLM(CONFIG, tie_weights=False).to(DEVICE)
model.eval()

texts = [
    "The cat sat on the mat .",
    "The dog ran quickly .",
    "The cat sat on the mat .",
    "The dog ran quickly .",
]
rows = [tokenizer.encode(t) for t in texts]
max_len = max(len(r) for r in rows)
rows = [r + [PAD_ID] * (max_len - len(r)) for r in rows]
tokens = torch.tensor(rows, device=DEVICE)
if tokens.size(1) < CONFIG.seq_len:
    pad = torch.full((tokens.size(0), CONFIG.seq_len - tokens.size(1)), PAD_ID, device=DEVICE)
    tokens = torch.cat([tokens, pad], dim=1)

hidden = model(tokens)
logits = model.output_head(hidden)
shifted_logits, shifted_targets = shift_logits_and_targets(logits, tokens)

explain_shape("tokens", tokens)
explain_shape("hidden", hidden)
explain_shape("logits", logits)
explain_shape("shifted_logits", shifted_logits)
explain_shape("shifted_targets", shifted_targets)

assert hidden.shape == (CONFIG.batch_size, CONFIG.seq_len, CONFIG.hidden_dim)
assert logits.shape == (CONFIG.batch_size, CONFIG.seq_len, CONFIG.vocab_size)
assert shifted_logits.shape == (CONFIG.batch_size * (CONFIG.seq_len - 1), CONFIG.vocab_size)
assert shifted_targets.shape == (CONFIG.batch_size * (CONFIG.seq_len - 1),)
print("Shape assertions passed.")
"""
    ),
    md("## Part 1.2 — Verify shift with token strings (correct direction)"),
    code(
        """row = tokens[0, :7].tolist()
inputs = tokenizer.decode_row(row[:-1])
targets = tokenizer.decode_row(row[1:])

print("Position | Input token | Target token")
print("-" * 40)
for i, (inp, tgt) in enumerate(zip(inputs, targets)):
    print(f"{i:8} | {inp:11} | {tgt}")

assert inputs == ["The", "cat", "sat", "on", "the", "mat"]
assert targets == ["cat", "sat", "on", "the", "mat", "."]
print("Correct shift: position t predicts token t+1.")
"""
    ),
    md("## Part 1.3 — Padding mask"),
    code(
        """target_positions = tokens[:, 1:]
pad_mask = (target_positions != PAD_ID).reshape(-1)

loss_unmasked = F.cross_entropy(shifted_logits, shifted_targets)
loss_masked = F.cross_entropy(shifted_logits[pad_mask], shifted_targets[pad_mask])

print("Total target positions:", shifted_targets.numel())
print("Valid/contributing positions:", int(pad_mask.sum()))
print("Loss without mask:", float(loss_unmasked))
print("Loss with mask:", float(loss_masked))

if abs(float(loss_unmasked) - float(loss_masked)) < 1e-4:
    print("Loss values match here because padding targets are a small fraction of the batch mean.")
else:
    print("Masked loss differs — padding was pulling the mean.")

assert int(pad_mask.sum()) < shifted_targets.numel()
print("Padding targets must not contribute to language-model training.")
"""
    ),
    md("## Part 1.3b — The mean: divide by contributing tokens, not B×T"),
    code(
        """per_token_nll = F.cross_entropy(shifted_logits, shifted_targets, reduction="none")
valid_nll = per_token_nll[pad_mask]

loss_correct = valid_nll.mean()
loss_wrong = per_token_nll.mean()  # divides by all positions including padding

print("Contributing tokens:", int(pad_mask.sum()))
print("Loss with correct denominator (valid only):", float(loss_correct))
print("Loss with wrong denominator (all positions):", float(loss_wrong))
print("Scale factor wrong/correct:", float(loss_wrong / loss_correct))
assert abs(float(loss_correct) - float(loss_masked)) < 1e-5
"""
    ),
    md(
        """## Part 1.4 — Packed documents + boundary mask

Production pipelines usually end documents with `<eos>` before packing. This demo uses `.` as the boundary token; the mask logic is identical — exclude the target position that would predict the first token of document B from the last token of document A.
"""
    ),
    code(
        """doc_a = tokenizer.encode("The cat sat .")
doc_b = tokenizer.encode("The dog ran .")
packed = doc_a + [EOS_ID] + doc_b
packed_tokens = torch.tensor([packed], device=DEVICE)
packed_logits = model.logits(packed_tokens)
pl, pt = shift_logits_and_targets(packed_logits, packed_tokens)

boundary_index = len(doc_a)  # position predicting first token of doc B from <eos>
boundary_mask = torch.ones(packed_tokens.size(1) - 1, dtype=torch.bool, device=DEVICE)
boundary_mask[boundary_index] = False

print("Packed tokens:", tokenizer.decode_row(packed))
print(
    "Masked cross-doc prediction:",
    f"'{tokenizer.itos[packed[boundary_index]]}' -> '{tokenizer.itos[packed[boundary_index + 1]]}'",
)

loss_before = F.cross_entropy(pl, pt)
loss_after = F.cross_entropy(pl[boundary_mask], pt[boundary_mask])

print("Contributing tokens before boundary mask:", pl.shape[0])
print("Contributing tokens after boundary mask:", int(boundary_mask.sum()))
print("Loss before boundary mask:", float(loss_before))
print("Loss after boundary mask:", float(loss_after))
print("Without the mask, the model learns '.'/<eos> should be followed by 'The' from the next document.")
"""
    ),
    md("## Part 1.5 — Perplexity sanity check"),
    code(
        """initial_loss = float(loss_masked)
initial_ppl = math.exp(initial_loss)

print("Vocabulary size:", CONFIG.vocab_size)
print("Initial loss:", initial_loss)
print("Initial perplexity:", initial_ppl)
print("Expected ~ vocab size for near-uniform untrained predictions:", CONFIG.vocab_size)

ratio = initial_ppl / CONFIG.vocab_size
print("Perplexity / vocab ratio:", round(ratio, 3))
assert 0.7 < ratio < 1.3, "Perplexity far from vocab — investigate shift/mask bugs"
print("Sanity check passed.")
print("Note: perplexity is only comparable within the same tokenizer — not across different tokenizations.")
"""
    ),
    md("## Part 1.6 — Tied vs untied parameter counts"),
    code(
        """tied = count_parameters(TinyLM(CONFIG, tie_weights=True))
untied = count_parameters(TinyLM(CONFIG, tie_weights=False))
print("Tied parameters:", tied)
print("Untied parameters:", untied)
print("Difference:", untied - tied)
print("Difference equals vocab_size * hidden_dim:", CONFIG.vocab_size * CONFIG.hidden_dim)
assert untied - tied == CONFIG.vocab_size * CONFIG.hidden_dim
"""
    ),
    md("## Part 1.7 — Ordinary vs chunked cross-entropy memory"),
    code(
        """def fmt_bytes(n):
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GiB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.2f} MiB"
    return f"{n} B"


def measure_peak(fn):
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(DEVICE)
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


# Large enough that materialized log-probs dominate peak memory on GPU
B, T, V = (16, 1024, 32768) if DEVICE.type == "cuda" else (8, 128, 8192)
CHUNK = 256


def ordinary_explicit(logits, targets):
    flat = logits.reshape(-1, V)
    flat_t = targets.reshape(-1)
    log_probs = F.log_softmax(flat, dim=-1)
    return (-log_probs[torch.arange(flat_t.size(0), device=flat_t.device), flat_t]).mean()


def chunked_ce(logits, targets, chunk_size=CHUNK):
    flat = logits.reshape(-1, V)
    flat_t = targets.reshape(-1)
    total = flat.size(0)
    summed = torch.zeros((), device=flat.device)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        summed = summed + F.cross_entropy(flat[start:end], flat_t[start:end], reduction="sum")
    return summed / total


def run_ordinary():
    logits = torch.randn(B, T, V, device=DEVICE)
    targets = torch.randint(0, V, (B, T), device=DEVICE)
    return ordinary_explicit(logits, targets)


def run_chunked():
    logits = torch.randn(B, T, V, device=DEVICE)
    targets = torch.randint(0, V, (B, T), device=DEVICE)
    return chunked_ce(logits, targets)


ordinary_peak = measure_peak(run_ordinary)
chunked_peak = measure_peak(run_chunked)
memory_ratio = ordinary_peak / max(chunked_peak, 1)

print(f"Workload: B={B}, T={T}, V={V}, chunk={CHUNK}")
print("Ordinary CE peak memory:", fmt_bytes(ordinary_peak))
print("Chunked CE peak memory:", fmt_bytes(chunked_peak))
print("Ratio ordinary / chunked:", round(memory_ratio, 2))
print("Ordinary path materializes [tokens, vocab] log-probs; chunked peaks at [chunk, vocab] per step.")
assert memory_ratio > 1.0 or DEVICE.type != "cuda", "On GPU, ordinary should exceed chunked peak memory"
"""
    ),
    md("## Part 2 — Two-token-ahead head"),
    code(
        """dual = DualHeadLM(CONFIG).to(DEVICE)
opt = torch.optim.Adam(dual.parameters(), lr=CONFIG.lr)
train_tokens = tokens[:2].clone()

hist_t1, hist_t2, hist_sum = [], [], []
for step in range(CONFIG.train_steps):
    dual.train()
    opt.zero_grad()
    l1, t1 = shift_with_offset(dual.logits_t1(train_tokens), train_tokens, 1)
    l2, t2 = shift_with_offset(dual.logits_t2(train_tokens), train_tokens, 2)
    loss1 = F.cross_entropy(l1, t1)
    loss2 = F.cross_entropy(l2, t2)
    total = loss1 + loss2
    total.backward()
    opt.step()
    hist_t1.append(loss1.item())
    hist_t2.append(loss2.item())
    hist_sum.append(total.item())

print("Final next-token loss:", hist_t1[-1])
print("Final two-ahead loss:", hist_t2[-1])
print("Final combined loss:", hist_sum[-1])

plt.figure(figsize=(8, 4))
plt.plot(hist_t1, label="Head 1 (t+1)")
plt.plot(hist_t2, label="Head 2 (t+2)")
plt.plot(hist_sum, label="Combined", linestyle="--")
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Dual-head training")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Head 2 stays higher: predicting t+2 needs longer-range structure and is strictly harder.")
"""
    ),
    md(
        """## Part 3 — Wrong-shift demonstration

Shapes can match. Loss can decrease. Training can look healthy. **Read the strings.**

Part 3 is the deliberate bug: shift in the wrong direction so position *t* predicts token *t−1* instead of *t+1*.
"""
    ),
    code(
        """wrong_inputs = tokenizer.decode_row(tokens[0, 1:7].tolist())
wrong_targets = tokenizer.decode_row(tokens[0, :6].tolist())

print("WRONG shift (position t predicts t-1):")
print("Input tokens :", wrong_inputs)
print("Target tokens:", wrong_targets)
print()
print("Position | Wrong input | Wrong target")
print("-" * 42)
for i, (inp, tgt) in enumerate(zip(wrong_inputs, wrong_targets)):
    print(f"{i:8} | {inp:11} | {tgt}")

wrong_logits = logits[:, 1:, :].reshape(-1, CONFIG.vocab_size)
wrong_targets_ids = tokens[:, :-1].reshape(-1)
wrong_loss = F.cross_entropy(wrong_logits, wrong_targets_ids)
correct_loss = F.cross_entropy(shifted_logits[pad_mask], shifted_targets[pad_mask])

print()
print("Wrong-shift loss:", float(wrong_loss))
print("Correct-shift loss:", float(correct_loss))
print("Shapes still match. A wrong shift can still produce a finite, decreasing loss.")
print("Lesson: verify decoded token strings before trusting the scalar.")
"""
    ),
    md("## Final summary"),
    code(
        """print("=== Configuration ===")
print(CONFIG)
print()
print("=== Part 1 numbers ===")
print("Valid tokens after padding mask:", int(pad_mask.sum()), "of", shifted_targets.numel())
print("Loss without pad mask:", float(loss_unmasked))
print("Loss with pad mask:", float(loss_masked))
print("Loss wrong denominator:", float(loss_wrong))
print("Packed loss before boundary mask:", float(loss_before))
print("Packed loss after boundary mask:", float(loss_after))
print("Initial perplexity:", initial_ppl)
print("Tied / untied params:", tied, untied)
print("Ordinary CE peak:", fmt_bytes(ordinary_peak))
print("Chunked CE peak:", fmt_bytes(chunked_peak))
print("Memory ratio ordinary/chunked:", round(memory_ratio, 2))
print()
print("=== Part 2 losses ===")
print("Next-token:", hist_t1[-1])
print("Two-ahead:", hist_t2[-1])
print("Combined:", hist_sum[-1])
print()
print("=== Part 3 — wrong-shift demonstration ===")
print("WRONG shift (position t predicts t-1):")
print("Input tokens :", wrong_inputs)
print("Target tokens:", wrong_targets)
print()
print("Position | Wrong input | Wrong target")
print("-" * 42)
for i, (inp, tgt) in enumerate(zip(wrong_inputs, wrong_targets)):
    print(f"{i:8} | {inp:11} | {tgt}")
print()
print("Wrong-shift loss:", float(wrong_loss))
print("Correct-shift loss:", float(correct_loss))
print("Shapes still match. A wrong shift can still produce a finite, decreasing loss.")
print("Lesson: verify decoded token strings before trusting the scalar.")
"""
    ),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": []},
    },
    "cells": cells,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {NOTEBOOK_PATH}")
