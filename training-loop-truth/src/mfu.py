"""MFU estimation utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .config import TrainConfig
from .model import TinyCharLM


@dataclass(frozen=True)
class MFUReport:
    num_params: int
    tokens_per_step: int
    steps_timed: int
    seconds_elapsed: float
    tokens_per_second: float
    flops_per_step: float
    achieved_flops_per_sec: float
    peak_flops_per_sec: float
    mfu_percent: float
    hardware_note: str
    bottleneck_notes: list[str]


def estimate_flops_per_step(num_params: int, tokens_per_step: int) -> float:
    """Standard rough estimate: 6 * N * tokens for forward+backward."""
    return 6.0 * num_params * tokens_per_step


def measure_mfu(config: TrainConfig, device: torch.device, peak_tflops: float = 15.0) -> MFUReport:
    """Time training steps and estimate MFU. peak_tflops is hardware estimate (T4 ~15)."""
    torch.manual_seed(config.seed)
    model = TinyCharLM(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    tokens = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len), device=device)

    # warmup
    for _ in range(5):
        opt.zero_grad()
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, config.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
        loss.backward()
        opt.step()

    steps = 30
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, config.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens_per_step = config.batch_size * (config.seq_len - 1)
    tps = steps * tokens_per_step / elapsed
    flops_step = estimate_flops_per_step(model.param_count(), tokens_per_step)
    achieved = flops_step * steps / elapsed
    peak = peak_tflops * 1e12
    mfu = 100.0 * achieved / peak

    notes = [
        "Small model → low GPU occupancy",
        "Tiny batch (8×15 tokens) → kernel launch overhead dominates",
        "Python loop + autograd interpreter overhead",
        "CPU dataloader absent but tensor ops still small",
        "No fused kernels / no torch.compile",
    ]
    if device.type == "cpu":
        notes.append("Running on CPU — MFU vs GPU peak is illustrative only")

    return MFUReport(
        num_params=model.param_count(),
        tokens_per_step=tokens_per_step,
        steps_timed=steps,
        seconds_elapsed=elapsed,
        tokens_per_second=tps,
        flops_per_step=flops_step,
        achieved_flops_per_sec=achieved,
        peak_flops_per_sec=peak,
        mfu_percent=mfu,
        hardware_note=f"{device} (peak assumed {peak_tflops} TFLOP/s for reference GPU)",
        bottleneck_notes=notes,
    )
