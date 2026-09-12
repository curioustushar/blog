"""Shared experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdamHandConfig:
    lr: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    initial_weight: float = 1.0
    gradients: tuple[float, ...] = (0.5, -0.3, 0.1, 0.4, -0.2)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    input_dim: int = 64
    output_dim: int = 64
    train_steps: int = 300
    eval_step: int = 200
    base_lr: float = 3e-3
    min_lr: float = 3e-5
    warmup_steps: int = 50
    weight_decay: float = 0.0
    # WSD: warmup -> stable -> decay (decay starts at decay_start)
    wsd_decay_start: int = 200
    wsd_decay_steps: int = 100
    # LR sweep
    sweep_steps: int = 80
    sweep_widths: tuple[int, ...] = (256, 512, 1024)
    sweep_lrs: tuple[float, ...] = (
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
    )
