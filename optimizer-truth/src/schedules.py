"""Learning-rate schedules: warmup, cosine, WSD."""

from __future__ import annotations

import math


def lr_warmup(step: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, (step + 1) / warmup_steps)


def lr_cosine(
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    if step < warmup_steps:
        return lr_warmup(step, warmup_steps, base_lr)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def lr_wsd(
    step: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    decay_start: int,
    decay_steps: int,
) -> float:
    if step < warmup_steps:
        return lr_warmup(step, warmup_steps, base_lr)
    if step < decay_start:
        return base_lr
    t = step - decay_start
    if t >= decay_steps:
        return min_lr
    progress = t / decay_steps
    return min_lr + (base_lr - min_lr) * (1.0 - progress)


def schedule_curve(
    kind: str,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    wsd_decay_start: int,
    wsd_decay_steps: int,
) -> list[float]:
    lrs = []
    for step in range(total_steps):
        if kind == "cosine":
            lr = lr_cosine(step, total_steps, base_lr, min_lr, warmup_steps)
        elif kind == "wsd":
            lr = lr_wsd(
                step, base_lr, min_lr, warmup_steps, wsd_decay_start, wsd_decay_steps
            )
        else:
            raise ValueError(kind)
        lrs.append(lr)
    return lrs
