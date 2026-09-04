#!/usr/bin/env python3
"""Run all experiments and write results.json + figures."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from src.config import TrainConfig
from src.float_repr import represent_0_1
from src.mfu import measure_mfu
from src.model import TinyCharLM
from src.training import (
    explain_shapes,
    finite_difference_check,
    find_grad_before_loss,
    lm_loss,
    plot_accumulation,
    plot_training_log,
    train_with_logging,
)

ROOT = Path(__file__).parent
FIG = ROOT / "figures"


def main() -> None:
    FIG.mkdir(exist_ok=True)
    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    model = TinyCharLM(config).to(device)
    tokens = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len), device=device)
    logits = model(tokens)
    loss = lm_loss(logits, tokens)

    grad_check = finite_difference_check(model, tokens)
    assert grad_check.rel_diff < 0.05, f"Grad check failed: {grad_check}"

    # micro-batches: 10 vs 100 tokens (assignment example)
    micro_a = torch.randint(0, config.vocab_size, (2, 11), device=device)  # 10 targets
    micro_b = torch.randint(0, config.vocab_size, (2, 101), device=device)  # 100 targets
    wrong, correct = [], []
    la = lm_loss(model(micro_a), micro_a)
    lb = lm_loss(model(micro_b), micro_b)
    na = micro_a.size(0) * (micro_a.size(1) - 1)
    nb = micro_b.size(0) * (micro_b.size(1) - 1)
    wrong_val = ((la + lb) / 2).item()
    correct_val = ((la * na + lb * nb) / (na + nb)).item()
    # curves over training steps with drifting weights
    train_model = TinyCharLM(config).to(device)
    opt = torch.optim.SGD(train_model.parameters(), lr=0.05)
    wrong_curve, correct_curve = [], []
    for _ in range(40):
        opt.zero_grad()
        la = lm_loss(train_model(micro_a), micro_a)
        lb = lm_loss(train_model(micro_b), micro_b)
        wrong_curve.append(((la + lb) / 2).item())
        correct_curve.append(((la * na + lb * nb) / (na + nb)).item())
        ((la + lb) / 2).backward()
        opt.step()

    plot_accumulation(wrong_curve, correct_curve, str(FIG / "gradient_accumulation.png"))

    log = train_with_logging(config, device)
    plot_training_log(log, str(FIG / "loss_vs_step.png"), str(FIG / "grad_norm_vs_step.png"))
    grad_event = find_grad_before_loss(log)

    mfu = measure_mfu(config, device)
    floats = [asdict(f) for f in represent_0_1()]

    results = {
        "config": asdict(config),
        "device": str(device),
        "param_count": model.param_count(),
        "shape_explanations": explain_shapes(tokens, logits, loss),
        "grad_check": asdict(grad_check),
        "accumulation": {
            "micro_a_tokens": int(na),
            "micro_b_tokens": int(nb),
            "wrong_single_pair": wrong_val,
            "correct_single_pair": correct_val,
            "gap": abs(wrong_val - correct_val),
        },
        "grad_norm_event": grad_event,
        "mfu": asdict(mfu),
        "float_0_1": floats,
        "precision_recommendation": "BF16 for training matmuls with FP32 accumulation for optimizer states",
    }

    out = ROOT / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
