#!/usr/bin/env python3
"""Run all optimizer-truth experiments."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from src.adam_manual import bias_correction_trajectory, compare_manual_pytorch
from src.config import AdamHandConfig, TrainConfig
from src.model import WidthMLP
from src.schedules import schedule_curve
from src.training import (
    extrapolate_lr,
    lr_sweep_width,
    plot_bias_correction,
    plot_lr_sweep,
    plot_schedules,
    plot_update_ratios,
    set_seed,
    train_steps,
    warmup_stops_changing_ratio,
)

ROOT = Path(__file__).parent
FIG = ROOT / "figures"
OUT = ROOT / "outputs"


def main() -> None:
    FIG.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = TrainConfig()
    adam_cfg = AdamHandConfig()

    # 1. Manual Adam vs PyTorch
    adam_cmp = compare_manual_pytorch(adam_cfg)
    assert adam_cmp["max_abs_weight_diff"] < 1e-5, adam_cmp

    # 2. Bias correction
    bias_data = bias_correction_trajectory(adam_cfg, num_steps=20)
    bias_extended = bias_correction_trajectory(adam_cfg, num_steps=200)
    plot_bias_correction(bias_data, str(FIG / "bias_correction.png"))

    # 3. Update ratio + warmup
    set_seed(train_cfg.seed)
    width = 256
    model = WidthMLP(train_cfg.input_dim, width, train_cfg.output_dim)
    ratio_run = train_steps(model, train_cfg, device, "cosine", log_update_ratio=True)
    warmup_analysis = warmup_stops_changing_ratio(
        ratio_run["update_to_weight_ratio"],
        train_cfg.warmup_steps,
    )
    plot_update_ratios(
        ratio_run["update_to_weight_ratio"],
        train_cfg.warmup_steps,
        str(FIG / "update_to_weight_ratio.png"),
    )
    with open(OUT / "update_ratios.json", "w") as f:
        json.dump(ratio_run["update_to_weight_ratio"], f)

    # 4. Cosine vs WSD (same init)
    set_seed(train_cfg.seed)
    m_cos = WidthMLP(train_cfg.input_dim, width, train_cfg.output_dim)
    set_seed(train_cfg.seed)
    m_wsd = WidthMLP(train_cfg.input_dim, width, train_cfg.output_dim)
    cos_run = train_steps(m_cos, train_cfg, device, "cosine")
    wsd_run = train_steps(m_wsd, train_cfg, device, "wsd")
    plot_schedules(
        cos_run["lrs"],
        wsd_run["lrs"],
        cos_run["losses"],
        wsd_run["losses"],
        train_cfg.eval_step,
        str(FIG / "schedules_lr.png"),
        str(FIG / "cosine_vs_wsd_loss.png"),
    )

    # 5. LR sweep
    sweep_rows: list[dict] = []
    best_by_width: dict[int, float] = {}
    for w in train_cfg.sweep_widths:
        rows = lr_sweep_width(w, train_cfg, device)
        sweep_rows.extend(rows)
        best = min(rows, key=lambda r: r["final_loss"])
        best_by_width[w] = best["lr"]
    plot_lr_sweep(sweep_rows, str(FIG / "lr_sweep.png"))
    extrap = extrapolate_lr(
        list(train_cfg.sweep_widths),
        [best_by_width[w] for w in train_cfg.sweep_widths],
        4096,
    )

    winner = "cosine" if cos_run["loss_at_eval"] <= wsd_run["loss_at_eval"] else "wsd"

    results = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "adam_hand_verify": adam_cmp,
        "bias_correction_plot_20_steps": bias_data,
        "bias_correction_convergence": {
            "empirical": {
                "criterion": bias_extended["criterion_empirical"],
                "first_step": bias_extended["first_step_empirical_criterion_met"],
            },
            "analytic": {
                "criterion": bias_extended["criterion_analytic"],
                "first_step": bias_extended["first_step_analytic_criterion_met"],
            },
            "max_rel_update_diff_first_20": max(bias_data["relative_update_diff"][1:]),
        },
        "warmup_ratio_analysis": warmup_analysis,
        "cosine_vs_wsd": {
            "eval_step": train_cfg.eval_step,
            "cosine_loss_at_eval": cos_run["loss_at_eval"],
            "wsd_loss_at_eval": wsd_run["loss_at_eval"],
            "cosine_lr_at_eval": cos_run["lr_at_eval"],
            "wsd_lr_at_eval": wsd_run["lr_at_eval"],
            "winner_at_eval": winner,
            "fair_tuning": (
                "Same seed, batch, width=256, AdamW, base_lr, warmup_steps; "
                "WSD decay_start=200 matches eval step so WSD still at peak LR at step 200 "
                "while cosine is mid-decay — schedules are not hyperparameter-tuned separately "
                "here; a fair paper comparison would grid-search decay_start and cosine floor jointly."
            ),
        },
        "lr_sweep": {
            "steps_per_run": train_cfg.sweep_steps,
            "best_lr_by_width": best_by_width,
            "all_runs": sweep_rows,
            "extrapolation_4096": extrap,
        },
        "conclusions": {
            "adam_matches_pytorch": adam_cmp["max_abs_weight_diff"] < 1e-5,
            "bias_correction_steps_until_negligible_analytic": bias_extended[
                "first_step_analytic_criterion_met"
            ],
            "bias_correction_steps_until_negligible_empirical": bias_extended[
                "first_step_empirical_criterion_met"
            ],
            "warmup_ratio_step": warmup_analysis["step_warmup_effect_negligible"],
            "keep_model_at_200": winner,
            "lr_4096": extrap["predicted_lr_width_4096"],
        },
    }

    out_path = ROOT / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(results["conclusions"], indent=2))


if __name__ == "__main__":
    main()
