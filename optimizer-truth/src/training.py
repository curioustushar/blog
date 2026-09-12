"""Training loops, update ratios, schedule comparison, LR sweeps."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn

from .config import TrainConfig
from .model import WidthMLP
from .schedules import lr_cosine, lr_wsd, schedule_curve


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def make_teacher(input_dim: int, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed + 999)
    return torch.randn(input_dim, input_dim, device=device, generator=g) * 0.1


def synthetic_batch(
    batch_size: int,
    input_dim: int,
    device: torch.device,
    teacher: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, input_dim, device=device)
    return x, x @ teacher


def train_steps(
    model: nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    schedule: str,
    log_update_ratio: bool = False,
) -> dict[str, Any]:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()
    teacher = make_teacher(cfg.input_dim, device, cfg.seed)

    losses: list[float] = []
    lrs: list[float] = []
    ratios_by_layer: dict[str, list[float]] = {}

    batch_gen = torch.Generator(device=device)
    batch_gen.manual_seed(cfg.seed + 7)

    for step in range(cfg.train_steps):
        if schedule == "cosine":
            lr = lr_cosine(step, cfg.train_steps, cfg.base_lr, cfg.min_lr, cfg.warmup_steps)
        else:
            lr = lr_wsd(
                step,
                cfg.base_lr,
                cfg.min_lr,
                cfg.warmup_steps,
                cfg.wsd_decay_start,
                cfg.wsd_decay_steps,
            )
        for pg in opt.param_groups:
            pg["lr"] = lr

        x = torch.randn(
            cfg.batch_size, cfg.input_dim, device=device, generator=batch_gen
        )
        y = x @ teacher
        opt.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        weights_before = {}
        if log_update_ratio:
            for name, p in model.named_parameters():
                weights_before[name] = p.detach().clone()

        opt.step()

        if log_update_ratio:
            for name, p in model.named_parameters():
                w_b = weights_before[name]
                upd = (p.detach() - w_b).norm().item()
                w_n = w_b.norm().item()
                ratios_by_layer.setdefault(name, []).append(upd / (w_n + 1e-12))

        losses.append(loss.item())
        lrs.append(lr)

    out: dict[str, Any] = {
        "schedule": schedule,
        "losses": losses,
        "lrs": lrs,
        "loss_at_eval": losses[cfg.eval_step - 1],
        "lr_at_eval": lrs[cfg.eval_step - 1],
    }
    if log_update_ratio:
        out["update_to_weight_ratio"] = ratios_by_layer
    return out


def warmup_stops_changing_ratio(
    ratios: dict[str, list[float]],
    warmup_steps: int,
    window: int = 5,
    rel_change_threshold: float = 0.05,
) -> dict[str, Any]:
    """
    After warmup, find first step where rolling mean of max layer ratio
    changes by less than rel_change_threshold vs previous window.
    """
    # aggregate: max ratio across layers per step
    n_steps = len(next(iter(ratios.values())))
    max_ratio = []
    for t in range(n_steps):
        max_ratio.append(max(ratios[k][t] for k in ratios))

    def rolling_mean(i: int) -> float:
        s = max(0, i - window + 1)
        chunk = max_ratio[s : i + 1]
        return sum(chunk) / len(chunk)

    first_stable = None
    for i in range(warmup_steps, n_steps):
        prev = rolling_mean(i - 1)
        curr = rolling_mean(i)
        if prev > 0 and abs(curr - prev) / prev < rel_change_threshold:
            first_stable = i
            break

    return {
        "criterion": (
            f"after warmup, rolling-{window} mean of max layer ||Δw||/||w|| "
            f"changes < {rel_change_threshold*100:.0f}% step-to-step"
        ),
        "warmup_steps": warmup_steps,
        "step_warmup_effect_negligible": first_stable,
        "max_ratio_per_step": max_ratio,
    }


def lr_sweep_width(
    width: int,
    cfg: TrainConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    results = []
    for lr in cfg.sweep_lrs:
        set_seed(cfg.seed)
        model = WidthMLP(cfg.input_dim, width, cfg.output_dim)
        teacher = make_teacher(cfg.input_dim, device, cfg.seed)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
        criterion = nn.MSELoss()
        final_loss = 0.0
        for step in range(cfg.sweep_steps):
            x, y = synthetic_batch(cfg.batch_size, cfg.input_dim, device, teacher)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            final_loss = loss.item()
        results.append({"lr": lr, "final_loss": final_loss, "width": width})
    return results


def extrapolate_lr(widths: list[int], best_lrs: list[float], target: int = 4096) -> dict[str, Any]:
    """Fit log(lr) vs log(width) — common muP-style 1/width => slope -1 in log-log."""
    import math

    log_w = [math.log(w) for w in widths]
    log_lr = [math.log(lr) for lr in best_lrs]
    n = len(widths)
    mean_w = sum(log_w) / n
    mean_lr = sum(log_lr) / n
    num = sum((log_w[i] - mean_w) * (log_lr[i] - mean_lr) for i in range(n))
    den = sum((log_w[i] - mean_w) ** 2 for i in range(n))
    slope = num / den if den > 0 else -1.0
    intercept = mean_lr - slope * mean_w
    pred_log_lr = intercept + slope * math.log(target)
    pred_lr = math.exp(pred_log_lr)
    return {
        "widths": widths,
        "best_lrs": best_lrs,
        "log_log_slope": slope,
        "intercept_log_lr": intercept,
        "predicted_lr_width_4096": pred_lr,
        "confidence_note": (
            "Low–medium: only 3 widths, short sweeps, synthetic data; "
            "slope is indicative not proof of muP transfer."
        ),
    }


def plot_bias_correction(data: dict[str, Any], path: str) -> None:
    steps = list(range(len(data["with_bias_correction"])))
    plt.figure(figsize=(8, 5))
    plt.plot(steps, data["with_bias_correction"], label="With bias correction", linewidth=2)
    plt.plot(steps, data["without_bias_correction"], label="Without bias correction", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Weight value")
    plt.title("Adam: first 20 steps with vs without bias correction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_schedules(
    cosine_lrs: list[float],
    wsd_lrs: list[float],
    cosine_loss: list[float],
    wsd_loss: list[float],
    eval_step: int,
    path_lr: str,
    path_loss: str,
) -> None:
    steps = list(range(len(cosine_lrs)))
    plt.figure(figsize=(8, 5))
    plt.plot(steps, cosine_lrs, label="Cosine", linewidth=2)
    plt.plot(steps, wsd_lrs, label="WSD", linewidth=2)
    plt.axvline(eval_step - 1, color="gray", linestyle="--", label=f"Eval @ step {eval_step}")
    plt.xlabel("Step")
    plt.ylabel("Learning rate")
    plt.title("Learning-rate schedules")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_lr, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(steps, cosine_loss, label="Cosine", linewidth=2)
    plt.plot(steps, wsd_loss, label="WSD", linewidth=2)
    plt.axvline(eval_step - 1, color="gray", linestyle="--", label=f"Eval @ step {eval_step}")
    plt.xlabel("Step")
    plt.ylabel("Training loss (MSE)")
    plt.title("Cosine vs WSD training loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_loss, dpi=150)
    plt.close()


def plot_update_ratios(ratios: dict[str, list[float]], warmup_steps: int, path: str) -> None:
    plt.figure(figsize=(9, 5))
    for name, series in ratios.items():
        plt.plot(series, label=name, alpha=0.85)
    plt.axvline(warmup_steps, color="gray", linestyle="--", label="Warmup end")
    plt.xlabel("Step")
    plt.ylabel("||update|| / ||weight||")
    plt.title("Update-to-weight ratio per layer")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_lr_sweep(all_results: list[dict[str, Any]], path: str) -> None:
    plt.figure(figsize=(8, 5))
    for width in sorted({r["width"] for r in all_results}):
        pts = sorted([r for r in all_results if r["width"] == width], key=lambda x: x["lr"])
        lrs = [p["lr"] for p in pts]
        losses = [p["final_loss"] for p in pts]
        plt.plot(lrs, losses, marker="o", label=f"width={width}", linewidth=2)
        best = min(pts, key=lambda x: x["final_loss"])
        plt.scatter([best["lr"]], [best["final_loss"]], s=120, zorder=5, edgecolors="black")
        plt.annotate(
            f"min lr={best['lr']:.0e}",
            (best["lr"], best["final_loss"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel(f"Final loss ({all_results[0].get('note', 'MSE')})")
    plt.title("LR sweep by model width")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2))
