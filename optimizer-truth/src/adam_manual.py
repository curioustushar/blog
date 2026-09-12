"""Manual Adam vs PyTorch for a single scalar weight."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .config import AdamHandConfig


@dataclass
class AdamStepRecord:
    step: int
    grad: float
    m: float
    v: float
    m_hat: float
    v_hat: float
    weight_before: float
    update: float
    weight_after: float


def manual_adam_steps(cfg: AdamHandConfig, bias_correction: bool = True) -> list[AdamStepRecord]:
    w = cfg.initial_weight
    m = 0.0
    v = 0.0
    records: list[AdamStepRecord] = []
    for t, g in enumerate(cfg.gradients, start=1):
        m = cfg.beta1 * m + (1.0 - cfg.beta1) * g
        v = cfg.beta2 * v + (1.0 - cfg.beta2) * (g * g)
        if bias_correction:
            m_hat = m / (1.0 - cfg.beta1**t)
            v_hat = v / (1.0 - cfg.beta2**t)
        else:
            m_hat = m
            v_hat = v
        update = cfg.lr * m_hat / (math.sqrt(v_hat) + cfg.eps)
        w_before = w
        w = w - update
        records.append(
            AdamStepRecord(
                step=t,
                grad=g,
                m=m,
                v=v,
                m_hat=m_hat,
                v_hat=v_hat,
                weight_before=w_before,
                update=update,
                weight_after=w,
            )
        )
    return records


def pytorch_adam_steps(cfg: AdamHandConfig, bias_correction: bool = True) -> list[dict[str, float]]:
    w = torch.tensor(cfg.initial_weight, dtype=torch.float64, requires_grad=False)
    param = torch.nn.Parameter(w.clone())
    opt = torch.optim.Adam(
        [param],
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=0.0,
    )
    # PyTorch 2.x: bias correction can be disabled via foreach or we mirror state manually
    # Use capturable=False; for bias_correction=False we implement custom step reading state
    if not bias_correction:
        return _pytorch_adam_no_bias(opt, param, cfg)
    rows: list[dict[str, float]] = []
    for t, g in enumerate(cfg.gradients, start=1):
        opt.zero_grad()
        param.grad = torch.tensor(g, dtype=torch.float64)
        opt.step()
        state = opt.state[param]
        m_t = state["exp_avg"].item()
        v_t = state["exp_avg_sq"].item()
        m_hat = m_t / (1.0 - cfg.beta1**t)
        v_hat = v_t / (1.0 - cfg.beta2**t)
        rows.append(
            {
                "step": t,
                "grad": g,
                "m": m_t,
                "v": v_t,
                "m_hat": m_hat,
                "v_hat": v_hat,
                "weight_after": param.item(),
            }
        )
    return rows


def _pytorch_adam_no_bias(
    opt: torch.optim.Adam, param: torch.nn.Parameter, cfg: AdamHandConfig
) -> list[dict[str, float]]:
    """Adam step without bias correction (use raw m, v in denominator)."""
    rows: list[dict[str, float]] = []
    m = torch.zeros((), dtype=torch.float64)
    v = torch.zeros((), dtype=torch.float64)
    for t, g in enumerate(cfg.gradients, start=1):
        m = cfg.beta1 * m + (1.0 - cfg.beta1) * g
        v = cfg.beta2 * v + (1.0 - cfg.beta2) * (g * g)
        update = cfg.lr * m / (torch.sqrt(v) + cfg.eps)
        param.data = param.data - update
        rows.append(
            {
                "step": t,
                "grad": g,
                "m": m.item(),
                "v": v.item(),
                "m_hat": m.item(),
                "v_hat": v.item(),
                "weight_after": param.item(),
            }
        )
    return rows


def compare_manual_pytorch(cfg: AdamHandConfig) -> dict[str, Any]:
    manual = manual_adam_steps(cfg, bias_correction=True)
    torch_rows = pytorch_adam_steps(cfg, bias_correction=True)
    comparisons: list[dict[str, float]] = []
    for m_row, t_row in zip(manual, torch_rows):
        comparisons.append(
            {
                "step": m_row.step,
                "abs_diff_m": abs(m_row.m - t_row["m"]),
                "abs_diff_v": abs(m_row.v - t_row["v"]),
                "abs_diff_m_hat": abs(m_row.m_hat - t_row["m_hat"]),
                "abs_diff_v_hat": abs(m_row.v_hat - t_row["v_hat"]),
                "abs_diff_weight": abs(m_row.weight_after - t_row["weight_after"]),
            }
        )
    max_weight_diff = max(c["abs_diff_weight"] for c in comparisons)
    return {
        "config": asdict(cfg),
        "manual_steps": [asdict(r) for r in manual],
        "pytorch_steps": torch_rows,
        "comparisons": comparisons,
        "max_abs_weight_diff": max_weight_diff,
    }


def bias_correction_trajectory(
    cfg: AdamHandConfig, num_steps: int = 20, grad_fn: Any = None
) -> dict[str, list[float]]:
    """Track weight under with/without bias correction for num_steps."""
    if grad_fn is None:
        base = list(cfg.gradients)
        grads = [base[i % len(base)] for i in range(num_steps)]
    else:
        grads = [grad_fn(i) for i in range(num_steps)]

    def run_with_updates(bias: bool) -> tuple[list[float], list[float]]:
        w = cfg.initial_weight
        m = 0.0
        v = 0.0
        weights = [w]
        updates: list[float] = []
        for t in range(1, num_steps + 1):
            g = grads[t - 1]
            m = cfg.beta1 * m + (1.0 - cfg.beta1) * g
            v = cfg.beta2 * v + (1.0 - cfg.beta2) * (g * g)
            if bias:
                m_use = m / (1.0 - cfg.beta1**t)
                v_use = v / (1.0 - cfg.beta2**t)
            else:
                m_use = m
                v_use = v
            upd = cfg.lr * m_use / (math.sqrt(v_use) + cfg.eps)
            updates.append(upd)
            w -= upd
            weights.append(w)
        return weights, updates

    with_bc, upd_bc = run_with_updates(True)
    without_bc, upd_no = run_with_updates(False)
    rel_weight_diffs = []
    rel_update_diffs = []
    for i in range(len(upd_bc)):
        denom_w = max(abs(with_bc[i + 1]), 1e-12)
        rel_weight_diffs.append(abs(with_bc[i + 1] - without_bc[i + 1]) / denom_w)
        denom_u = max(abs(upd_bc[i]), 1e-12)
        rel_update_diffs.append(abs(upd_bc[i] - upd_no[i]) / denom_u)

    # Criterion: bias-correction inflation on m is <1%:
    # m_hat/m = 1/(1-beta1^t)  =>  beta1^t < 0.01
    analytic_step = None
    for t in range(1, num_steps + 1):
        if cfg.beta1**t < 0.01:
            analytic_step = t
            break

    threshold = 0.01
    first_stable = None
    streak = 0
    for i, rd in enumerate(rel_update_diffs):
        if i == 0:
            continue  # skip degenerate first step
        if rd < threshold:
            streak += 1
            if streak >= 3 and first_stable is None:
                first_stable = i - 1
        else:
            streak = 0

    return {
        "gradients": grads,
        "with_bias_correction": with_bc,
        "without_bias_correction": without_bc,
        "relative_weight_diff": rel_weight_diffs,
        "relative_update_diff": rel_update_diffs,
        "criterion_empirical": f"rel_update_diff < {threshold} for 3 consecutive steps (skip step 1)",
        "first_step_empirical_criterion_met": first_stable,
        "criterion_analytic": "beta1^t < 0.01 (m bias inflation < ~1%)",
        "first_step_analytic_criterion_met": analytic_step,
    }
