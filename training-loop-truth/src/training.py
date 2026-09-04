"""Training loop helpers and experiments."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .config import TrainConfig
from .model import TinyCharLM


def explain_shapes(tokens: torch.Tensor, logits: torch.Tensor, loss: torch.Tensor) -> list[str]:
    b, t = tokens.shape
    v = logits.size(-1)
    lines = [
        f"tokens: {list(tokens.shape)}  # batch={b}, sequence_length={t}",
        f"logits: {list(logits.shape)}  # batch={b}, sequence_length={t}, vocab_size={v}",
        f"loss: {list(loss.shape)}  # scalar (0-d tensor)",
    ]
    return lines


def lm_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))


def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


@dataclass
class GradCheckResult:
    param_name: str
    index: tuple[int, ...]
    original_weight: float
    epsilon: float
    loss_plus: float
    loss_minus: float
    numerical_grad: float
    autograd_grad: float
    abs_diff: float
    rel_diff: float


def finite_difference_check(model: TinyCharLM, tokens: torch.Tensor, eps: float = 1e-4) -> GradCheckResult:
    model.train()
    param = model.fc2.bias
    i = 0
    w0 = param[i].item()

    def loss_at(val: float) -> float:
        with torch.no_grad():
            param.data[i] = val
            logits = model(tokens)
            return float(lm_loss(logits, tokens).double())

    loss_plus = loss_at(w0 + eps)
    loss_minus = loss_at(w0 - eps)
    param.data[i] = w0

    model.zero_grad()
    logits = model(tokens)
    loss = lm_loss(logits, tokens)
    loss.backward()
    autograd = param.grad[i].item()

    numerical = (loss_plus - loss_minus) / (2 * eps)
    abs_diff = abs(numerical - autograd)
    rel_diff = abs_diff / max(abs(autograd), 1e-12)

    return GradCheckResult(
        param_name="fc2.bias",
        index=(i,),
        original_weight=w0,
        epsilon=eps,
        loss_plus=loss_plus,
        loss_minus=loss_minus,
        numerical_grad=numerical,
        autograd_grad=autograd,
        abs_diff=abs_diff,
        rel_diff=rel_diff,
    )


def accumulation_experiment(
    model: TinyCharLM,
    micro_a: torch.Tensor,
    micro_b: torch.Tensor,
    steps: int = 50,
) -> tuple[list[float], list[float]]:
    """Return wrong (avg of avgs) and correct (token-weighted) loss curves."""
    wrong_curve, correct_curve = [], []
    for step in range(steps):
        torch.manual_seed(step)
        # fresh noise on logits path via dropout-free re-init small perturbation
        logits_a = model(micro_a)
        logits_b = model(micro_b)
        la = lm_loss(logits_a, micro_a)
        lb = lm_loss(logits_b, micro_b)
        na = (micro_a.size(0) * (micro_a.size(1) - 1))
        nb = (micro_b.size(0) * (micro_b.size(1) - 1))
        wrong = (la + lb) / 2
        correct = (la * na + lb * nb) / (na + nb)
        wrong_curve.append(wrong.item())
        correct_curve.append(correct.item())
    return wrong_curve, correct_curve


def plot_accumulation(wrong: list[float], correct: list[float], path: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(wrong, label="Incorrect: average of micro-batch means")
    plt.plot(correct, label="Correct: token-weighted global mean")
    plt.xlabel("Micro-batch pair index")
    plt.ylabel("Loss")
    plt.title("Gradient accumulation: wrong vs correct averaging")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


@dataclass
class TrainLog:
    steps: list[int]
    losses: list[float]
    grad_norms: list[float]


def train_with_logging(config: TrainConfig, device: torch.device) -> TrainLog:
    torch.manual_seed(config.seed)
    model = TinyCharLM(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    log = TrainLog(steps=[], losses=[], grad_norms=[])
    tokens = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len), device=device)

    for step in range(config.train_steps):
        opt.zero_grad()
        logits = model(tokens)
        loss = lm_loss(logits, tokens)
        loss.backward()
        gn = grad_norm(model)
        opt.step()
        log.steps.append(step)
        log.losses.append(loss.item())
        log.grad_norms.append(gn)
        # mild input drift so gradients react before loss flatlines
        tokens = (tokens + torch.randint(0, 3, tokens.shape, device=device)) % config.vocab_size

    return log


def find_grad_before_loss(log: TrainLog) -> dict:
    """Find step where grad norm changes sharply while loss is still relatively flat."""
    best_step = 1
    best_ratio = 0.0
    for i in range(2, len(log.steps) - 1):
        dg = abs(log.grad_norms[i] - log.grad_norms[i - 1])
        dl = abs(log.losses[i] - log.losses[i - 1])
        if dg > 1e-6 and dl < 0.05:
            ratio = dg / max(dl, 1e-8)
            if ratio > best_ratio:
                best_ratio = ratio
                best_step = i
    return {
        "step": best_step,
        "grad_norm": log.grad_norms[best_step],
        "grad_norm_prev": log.grad_norms[best_step - 1],
        "loss": log.losses[best_step],
        "loss_prev": log.losses[best_step - 1],
    }


def plot_training_log(log: TrainLog, path_loss: str, path_grad: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(log.steps, log.losses, label="Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss vs step")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_loss, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(log.steps, log.grad_norms, label="Grad norm", color="orange")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient L2 norm")
    ax.set_title("Gradient norm vs step")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_grad, dpi=150)
    plt.close(fig)
