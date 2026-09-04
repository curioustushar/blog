"""Generate notebook.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "notebook.ipynb"


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": s.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


cells = [
    md(
        """# Training Loop Truth

Take a small model and a real loop, and make it tell you the truth about itself.

**GitHub:** [training-loop-truth](https://github.com/curioustushar/blog/tree/master/training-loop-truth)
"""
    ),
    code(
        """import sys
from pathlib import Path
ROOT = Path.cwd()
if not (ROOT / 'src').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch

from src.config import TrainConfig
from src.float_repr import represent_0_1
from src.mfu import measure_mfu
from src.model import TinyCharLM
from src.training import (
    explain_shapes,
    finite_difference_check,
    find_grad_before_loss,
    grad_norm,
    lm_loss,
    plot_accumulation,
    plot_training_log,
    train_with_logging,
)

SEED = 42
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG = TrainConfig(device=str(DEVICE))
print('Device:', DEVICE)
print('Config:', CONFIG)
"""
    ),
    md("## 1. Configuration & model"),
    code(
        """model = TinyCharLM(CONFIG).to(DEVICE)
print('Parameters:', model.param_count())
print('Architecture: Embedding -> ReLU(Linear) -> Linear (char LM head)')
print('Optimizer: Adam, lr=', CONFIG.lr)
print('Loss: token-level cross-entropy (next-token)')
"""
    ),
    md("## 2. Tensor shapes (one training step)"),
    code(
        """tokens = torch.randint(0, CONFIG.vocab_size, (CONFIG.batch_size, CONFIG.seq_len), device=DEVICE)
model.zero_grad()
logits = model(tokens)
loss = lm_loss(logits, tokens)
for line in explain_shapes(tokens, logits, loss):
    print(line)
loss.backward()
print('fc2.bias grad shape:', list(model.fc2.bias.grad.shape), '# vocab_size')
"""
    ),
    md("## 3. Finite-difference gradient check"),
    code(
        """gc = finite_difference_check(model, tokens)
print('Parameter:', gc.param_name, gc.index)
print('Original weight:', gc.original_weight)
print('Epsilon:', gc.epsilon)
print('Loss(w+eps):', gc.loss_plus)
print('Loss(w-eps):', gc.loss_minus)
print('Numerical grad:', gc.numerical_grad)
print('Autograd grad:', gc.autograd_grad)
print('Abs diff:', gc.abs_diff)
print('Rel diff:', gc.rel_diff)
assert gc.rel_diff < 0.05
"""
    ),
    md(
        """## 4. Gradient accumulation — wrong vs correct

Micro-batch A: 10 tokens. Micro-batch B: 100 tokens.

**Wrong:** `(loss_A + loss_B) / 2` — equal weight per micro-batch.  
**Correct:** token-weighted global mean.
"""
    ),
    code(
        """micro_a = torch.randint(0, CONFIG.vocab_size, (2, 11), device=DEVICE)
micro_b = torch.randint(0, CONFIG.vocab_size, (2, 101), device=DEVICE)
na = micro_a.size(0) * (micro_a.size(1) - 1)
nb = micro_b.size(0) * (micro_b.size(1) - 1)

train_m = TinyCharLM(CONFIG).to(DEVICE)
opt = torch.optim.SGD(train_m.parameters(), lr=0.05)
wrong_curve, correct_curve = [], []
for _ in range(40):
    opt.zero_grad()
    la = lm_loss(train_m(micro_a), micro_a)
    lb = lm_loss(train_m(micro_b), micro_b)
    wrong_curve.append(((la + lb) / 2).item())
    correct_curve.append(((la * na + lb * nb) / (na + nb)).item())
    ((la + lb) / 2).backward()
    opt.step()

plot_accumulation(wrong_curve, correct_curve, str(ROOT / 'figures/gradient_accumulation.png'))
plt.show()
print('Single-pair wrong:', wrong_curve[0], 'correct:', correct_curve[0])
"""
    ),
    md("## 5. Gradient norm vs loss"),
    code(
        """log = train_with_logging(CONFIG, DEVICE)
plot_training_log(log, str(ROOT/'figures/loss_vs_step.png'), str(ROOT/'figures/grad_norm_vs_step.png'))
event = find_grad_before_loss(log)
print('Grad-norm-before-loss event:', event)
plt.figure(figsize=(8,4))
plt.plot(log.steps, log.losses, label='Loss')
plt.plot(log.steps, log.grad_norms, label='Grad norm')
plt.legend(); plt.xlabel('Step'); plt.title('Loss and grad norm')
plt.grid(True, alpha=0.3); plt.show()
"""
    ),
    md("## 6. MFU (honest estimate)"),
    code(
        """mfu = measure_mfu(CONFIG, DEVICE)
print('MFU %:', round(mfu.mfu_percent, 4))
print('Achieved FLOP/s:', mfu.achieved_flops_per_sec)
print('Peak assumed:', mfu.peak_flops_per_sec)
print('Tokens/sec:', round(mfu.tokens_per_second, 1))
for note in mfu.bottleneck_notes:
    print('-', note)
"""
    ),
    md(
        """## 7. Floating-point: 0.1 in FP32, BF16, FP8 E4M3

Manual bit patterns (then verified in code).
"""
    ),
    code(
        """for fb in represent_0_1():
    print(f"\\n{fb.format_name}")
    print(f"  sign={fb.sign} | exp={fb.exponent_bits} | frac={fb.fraction_bits}")
    print(f"  hex={fb.hex_pattern} stored={fb.stored_value} error={fb.error_vs_0_1}")
"""
    ),
    md(
        """## 8. Precision recommendation

**Train in BF16** (matmuls) with **FP32 optimizer master weights**.

- FP32: safe reference, 2× memory vs BF16  
- BF16: same exponent range as FP32, half memory, good HW support  
- FP8 E4M3: highest throughput on H100+, narrow mantissa — use for matmuls with scaling, not every op
"""
    ),
    md("## Final summary"),
    code(
        """print('=== Summary ===')
print('Grad check rel diff:', gc.rel_diff)
print('Accumulation gap (step 0):', abs(wrong_curve[0]-correct_curve[0]))
print('Grad event step:', event['step'])
print('MFU %:', mfu.mfu_percent)
print('Precision: BF16 + FP32 optimizer states')
"""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
        "colab": {"provenance": []},
    },
    "cells": cells,
}
OUT.write_text(json.dumps(nb, indent=1))
print(f"Wrote {OUT}")
