#!/usr/bin/env python3
"""Print manual vs PyTorch Adam verification table."""

from src.adam_manual import compare_manual_pytorch
from src.config import AdamHandConfig

if __name__ == "__main__":
    r = compare_manual_pytorch(AdamHandConfig())
    for m, t, c in zip(r["manual_steps"], r["pytorch_steps"], r["comparisons"]):
        print(
            f"step {m['step']}: w_manual={m['weight_after']:.10f} w_torch={t['weight_after']:.10f} "
            f"diff={c['abs_diff_weight']:.2e}"
        )
    print("max weight diff:", r["max_abs_weight_diff"])
