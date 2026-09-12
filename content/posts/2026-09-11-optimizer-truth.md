---
title: "Optimizer Truth"
slug: "optimizer-truth"
date: 2026-09-11T20:10:00-05:00
categories: ["machine-learning"]
tags:
  ["adam", "learning-rate", "wsd", "cosine", "warmup", "optimizer"]
author: Tushar Gupta
description: "Hand-verified Adam, bias correction, warmup update ratios, cosine vs WSD at step 200, and width-scaled LR sweeps — with the theory behind each measurement."
---

<div class="post-summary">

`loss.backward()` gives a gradient for every weight; `optimizer.step()` decides how far to move. That distance is not in the gradient — it is chosen by learning rate, momentum estimates, schedules, and warmup.

This post walks through a reproducible **optimizer-truth** lab: derive Adam on paper, match PyTorch on a scalar, measure when bias correction and warmup stop mattering, compare cosine vs WSD at step 200, and sweep learning rate across model widths.

<div class="plan-status" role="status" aria-label="Project status">
  <span class="status-badge status-ok">Hand Adam = PyTorch</span>
  <span class="status-badge status-ok">Cosine vs WSD @ 200</span>
  <span class="status-badge status-ok">LR sweep + 4096 extrapolation</span>
</div>

</div>

---

## Links

| | URL |
|---|---|
| **GitHub** | [optimizer-truth/](https://github.com/curioustushar/blog/tree/master/optimizer-truth) |
| **Full report** | [README.md](https://github.com/curioustushar/blog/blob/master/optimizer-truth/README.md) |
| **Write-up** | [docs/optimizer-truth-writeup.md](https://github.com/curioustushar/blog/blob/master/docs/optimizer-truth-writeup.md) |

---

## Theory: gradient → distance

Plain gradient descent uses one global scale \(\eta\) for every parameter:

\[
w_{t+1} = w_t - \eta \, g_t
\]

The magnitude of \(g_t\) does not tell you how far to move — only direction and relative urgency. **Adam** combines two ideas:

1. **First moment** \(m_t\): exponential moving average of gradients (momentum-like smoothing).
2. **Second moment** \(v_t\): per-parameter scale from recent squared gradients (RMSprop-like).

With \(\beta_1=0.9\), \(\beta_2=0.999\), \(\epsilon=10^{-8}\):

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\]

\[
\hat m_t = \frac{m_t}{1-\beta_1^t},\qquad
\hat v_t = \frac{v_t}{1-\beta_2^t},\qquad
w_{t+1} = w_t - \eta \,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

**Bias correction** (\(\hat m_t\), \(\hat v_t\)) fixes the fact that \(m_0=v_0=0\): early EMAs are biased toward zero, so uncorrected Adam takes tiny steps at first; dividing by \(1-\beta^t\) inflates early estimates toward their steady-state scale.

**Warmup** ramps \(\eta\) from 0 to its target over the first \(W\) steps. Early batches are correlated and moment estimates are cold; a full learning rate would produce oversized **update-to-weight** ratios. Warmup caps how aggressively weights move while statistics warm up.

**Schedules** change \(\eta\) over training:

- **Cosine:** after warmup, \(\eta\) follows a cosine decay to a floor — needs a known total step count up front.
- **WSD (warmup–stable–decay):** hold peak \(\eta\) for a plateau, then decay — useful when you may stop, branch, or extend the run without fixing horizon at step 0.

**Width and learning rate:** wider layers change activation scale and gradient variance. In the **muP** picture, a learning rate that works at width \(d\) often scales roughly as \(1/d\) (or related power laws) when widths change but architecture family stays fixed — sweeps at small widths are probes, not guarantees at 4096.

---

## Experimental setup

| Setting | Value |
|---------|-------|
| Task | Synthetic MSE: `y = x @ W_teacher` |
| Model | 2-layer **WidthMLP** (`Linear → ReLU → Linear`) |
| Optimizer | AdamW |
| Seed | 42 |
| Base LR | 3×10⁻³ |
| Warmup | 50 steps (linear) |
| Schedule run | 300 steps, report @ **step 200** |
| Width (schedule) | 256 |
| LR sweep | 80 steps per candidate; widths **256 / 512 / 1024** |

All numbers below come from `python run_experiments.py` (PyTorch 2.2.2, CPU reference run).

---

## Results at a glance

| Experiment | Result |
|------------|--------|
| Manual Adam (5 scalar steps) | Matches PyTorch — **max weight error 0** |
| Bias correction (analytic) | Negligible when \(\beta_1^t < 0.01\) → **step 44** |
| Bias correction (empirical path) | Weight trajectories still differ — **no** 3-step merge @ 200 steps |
| ‖Δw‖/‖w‖ vs warmup | Rolling ratio stabilizes @ **step 50** (warmup end) |
| Loss @ step 200 | Cosine **0.02206** vs WSD **0.02082** |
| LR @ step 200 | Cosine **0.00107** vs WSD **0.00300** (peak) |
| Best LR (sweep) | 256: **1e-2**, 512: **3e-3**, 1024: **3e-3** |
| LR @ width 4096 (log-log fit) | **7.36×10⁻⁴** (extrapolation) |

```json
{
  "adam_matches_pytorch": true,
  "bias_correction_steps_until_negligible_analytic": 44,
  "warmup_ratio_step": 50,
  "keep_model_at_200": "wsd",
  "lr_4096": 0.0007363699407000241
}
```

---

## 1. Hand Adam vs PyTorch

**Question:** Do the recurrences for \(m\), \(v\), \(\hat m\), \(\hat v\) and the weight update match the framework?

**Setup:** one scalar weight \(w_0=1.0\), five gradients `[0.5, -0.3, 0.1, 0.4, -0.2]`, `lr=0.01`, standard \(\beta_1,\beta_2\).

| Step | Manual \(w\) | PyTorch \(w\) |
|------|--------------|---------------|
| 1 | 0.9900000002 | 0.9900000002 |
| 5 | 0.9779843898 | 0.9779843898 |

At step 1, bias correction matters numerically: \(\hat m_1 = m_1/(1-\beta_1) = 0.05/0.1 = 0.5\), so the first update is a full `0.01 × 0.5 / 0.5 = 0.01` step — without correction you would use \(m_1=0.05\) and move ten times less.

**Interpretation:** Implementation bugs hide in the denominators \(1-\beta^t\). Matching PyTorch on a scalar removes that doubt before you trust a large training run.

---

## 2. Bias correction over 20 steps

**Question:** When can you ignore \(\hat m\) vs \(m\) in practice?

Same five gradients **cycled** for 20 steps; two trajectories — with and without correction.

![Adam with vs without bias correction](../../optimizer-truth/bias_correction.png)

**Theory:** The inflation factor on the first moment is \(1/(1-\beta_1^t)\). For \(\beta_1=0.9\), \(\beta_1^{44} \approx 0.01\), so correction on \(m\) is under ~1% from **step 44** onward.

**Experiment:** Early steps diverge sharply (step 2–5 relative weight gap grows). That is expected: step 1 without correction is effectively using a learning rate **ten times smaller** on the momentum channel.

**Empirical caveat:** Even after correction *factors* saturate, **weights** with vs without correction need not reunite — different paths, same local rules. Do not confuse “correction negligible” with “runs identical.”

---

## 3. Update-to-weight ratio and warmup

**Metric (per layer, per step):**

\[
\text{ratio} = \frac{\|\Delta w\|_2}{\|w\|_2}
\]

**Theory:** Warmup scales effective \(\eta_t = \eta \cdot \min(1, (t+1)/W)\). While \(\eta_t\) is rising, the optimizer is allowed to take larger parameter-relative steps even if loss has not caught up — so the ratio trace is a direct read on “how hard are we pushing weights” vs “how big are they already.”

![Update-to-weight ratio by layer](../../optimizer-truth/update_to_weight_ratio.png)

**Result:** Criterion met at **step 50** — exactly when linear warmup reaches full `base_lr`. `fc1.weight` usually shows the largest ratio; biases move less in relative terms.

---

## 4. Cosine vs WSD @ step 200

**Question:** Same initialization and optimizer; only the LR schedule differs. Which checkpoint do you keep at step 200?

![Learning-rate schedules](../../optimizer-truth/schedules_lr.png)

![Training loss: cosine vs WSD](../../optimizer-truth/cosine_vs_wsd_loss.png)

| @ step 200 | Cosine | WSD |
|------------|--------|-----|
| Train loss (MSE) | 0.02206 | **0.02082** |
| Learning rate | 0.00107 | **0.00300** |

**Theory:** Cosine has been decaying since warmup ended (~step 50). WSD in this protocol holds **peak** LR until `decay_start=200`, then linear decay — so at step 200, WSD still uses **3×10⁻³** while cosine is near **1.1×10⁻³**. Lower loss at 200 is therefore **not** a pure “WSD is better” statement; it is **WSD at high LR vs cosine already annealed**.

**Choice:** Keep the **WSD** weights at 200 if that checkpoint is the product goal *and* you accept the schedule asymmetry. For a fair schedule shootout, grid-search `decay_start`, cosine floor, and warmup jointly with equal tuning budget.

---

## 5. Learning-rate sweep and width 4096

**Theory:** Wider hidden layers change gradient noise and curvature. A sensible protocol fixes steps, batch, and seed, then scans \(\eta\) on a log grid per width.

![Loss vs learning rate by width](../../optimizer-truth/lr_sweep.png)

| Width | Best LR | Final loss (80 steps) |
|-------|---------|------------------------|
| 256 | **1×10⁻²** | 0.0327 |
| 512 | **3×10⁻³** | 0.0236 |
| 1024 | **3×10⁻³** | 0.0153 |

**Observation:** Optimum \(\eta\) **drops** from 256 → 512, then plateaus 512–1024 on this toy task — consistent with “wider needs smaller LR” but not a clean \(1/d\) law from three points.

**Extrapolation to 4096:** Log-log fit of best LR vs width gives slope ≈ **−0.87** and

\[
\eta_{4096} \approx 7.4 \times 10^{-4}
\]

**Confidence:** low–medium. Three widths, short runs, synthetic regression — treat as a **hypothesis** for the next sweep, not a validated production LR.

---

## Fair comparisons (principle)

> Tune both sides before accepting a comparison. Almost every optimizer claim that failed to replicate was a well-tuned method measured against a badly tuned one.

Applied here:

- LR sweeps use the **same** step count and data protocol per width.
- Cosine vs WSD shares seed, model, and AdamW — but **schedule hyperparameters are not co-tuned**; eval at 200 aligns with WSD’s decay knee.

Before publishing “cosine beats WSD” (or the reverse), spend comparable compute tuning both.

---

## Reproduce

```bash
git clone https://github.com/curioustushar/blog.git
cd blog/optimizer-truth
pip install -r requirements.txt
python run_experiments.py
```

Regenerates `results.json`, `figures/*.png`, and `outputs/update_ratios.json`.

---

## Takeaways

1. **Verify Adam on a scalar** before trusting multi-million-parameter state.
2. **Bias correction** is a early-training effect (~44 steps for \(\beta_1=0.9\)); trajectories can stay split forever.
3. **Warmup** shows up directly in ‖Δw‖/‖w‖ — not only in loss.
4. **Schedule comparisons** need matched tuning and honest checkpoint semantics (LR at eval matters).
5. **Width sweeps** suggest smaller LR as width grows; extrapolation to 4096 is engineering judgment, not measurement.
