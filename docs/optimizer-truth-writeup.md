# Optimizer Truth — Write-Up

**GitHub:** [`optimizer-truth/`](https://github.com/curioustushar/blog/tree/master/optimizer-truth)  
**Live post:** [Optimizer Truth](https://curioustushar.github.io/blog/posts/optimizer-truth/)

## Theory (short)

- Gradient → direction; \(\eta\), Adam moments, and schedules → distance.
- Adam: EMA \(m_t, v_t\) with bias-corrected \(\hat m_t, \hat v_t\); update \(\propto \hat m_t/(\sqrt{\hat v_t}+\epsilon)\).
- Bias correction matters while \(\beta_1^t\) is non-negligible (~44 steps for \(\beta_1=0.9\)).
- Warmup limits early ‖Δw‖/‖w‖ while moments are cold.
- Cosine needs horizon; WSD = warmup + plateau + decay.
- Width sweeps probe LR scaling; muP-style \(1/d\) is a hypothesis to test, not assume.

## Verified results (`run_experiments.py`)

| Item | Value |
|------|-------|
| Adam hand vs PyTorch | max weight diff **0** |
| Bias correction (analytic) | step **44** |
| Warmup ratio stable | step **50** |
| Loss @ 200 cosine / WSD | **0.02206** / **0.02082** |
| LR @ 200 cosine / WSD | **0.00107** / **0.00300** |
| Best LR 256 / 512 / 1024 | **1e-2** / **3e-3** / **3e-3** |
| η @ 4096 (fit) | **7.36e-4** |

## Run

```bash
cd optimizer-truth && pip install -r requirements.txt && python run_experiments.py
```
