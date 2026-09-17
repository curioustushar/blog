# ZeRO Memory Truth — Write-Up

**GitHub:** [`zero-memory-sim/`](https://github.com/curioustushar/blog/tree/master/zero-memory-sim)  
**Live post:** [ZeRO Memory Truth](https://curioustushar.github.io/blog/posts/zero-memory-sim/)

## Summary

Educational 32-rank simulator for **DeepSpeed ZeRO stages 1, 2 and 3** on a
single CPU. Uses real per-rank tensor shards (bytes are exact) and analytical
ring-collective communication volumes.

## Verified numbers

For `P = 4,198,912` params, fp32, Adam, `N = 32`:

| Strategy | Params/rank | Grads/rank | Optim/rank | **Total/rank** |
|----------|------------:|-----------:|-----------:|---------------:|
| DP | 16.02 MiB | 16.02 MiB | 32.03 MiB | **64.07 MiB** |
| ZeRO-1 | 16.02 MiB | 16.02 MiB | 1.00 MiB | **33.04 MiB** |
| ZeRO-2 | 16.02 MiB | 0.50 MiB | 1.00 MiB | **17.52 MiB** |
| ZeRO-3 | 0.50 MiB | 0.50 MiB | 1.00 MiB | **2.00 MiB** |

Communication (bytes/rank/step):

- DP / ZeRO-1: ~2P (all-reduce grads)
- ZeRO-2: ~P (reduce-scatter grads)
- ZeRO-3: ~3P (all-gather params + reduce-scatter grads)

## Sanity checks (all pass)

- ZeRO-1 optimizer bytes/rank × N ≈ 2P·b
- ZeRO-2 gradient bytes/rank × N ≈ P·b
- ZeRO-3 parameter bytes/rank × N ≈ P·b
- ZeRO-1/2 parameters still replicated

## Run

```bash
cd zero-memory-sim
pip install -r requirements.txt
python run_experiments.py
jupyter notebook notebook.ipynb
```
