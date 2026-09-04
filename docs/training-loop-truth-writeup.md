# Training Loop Truth — Write-Up

**Author:** Tushar Gupta  
**GitHub:** [`training-loop-truth/`](https://github.com/curioustushar/blog/tree/master/training-loop-truth)  
**Live post:** [Training Loop Truth](https://curioustushar.github.io/blog/posts/training-loop-truth/)

---

## Summary

Small char LM (8,320 params), real Adam training loop, six transparency experiments.

---

## Results

### 1. Shapes

- `tokens [8,16]` — batch × sequence  
- `logits [8,16,64]` — batch × sequence × vocab  
- `loss` — scalar  

### 2. Gradient check

| | Value |
|---|-------|
| Parameter | `fc2.bias[0]` |
| Numerical | -0.01431 |
| Autograd | -0.01498 |
| Rel diff | 4.5% |

### 3. Accumulation

| | Loss |
|---|------|
| Wrong (avg of avgs) | 4.196 |
| Correct (token-weighted) | 4.210 |

### 4. Grad norm event

Step 91: grad norm fell before loss moved materially.

### 5. MFU

**0.03%** on CPU reference run. Below 40% because: tiny model, tiny batch, Python overhead, no GPU saturation.

### 6. Float 0.1

| Format | Hex | Value |
|--------|-----|-------|
| FP32 | 0x3DCCCCCD | 0.1000000015 |
| BF16 | 0x3DCC | 0.099609375 |
| FP8 E4M3 | 0x2A | 0.1015625 |

### 7. Precision

**BF16** training + **FP32** optimizer master weights.

---

## Reproduce

```bash
cd training-loop-truth
pip install -r requirements.txt
python run_experiments.py
jupyter notebook notebook.ipynb
```
