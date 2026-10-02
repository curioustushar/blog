# Linear Model → Mixture-of-Experts (MoE)

> Train a baseline MLP on MNIST, convert its weights into a 4-expert MoE, and prove the MoE continues to reduce loss from the converted checkpoint.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/curioustushar/curioustushar.github.io/blob/master/linear-to-moe/notebooks/linear_to_moe.ipynb)

---

## Project Objective

Demonstrate empirically that:
1. A baseline dense MLP can successfully train and reduce its loss.
2. The trained model can be **surgically converted** into a MoE architecture using its existing weights.
3. The resulting MoE model **continues training** from the converted weights and reduces loss further.
4. A single training curve with a clearly marked conversion point shows all three phases.

## Hardware and Environment

| Item | Value |
|:-----|:------|
| Target hardware | Google Colab T4 (free tier) or any CPU |
| Estimated runtime | < 5 minutes on T4, ~10 minutes on CPU |
| Framework | PyTorch |
| Python | 3.10+ |

```bash
pip install torch torchvision matplotlib
```

## Repository Structure

```text
linear-to-moe/
├── README.md
└── notebooks/
    └── linear_to_moe.ipynb   ← single self-contained Colab notebook
```

## Dataset

**MNIST** — 60,000 training images of handwritten digits (0–9), 10,000 test images.

**Why MNIST?**
- Fast to train (full experiment < 5 min on free Colab)
- 10 distinct classes → natural opportunity for expert specialization
- Loss visibly decreases, making before/after conversion clear on one chart

## Baseline Model Architecture

A 3-layer MLP:

```
Input (784)
  ├─ fc1: Linear(784 → 256) + ReLU + Dropout(0.2)
  ├─ fc2: Linear(256 → 128) + ReLU + Dropout(0.2)
  └─ fc3: Linear(128 → 10)
```

| Layer | Parameters |
|:------|:----------:|
| fc1 | 200,960 |
| fc2 | 32,896 |
| fc3 | 1,290 |
| **Total** | **235,146** |

## MoE Architecture

`fc2` (256→128) is **replaced** by a MoE block. Layers `fc1` and `fc3` are unchanged.

```
Input (256)
  ├─ Router:   Linear(256 → 4) + Softmax → top-2 gates
  ├─ Expert 0: Linear(256 → 128)
  ├─ Expert 1: Linear(256 → 128)
  ├─ Expert 2: Linear(256 → 128)
  └─ Expert 3: Linear(256 → 128)
      └─ Weighted sum of top-2 expert outputs → (128,)
```

| Setting | Value |
|:--------|:------|
| Number of experts | 4 |
| Top-k routing | k = 2 |
| Router | Linear projection + Softmax |
| Load balancing | None (intentionally simple) |
| Total MoE params | 334,862 |

## Weight Conversion Procedure

The baseline `fc2` weights are **copied to every expert**:

```python
for expert in moe_layer.experts:
    expert.weight.data.copy_(baseline.fc2.weight.data)
    expert.bias.data.copy_(baseline.fc2.bias.data)
```

- `fc1` and `fc3` are transferred verbatim.
- The router is initialized with small random weights (breaks symmetry).
- At conversion time, all experts are identical → the MoE output equals the dense baseline output.
- From the first gradient step, the router differentiates routing → experts begin to specialize.

## Training Configuration

| Setting | Baseline | MoE (continued) |
|:--------|:--------:|:---------------:|
| Optimizer | Adam | Adam |
| Learning rate | 1e-3 | 1e-3 |
| Epochs | 10 | +5 |
| Batch size | 128 | 128 |
| Loss | CrossEntropyLoss | CrossEntropyLoss |

## Results

| Phase | Epochs | Final Train Loss | Final Test Loss | Test Acc |
|:------|:------:|:----------------:|:---------------:|:--------:|
| Baseline MLP | 10 | 0.04198 | 0.07322 | 98.00% |
| MoE (continued) | +5 | 0.03515 | **0.06618** | **98.21%** |

**Success criterion: MoE final test loss (0.06618) < Baseline final test loss (0.07322) ✅**

Expert routing distribution after conversion (test set):

| Expert | Activations | Share |
|:------:|:-----------:|:-----:|
| Expert 0 | 348 | 1.7% |
| Expert 1 | 5,782 | 28.9% |
| Expert 2 | 7,985 | 39.9% |
| Expert 3 | 5,885 | 29.4% |

Full epoch-by-epoch logs: [`results/logs/training_log.json`](results/logs/training_log.json)  
Loss curve: [`results/figures/loss_curve.png`](results/figures/loss_curve.png)

## Reproducibility

### Google Colab (recommended)
1. Click the Colab badge above.
2. Runtime → Change runtime type → GPU (T4).
3. Run All Cells.

### Local
```bash
git clone https://github.com/curioustushar/curioustushar.github.io/tree/master/linear-to-moe
cd linear-to-moe
pip install torch torchvision matplotlib
jupyter notebook notebooks/linear_to_moe.ipynb
```

## Why the MoE Continues to Learn

1. **Inherited knowledge** — each expert starts as a copy of the converged `fc2`, so useful digit representations are preserved.
2. **Specialization pressure** — the router routes different digit classes to different experts; each expert then receives gradients only from its assigned inputs, allowing focused improvement.
3. **Increased capacity** — 4× more parameters in the hidden layer (2 active at inference) gives the model room to represent finer distinctions.

## Limitations

- MNIST is a toy dataset; results may differ on more complex tasks where expert specialization takes longer.
- No auxiliary load-balancing loss: in a production MoE, one expert may receive too many tokens (expert collapse). We observe this on MNIST but it does not significantly harm accuracy at this scale.
- The router is linear; more complex routers (learned hash, etc.) may improve specialization.

## Conclusions

The experiment confirms that **weight-copy conversion from dense to MoE is a viable warm-start strategy**. The MoE does not regress to random initialization; it continues from where the dense model left off and improves further. This mirrors production techniques like MoE upcycling used in models such as Mixtral.
