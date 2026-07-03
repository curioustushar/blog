---
title: Neural Network Foundations
date: 2026-07-03T11:00:00-05:00
categories: ["machine-learning"]
tags: ["neural-networks", "activations", "embeddings", "generalization"]
author: Tushar Gupta
interactive: true
description: "Four hands-on experiments on why activations matter, depth is useless without them, embeddings emerge from next-token prediction, and data closes the generalization gap."
---

<div class="post-summary">

Four small experiments that make the mechanics undeniable: **nonlinearities create boundaries**, **depth without them is a no-op**, **similarity emerges from next-token pressure alone**, and **data — not architecture — closes the generalization gap**. Each section is one claim, one build, one proof — press **Train** and watch it happen in the browser.

</div>

---

## <span class="section-pill">S1-1</span> Activations exist for a reason

### Claim

A model with no nonlinearity can only draw a **straight boundary**. It cannot separate two interleaved or concentric rings. Add one ReLU hidden layer and it can.

### Build

Generate ~300 noisy 2D points as two rings — inner ring = class 0, outer ring = class 1. The setup is deliberately not linearly separable.

Train two models on the same data:

- **(a)** A single linear layer + sigmoid (logistic regression in disguise)
- **(b)** One hidden layer with ReLU + linear output + sigmoid

### Proof

Plot both decision boundaries on the same scatter. The linear model draws a **single straight line** — accuracy stalls near **~55%**. The ReLU model bends the boundary into a **ring-shaped curve** and reaches **~99%**.

<figure class="figure-card interactive">
  <div class="nn-demo" id="s1-activations"></div>
  <figcaption><strong>The money shot.</strong> Only the activation changed — same data, same optimizer. Without nonlinearity you fit a hyperplane; with one ReLU layer you fit a shape.</figcaption>
</figure>

---

## <span class="section-pill">S1-2</span> Depth without nonlinearity is a lie

### Claim

Five stacked linear layers collapse to a **single linear map**. A 5-layer linear network is no stronger than a 1-layer network — both fail the ring task identically. Insert ReLUs between the same five layers and the task suddenly becomes solvable.

### Build

Same ring data. Train three variants:

1. **1 linear layer** (no hidden activations)
2. **5 linear layers** stacked with no activations between them
3. **5 layers with ReLU** between each

### Proof

The 1-layer and 5-linear-layer models produce **identical accuracies** and **identical straight-line boundaries**. Depth bought nothing. ReLU breaks the tie.

<figure class="figure-card interactive">
  <div class="nn-demo" id="s2-depth"></div>
  <figcaption><strong>Depth is only depth with nonlinearity.</strong> Left and center: the same straight boundary. Right: a curved boundary that wraps the ring.</figcaption>
</figure>

---

## <span class="section-pill">S1-3</span> Embeddings learn similarity from next-token prediction

### Claim

Trained only to predict the **next token** in a tiny synthetic grammar, the embedding table clusters related tokens — even though no similarity label was ever supplied.

### Build

A toy language with categories:

| Category | Tokens |
|----------|--------|
| Animals  | cat, dog, cow |
| Fruits   | apple, mango |
| Verbs    | eat, chase, see |

Templates like `<animal> <verb> <fruit>` mean same-category tokens share next-token distributions.

### Proof

Project the learned embeddings to 2D. Tokens from the same category **land together**. Nearest neighbors are almost always same-category.

<figure class="figure-card interactive">
  <div class="nn-demo" id="s3-embeddings"></div>
  <figcaption><strong>Emergent clustering.</strong> No similarity loss, no contrastive pairs — just next-token pressure, and the geometry organizes itself.</figcaption>
</figure>

---

## <span class="section-pill">S1-4</span> Memorization vs generalization

### Claim

A high-capacity model on tiny data drives **train loss to ~0** while **held-out loss stays high**. Growing the dataset closes the gap.

### Build

A learnable noisy classification task with a held-out test split. Train an over-parameterized network at training set sizes **20**, **200**, and **2000**.

### Proof

| Train size | Train acc | Test acc | Gap |
|------------|-----------|----------|-----|
| 20         | ~100%     | poor     | **huge** |
| 200        | ~100%     | better   | **shrinking** |
| 2000       | ~100%     | strong   | **small** |

<figure class="figure-card interactive">
  <div class="nn-demo" id="s4-generalization"></div>
  <figcaption><strong>Data is everything.</strong> Architecture gives you capacity; data gives you generalization. The gap collapses as training set size grows.</figcaption>
</figure>

At 20 samples the model **memorizes**. At 2000 it must learn a rule.

---

## Takeaways

1. **Activations** turn linear algebra into geometry — one ReLU is the difference between a line and a ring.
2. **Depth** without nonlinearity is matrix multiplication in a trench coat.
3. **Embeddings** organize meaning from prediction alone — similarity is a side effect of next-token pressure.
4. **Generalization** is a data problem dressed as a model problem — more data closes the gap that capacity opens.
