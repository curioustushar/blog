"""
Linear → MoE training script.
Trains a baseline MLP on MNIST, converts to MoE, continues training.
Saves: results/training_log.json, results/loss_curve.png
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, time, os, sys

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
os.makedirs("results/logs", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0)
print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

# ── Baseline MLP ──────────────────────────────────────────────────────────────
class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(784, 256)
        self.fc2  = nn.Linear(256, 128)
        self.fc3  = nn.Linear(128, 10)
        self.drop = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.drop(self.fc1(x)))
        x = self.relu(self.drop(self.fc2(x)))
        return self.fc3(x)

baseline = BaselineMLP().to(device)
print(f"Baseline params: {sum(p.numel() for p in baseline.parameters()):,}")

# ── MoE components ────────────────────────────────────────────────────────────
class MoELayer(nn.Module):
    def __init__(self, in_features, out_features, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.router      = nn.Linear(in_features, num_experts)
        self.experts     = nn.ModuleList([
            nn.Linear(in_features, out_features) for _ in range(num_experts)
        ])

    def forward(self, x):
        scores           = torch.softmax(self.router(x), dim=-1)
        topk_scores, topk_idx = scores.topk(self.top_k, dim=-1)
        topk_scores      = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        out_dim          = self.experts[0].out_features
        out              = torch.zeros(x.size(0), out_dim, device=x.device)
        for k in range(self.top_k):
            for e_i in range(self.num_experts):
                mask = (topk_idx[:, k] == e_i)
                if mask.any():
                    out[mask] += topk_scores[mask, k].unsqueeze(1) * self.experts[e_i](x[mask])
        return out

class MoEMLP(nn.Module):
    def __init__(self, num_experts=4, top_k=2):
        super().__init__()
        self.fc1  = nn.Linear(784, 256)
        self.moe  = MoELayer(256, 128, num_experts=num_experts, top_k=top_k)
        self.fc3  = nn.Linear(128, 10)
        self.drop = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.drop(self.fc1(x)))
        x = self.relu(self.drop(self.moe(x)))
        return self.fc3(x)

# ── Training helpers ──────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, n = 0, 0, 0
    t0 = time.time()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        n          += X.size(0)
    return total_loss / n, correct / n, time.time() - t0

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, n = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out  = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * X.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        n          += X.size(0)
    return total_loss / n, correct / n

# ── Phase 1: Baseline training ────────────────────────────────────────────────
BASELINE_EPOCHS = 10
optimizer_base  = optim.Adam(baseline.parameters(), lr=1e-3)
log             = {"baseline": [], "moe": [], "conversion_epoch": BASELINE_EPOCHS}

print("\n" + "="*65)
print("PHASE 1 — BASELINE TRAINING")
print("="*65)
print(f"{'Epoch':>5} {'Train Loss':>11} {'Train Acc':>10} {'Test Loss':>10} {'Test Acc':>9} {'Time(s)':>8}")
print("-"*65)

for epoch in range(1, BASELINE_EPOCHS + 1):
    tl, ta, secs = train_epoch(baseline, train_loader, optimizer_base)
    vl, va       = evaluate(baseline, test_loader)
    rec = {"epoch": epoch, "train_loss": round(tl,5), "train_acc": round(ta,5),
           "test_loss": round(vl,5), "test_acc": round(va,5), "time_s": round(secs,2)}
    log["baseline"].append(rec)
    print(f"{epoch:>5} {tl:>11.5f} {ta:>10.4f} {vl:>10.5f} {va:>9.4f} {secs:>8.1f}s")

print(f"\nBaseline final  →  test loss: {log['baseline'][-1]['test_loss']}  "
      f"test acc: {log['baseline'][-1]['test_acc']:.4f}")

# ── Phase 2: Convert to MoE ───────────────────────────────────────────────────
print("\n" + "="*65)
print("PHASE 2 — CONVERTING BASELINE → MoE")
print("="*65)

moe_model = MoEMLP(num_experts=4, top_k=2).to(device)

# Transfer weights
moe_model.fc1.weight.data.copy_(baseline.fc1.weight.data)
moe_model.fc1.bias.data.copy_(baseline.fc1.bias.data)
moe_model.fc3.weight.data.copy_(baseline.fc3.weight.data)
moe_model.fc3.bias.data.copy_(baseline.fc3.bias.data)
for expert in moe_model.moe.experts:
    expert.weight.data.copy_(baseline.fc2.weight.data)
    expert.bias.data.copy_(baseline.fc2.bias.data)

print(f"MoE params: {sum(p.numel() for p in moe_model.parameters()):,}")

# Verify conversion preserves loss
sample_X, sample_y = next(iter(test_loader))
sample_X, sample_y = sample_X[:64].to(device), sample_y[:64].to(device)
with torch.no_grad():
    base_loss_at_conversion = criterion(baseline(sample_X),  sample_y).item()
    moe_loss_at_conversion  = criterion(moe_model(sample_X), sample_y).item()

print(f"Baseline loss on sample batch (at conversion): {base_loss_at_conversion:.5f}")
print(f"MoE loss on sample batch (just after conversion): {moe_loss_at_conversion:.5f}")
print(f"Delta: {abs(moe_loss_at_conversion - base_loss_at_conversion):.5f}  "
      f"(should be small — confirms weights were preserved)")
log["conversion_check"] = {
    "baseline_loss": round(base_loss_at_conversion, 5),
    "moe_loss_at_conversion": round(moe_loss_at_conversion, 5)
}

# ── Phase 3: Continue training MoE ───────────────────────────────────────────
MOE_EPOCHS    = 5
optimizer_moe = optim.Adam(moe_model.parameters(), lr=1e-3)

print("\n" + "="*65)
print("PHASE 3 — MoE CONTINUED TRAINING")
print("="*65)
print(f"{'Epoch':>5} {'Train Loss':>11} {'Train Acc':>10} {'Test Loss':>10} {'Test Acc':>9} {'Time(s)':>8}")
print("-"*65)

for epoch in range(1, MOE_EPOCHS + 1):
    tl, ta, secs = train_epoch(moe_model, train_loader, optimizer_moe)
    vl, va       = evaluate(moe_model, test_loader)
    global_epoch = BASELINE_EPOCHS + epoch
    rec = {"epoch": global_epoch, "train_loss": round(tl,5), "train_acc": round(ta,5),
           "test_loss": round(vl,5), "test_acc": round(va,5), "time_s": round(secs,2)}
    log["moe"].append(rec)
    print(f"{global_epoch:>5} {tl:>11.5f} {ta:>10.4f} {vl:>10.5f} {va:>9.4f} {secs:>8.1f}s")

print(f"\nMoE final  →  test loss: {log['moe'][-1]['test_loss']}  "
      f"test acc: {log['moe'][-1]['test_acc']:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
baseline_final_loss = log["baseline"][-1]["test_loss"]
moe_final_loss      = log["moe"][-1]["test_loss"]
success             = moe_final_loss < baseline_final_loss

print("\n" + "="*65)
print("FINAL COMPARISON")
print("="*65)
print(f"  {'Model':<22} {'Params':>9}  {'Final Test Loss':>15}  {'Test Acc':>9}")
print(f"  {'-'*60}")
print(f"  {'Baseline MLP':<22} {235146:>9,}  {baseline_final_loss:>15.5f}  {log['baseline'][-1]['test_acc']:>9.4f}")
print(f"  {'MoE (continued)':<22} {sum(p.numel() for p in moe_model.parameters()):>9,}  {moe_final_loss:>15.5f}  {log['moe'][-1]['test_acc']:>9.4f}")
print(f"\n  Success criterion (MoE loss < Baseline loss): {'✅ PASS' if success else '❌ FAIL'}")

log["summary"] = {
    "baseline_params": 235146,
    "moe_params": sum(p.numel() for p in moe_model.parameters()),
    "baseline_final_test_loss": baseline_final_loss,
    "moe_final_test_loss": moe_final_loss,
    "success": success,
}

# ── Expert routing distribution ───────────────────────────────────────────────
print("\n  Expert routing distribution (test set):")
moe_model.eval()
expert_counts = torch.zeros(4)
with torch.no_grad():
    for X, _ in test_loader:
        X = X.to(device).view(X.size(0), -1)
        h = torch.relu(moe_model.fc1(X))
        scores   = torch.softmax(moe_model.moe.router(h), dim=-1)
        topk_idx = scores.topk(2, dim=-1).indices
        for i in range(4):
            expert_counts[i] += (topk_idx == i).sum().item()
routing = {}
for i, c in enumerate(expert_counts):
    pct = 100 * c.item() / expert_counts.sum().item()
    routing[f"expert_{i}"] = {"activations": int(c.item()), "pct": round(pct, 1)}
    print(f"    Expert {i}: {int(c.item()):6d} activations ({pct:.1f}%)")
log["expert_routing"] = routing

# ── Save log ──────────────────────────────────────────────────────────────────
with open("results/logs/training_log.json", "w") as f:
    json.dump(log, f, indent=2)
print("\n  Training log → results/logs/training_log.json")

# ── Loss curve ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

x_base    = [r["epoch"] for r in log["baseline"]]
tl_base   = [r["train_loss"] for r in log["baseline"]]
vl_base   = [r["test_loss"]  for r in log["baseline"]]
x_moe     = [r["epoch"] for r in log["moe"]]
tl_moe    = [r["train_loss"] for r in log["moe"]]
vl_moe    = [r["test_loss"]  for r in log["moe"]]

ax.plot(x_base, tl_base, 'b-o',  markersize=5, label='Baseline train loss')
ax.plot(x_base, vl_base, 'b--s', markersize=5, label='Baseline test loss')
ax.plot(x_moe,  tl_moe,  'r-o',  markersize=5, label='MoE train loss')
ax.plot(x_moe,  vl_moe,  'r--s', markersize=5, label='MoE test loss')

ax.axvline(x=BASELINE_EPOCHS + 0.5, color='green', linestyle=':', linewidth=2.5)
ax.text(BASELINE_EPOCHS + 0.6, max(tl_base) * 0.97, '↑ Converted\n  to MoE',
        color='green', fontsize=9, va='top')

ax.set_xlabel('Epoch')
ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('Baseline MLP → MoE: Training Loss Before and After Conversion')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/loss_curve.png", dpi=150, bbox_inches='tight')
print("  Loss curve   → results/figures/loss_curve.png")
print("\nDone.")
