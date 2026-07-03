#!/usr/bin/env python3
"""Generate figures for Neural Network Foundations blog post."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent.parent / "static" / "images" / "neural-network"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "class0": "#2d6a6a",
    "class1": "#c45c3e",
    "grid": "#e2dfd8",
    "text": "#1a1a1a",
    "muted": "#5c5c5c",
    "accent": "#2d6a6a",
    "bg": "#faf9f7",
}


def style_axes(ax, title, subtitle=""):
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(True, alpha=0.35, color=PALETTE["grid"])
    ax.set_title(title, fontsize=13, fontweight="bold", color=PALETTE["text"], pad=10)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", fontsize=9,
                color=PALETTE["muted"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])


def make_rings(n=300, noise=0.08, seed=42):
    rng = np.random.default_rng(seed)
    angles = rng.random(n) * 2 * np.pi
    labels = rng.integers(0, 2, n)
    radii = np.where(labels == 0, 0.5, 1.0)
    x = radii * np.cos(angles) + rng.normal(0, noise, n)
    y = radii * np.sin(angles) + rng.normal(0, noise, n)
    return np.stack([x, y], axis=1), labels


def plot_boundary(ax, model, X, y, title, subtitle=""):
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 200),
        np.linspace(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3, 200),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    cmap = ListedColormap([(*plt.matplotlib.colors.to_rgb(PALETTE["class0"]), 0.15),
                           (*plt.matplotlib.colors.to_rgb(PALETTE["class1"]), 0.15)])
    ax.contourf(xx, yy, Z, alpha=0.55, cmap=cmap, levels=[0, 0.5, 1])
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c=PALETTE["class0"], s=22, edgecolors="white", linewidths=0.4, label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c=PALETTE["class1"], s=22, edgecolors="white", linewidths=0.4, label="Class 1")
    acc = accuracy_score(y, model.predict(X))
    style_axes(ax, f"{title}\nAccuracy: {acc:.0%}", subtitle)
    ax.set_xlabel("x₁", fontsize=10)
    ax.set_ylabel("x₂", fontsize=10)
    ax.set_aspect("equal")
    return acc


def fig_s1_1():
    X, y = make_rings()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linear = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    relu = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(16,), activation="relu", max_iter=2000, random_state=42),
    )
    linear.fit(X_train, y_train)
    relu.fit(X_train, y_train)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=PALETTE["bg"])
    plot_boundary(axes[0], linear, X_test, y_test, "Linear only", "No hidden activation")
    plot_boundary(axes[1], relu, X_test, y_test, "One ReLU layer", "Same data, one nonlinearity")
    fig.suptitle("S1-1 · Activations create boundaries", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "s1-1-activations.png", dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print("wrote s1-1-activations.png")


def fig_s1_2():
    X, y = make_rings()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    one_linear = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    deep_linear = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(16, 16, 16, 16), activation="identity", max_iter=3000, random_state=42),
    )
    deep_relu = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(16, 16, 16, 16), activation="relu", max_iter=3000, random_state=42),
    )
    for m in (one_linear, deep_linear, deep_relu):
        m.fit(X_train, y_train)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor=PALETTE["bg"])
    plot_boundary(axes[0], one_linear, X_test, y_test, "1 linear layer")
    plot_boundary(axes[1], deep_linear, X_test, y_test, "5 linear layers", "Collapses to one map")
    plot_boundary(axes[2], deep_relu, X_test, y_test, "5 layers + ReLU", "Depth matters now")
    fig.suptitle("S1-2 · Depth without nonlinearity is a lie", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "s1-2-depth.png", dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print("wrote s1-2-depth.png")


def fig_s1_3():
    rng = np.random.default_rng(7)
    categories = {
        "animals": ["cat", "dog", "cow"],
        "fruits": ["apple", "mango"],
        "verbs": ["eat", "chase", "see"],
    }
    colors = {"animals": PALETTE["class0"], "fruits": PALETTE["class1"], "verbs": "#6b5b95"}
    centers = {"animals": (-1.2, 0.8), "fruits": (1.3, 0.5), "verbs": (0.0, -1.4)}

    points, labels, tokens = [], [], []
    for cat, words in categories.items():
        cx, cy = centers[cat]
        for w in words:
            points.append([cx + rng.normal(0, 0.18), cy + rng.normal(0, 0.18)])
            labels.append(cat)
            tokens.append(w)
    points = np.array(points)

    fig, ax = plt.subplots(figsize=(7.5, 6), facecolor=PALETTE["bg"])
    for cat in categories:
        mask = np.array(labels) == cat
        ax.scatter(points[mask, 0], points[mask, 1], c=colors[cat], s=120, label=cat.title(),
                   edgecolors="white", linewidths=0.8, zorder=2)
        for i, tok in enumerate(tokens):
            if labels[i] == cat:
                ax.annotate(tok, (points[i, 0], points[i, 1]), textcoords="offset points",
                            xytext=(6, 6), fontsize=10, color=PALETTE["text"])
    style_axes(ax, "Learned embeddings (2D projection)", "Clustered by next-token co-occurrence only")
    ax.legend(frameon=True, facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"])
    ax.set_xlabel("embedding dim 1")
    ax.set_ylabel("embedding dim 2")
    fig.tight_layout()
    fig.savefig(OUT / "s1-3-embeddings.png", dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print("wrote s1-3-embeddings.png")


def fig_s1_3_trained():
    """Train a tiny next-token model and project real embeddings."""
    vocab = ["cat", "dog", "cow", "apple", "mango", "eat", "chase", "see"]
    token_to_id = {t: i for i, t in enumerate(vocab)}
    animals = ["cat", "dog", "cow"]
    verbs = ["eat", "chase", "see"]
    fruits = ["apple", "mango"]

    pairs = []
    rng = np.random.default_rng(0)
    for _ in range(8000):
        a, v, f = rng.choice(animals), rng.choice(verbs), rng.choice(fruits)
        sent = [a, v, f]
        for i in range(len(sent) - 1):
            pairs.append((token_to_id[sent[i]], token_to_id[sent[i + 1]]))

    n_vocab, dim = len(vocab), 8
    W = rng.normal(0, 0.05, (n_vocab, dim))
    lr = 0.08
    for ctx, nxt in pairs:
        logits = W[ctx] @ W.T
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        grad = probs.copy()
        grad[nxt] -= 1
        dW = np.outer(grad, W[ctx])
        W[ctx] -= lr * (grad @ W)
        W -= lr * dW
        W = np.clip(W, -3, 3)

    emb = PCA(n_components=2).fit_transform(W)
    cat_map = {t: "animals" for t in animals}
    cat_map.update({t: "fruits" for t in fruits})
    cat_map.update({t: "verbs" for t in verbs})
    colors = {"animals": PALETTE["class0"], "fruits": PALETTE["class1"], "verbs": "#6b5b95"}

    fig, ax = plt.subplots(figsize=(7.5, 6), facecolor=PALETTE["bg"])
    for i, tok in enumerate(vocab):
        c = colors[cat_map[tok]]
        ax.scatter(emb[i, 0], emb[i, 1], c=c, s=120, edgecolors="white", linewidths=0.8)
        ax.annotate(tok, (emb[i, 0], emb[i, 1]), textcoords="offset points", xytext=(6, 6),
                    fontsize=10, color=PALETTE["text"])
    style_axes(ax, "Trained embedding table (PCA)", "Tiny next-token model, no similarity labels")
    fig.tight_layout()
    fig.savefig(OUT / "s1-3-embeddings-trained.png", dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print("wrote s1-3-embeddings-trained.png")


def fig_s1_4():
    rng = np.random.default_rng(99)

    def make_data(n, d=10):
        X = rng.normal(0, 1, (n, d))
        y = (X[:, 0] + X[:, 1] - 0.5 * X[:, 2] + rng.normal(0, 0.3, n) > 0).astype(int)
        return X, y

    sizes = [20, 200, 2000]
    train_acc, test_acc = [], []
    for n in sizes:
        X, y = make_data(n + 400)
        X_train, y_train = X[:n], y[:n]
        X_test, y_test = X[n:], y[n:]
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 128, 64), activation="relu", max_iter=800, random_state=42),
        )
        clf.fit(X_train, y_train)
        train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
        test_acc.append(accuracy_score(y_test, clf.predict(X_test)))

    gaps = [t - e for t, e in zip(train_acc, test_acc)]

    fig = plt.figure(figsize=(10, 4.5), facecolor=PALETTE["bg"])
    gs = GridSpec(1, 2, width_ratios=[1.1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    x = np.arange(len(sizes))
    w = 0.35
    ax1.bar(x - w / 2, train_acc, w, label="Train", color=PALETTE["class0"], alpha=0.85)
    ax1.bar(x + w / 2, test_acc, w, label="Test", color=PALETTE["class1"], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.set_xlabel("Training set size")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    style_axes(ax1, "Train vs test accuracy")
    ax1.legend(facecolor=PALETTE["bg"])

    ax2.plot(sizes, gaps, "o-", color=PALETTE["accent"], linewidth=2.5, markersize=9)
    ax2.fill_between(sizes, gaps, alpha=0.12, color=PALETTE["accent"])
    style_axes(ax2, "Generalization gap", "Train acc − test acc")
    ax2.set_xlabel("Training set size")
    ax2.set_ylabel("Gap")
    ax2.set_xscale("log")

    fig.suptitle("S1-4 · Data closes the generalization gap", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "s1-4-generalization.png", dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print("wrote s1-4-generalization.png")


def fig_hero():
    """Cover-style composite for the post header."""
    X, y = make_rings(n=200, noise=0.06)
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=PALETTE["bg"])
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c=PALETTE["class0"], s=16, alpha=0.75)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c=PALETTE["class1"], s=16, alpha=0.75)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.5 * np.cos(theta), 0.5 * np.sin(theta), "--", color=PALETTE["muted"], lw=1.2, alpha=0.7)
    ax.plot(1.0 * np.cos(theta), 1.0 * np.sin(theta), "--", color=PALETTE["muted"], lw=1.2, alpha=0.7)
    ax.axhline(0.15, color=PALETTE["class1"], lw=2, linestyle="-", alpha=0.5, label="Linear boundary (fails)")
    style_axes(ax, "Neural Network Foundations", "Four experiments · one story")
    ax.legend(loc="upper right", frameon=True, facecolor=PALETTE["bg"])
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "hero.png", dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print("wrote hero.png")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    fig_hero()
    fig_s1_1()
    fig_s1_2()
    fig_s1_3_trained()
    fig_s1_4()
    print(f"Done → {OUT}")
