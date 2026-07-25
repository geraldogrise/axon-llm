"""Multiclass classification with pyaxon: 3 point clouds (blobs) in 2D.

Demonstrates Linear + GELU + cross_entropy + Adam learning to separate 3 classes,
and measures the final accuracy. Uses the Python API (core in C++)."""

import math
import random

import pyaxon as ax

random.seed(0)

# --- Generate 3 Gaussian blobs, one per class, around distinct centers ---
CENTERS = [(-2.0, -2.0), (2.0, -2.0), (0.0, 2.0)]
N_PER_CLASS = 60

xs, ys = [], []
for cls, (cx, cy) in enumerate(CENTERS):
    for _ in range(N_PER_CLASS):
        xs.append([cx + random.gauss(0, 0.6), cy + random.gauss(0, 0.6)])
        ys.append(cls)

x = ax.tensor(xs)             # (180, 2)
y = ax.tensor([float(c) for c in ys])  # (180,) class indices

# --- Model: 2 -> 16 -> GELU -> 3 (logits) ---
model = ax.nn.Sequential([
    ax.nn.Linear(2, 16, seed=3),
    ax.nn.GELU(),
    ax.nn.Linear(16, 3, seed=4),
])
opt = ax.optim.Adam(model.parameters(), lr=0.05)

print("Training classifier (3 classes)...")
for epoch in range(1, 301):
    logits = model(x)
    loss = ax.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 50 == 0 or epoch == 1:
        print(f"epoch {epoch:4d} | loss = {loss.item():.4f}")

# --- Final accuracy: argmax of logits vs. true class ---
logits = model(x).tolist()
correct = sum(1 for row, t in zip(logits, ys) if row.index(max(row)) == t)
acc = 100.0 * correct / len(ys)
print(f"\nAccuracy: {correct}/{len(ys)} = {acc:.1f}%")
