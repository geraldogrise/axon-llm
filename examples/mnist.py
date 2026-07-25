"""MNIST digit classifier with pyaxon (recognize numbers — plan.md).

Uses the IDX loader (`pyaxon.data.load_idx`) and trains an MLP. Downloads nothing:
point it at local MNIST IDX files. If they don't exist, it prints instructions and exits.

Download (IDX format) from any MNIST mirror and unpack, e.g.:
  train-images-idx3-ubyte, train-labels-idx1-ubyte
  t10k-images-idx3-ubyte,  t10k-labels-idx1-ubyte
"""

import os
import sys

import pyaxon as ax
from pyaxon import data

# Directory with the IDX files (adjust or pass as an argument).
MNIST_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "mnist")

FILES = {
    "train_x": "train-images-idx3-ubyte",
    "train_y": "train-labels-idx1-ubyte",
    "test_x": "t10k-images-idx3-ubyte",
    "test_y": "t10k-labels-idx1-ubyte",
}


def main():
    paths = {k: os.path.join(MNIST_DIR, v) for k, v in FILES.items()}
    if not all(os.path.exists(p) for p in paths.values()):
        print("MNIST IDX files not found in:", MNIST_DIR)
        print("Download the 4 MNIST IDX files and place them in this folder (or pass the path).")
        print("The loader (ax.data.load_idx) is ready; only the data is missing.")
        return

    # Load and normalize (pixels 0..255 -> 0..1).
    N_TRAIN, N_TEST = 4000, 1000  # subset for fast CPU training
    xr, xs = data.load_idx(paths["train_x"])
    yr, _ = data.load_idx(paths["train_y"])
    xtr, xts_shape = data.load_idx(paths["test_x"])
    ytr, _ = data.load_idx(paths["test_y"])
    px = xs[1] * xs[2]  # 28*28 = 784

    def make(imgs, labels, n):
        X = [[imgs[i * px + j] / 255.0 for j in range(px)] for i in range(n)]
        y = [float(labels[i]) for i in range(n)]
        return ax.tensor(X), ax.tensor(y)

    Xtr, Ytr = make(xr, yr, N_TRAIN)
    Xte, Yte = make(xtr, ytr, N_TEST)
    print(f"MNIST: train={N_TRAIN} test={N_TEST} features={px}")

    model = ax.nn.Sequential([
        ax.nn.Linear(px, 128, seed=1),
        ax.nn.ReLU(),
        ax.nn.Linear(128, 10, seed=2),
    ])
    opt = ax.optim.Adam(model.parameters(), lr=0.001)

    BATCH = 64
    Xtr_l, Ytr_l = Xtr.tolist(), Ytr.tolist()
    for epoch in range(1, 6):
        total = n = 0
        for s in range(0, N_TRAIN, BATCH):
            xb = ax.tensor(Xtr_l[s:s + BATCH])
            yb = ax.tensor(Ytr_l[s:s + BATCH])
            loss = ax.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1
        # test accuracy
        with ax.no_grad():
            logits = model(Xte).tolist()
        correct = sum(1 for row, t in zip(logits, Yte.tolist()) if row.index(max(row)) == t)
        print(f"epoch {epoch} | loss={total / n:.4f} | test accuracy={100 * correct / N_TEST:.1f}%")


if __name__ == "__main__":
    main()
