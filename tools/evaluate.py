"""Proper evaluation of the router: stratified split + per-class metrics.

Replaces "18 hand-picked questions" with a real held-out evaluation over the generated
lessons: trains a ModularRouter on a train split and reports, per area, precision /
recall / F1 plus a confusion matrix and macro-F1. This is how you know the model is
robust (or where it confuses areas).

Usage: python tools/evaluate.py            # evaluates area-level routing
       AXON_TEST=0.2 python tools/evaluate.py
"""

import glob
import os
import random

import pyaxon as ax

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = os.environ.get("AXON_LESSONS_DIR",
                       os.path.join(ROOT, "..", "treinamento", "treinamento_portugues"))
TEST_FRAC = float(os.environ.get("AXON_TEST", 0.2))


def read_local():
    items = []
    for fp in glob.glob(os.path.join(LOCAL, "**", "*.md"), recursive=True):
        parts = os.path.relpath(fp, LOCAL).replace("\\", "/").split("/")[:-1]
        if not parts:
            continue
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        if len(text) >= 200:
            items.append((parts, text))
    return items


def confusion_and_metrics(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    cm = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if p in idx:
            cm[idx[t]][idx[p]] += 1
    print("\n=== matriz de confusão (linha=verdadeiro, coluna=previsto) ===")
    print("            " + "".join(f"{l[:6]:>8}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l[:10]:>10}  " + "".join(f"{cm[i][j]:>8}" for j in range(len(labels))))

    print("\n=== métricas por área ===")
    print(f"{'área':>12} {'precisão':>10} {'recall':>10} {'F1':>10} {'suporte':>10}")
    f1s = []
    for i, l in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels))) - tp
        fn = sum(cm[i]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        print(f"{l:>12} {prec:>10.1%} {rec:>10.1%} {f1:>10.1%} {sum(cm[i]):>10}")
    acc = sum(cm[i][i] for i in range(len(labels))) / max(1, len(y_true))
    print(f"\nacurácia: {acc:.1%} | macro-F1: {sum(f1s)/len(f1s):.1%}")


def main():
    data = read_local()
    if not data:
        print("sem dados em treinamento_portugues/")
        return
    rng = random.Random(0)
    rng.shuffle(data)
    # stratified split by area
    by_area = {}
    for path, text in data:
        by_area.setdefault(path[0], []).append((path, text))
    train, test = [], []
    for area, items in by_area.items():
        cut = max(1, int(len(items) * (1 - TEST_FRAC)))
        train += items[:cut]
        test += items[cut:]
    print(f"treino {len(train)} | teste {len(test)} | áreas {sorted(by_area)}")

    r = ax.modular.ModularRouter(epochs=150)
    for path, text in train:
        r.add(text, path)
    r.fit(dirty_only=False)

    labels = sorted(by_area)
    y_true = [p[0] for p, _ in test]
    y_pred = [(r.route(t) or ["?"])[0] for _, t in test]
    confusion_and_metrics(y_true, y_pred, labels)


if __name__ == "__main__":
    main()
