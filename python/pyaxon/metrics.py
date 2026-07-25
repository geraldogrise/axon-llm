"""Classification metrics (scikit-learn style): accuracy, precision, recall, F1,
confusion matrix and a per-class report. Works on NumPy arrays / lists.
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def confusion_matrix(y_true, y_pred, labels=None):
    """Rows = true, columns = predicted. Returns (matrix, labels)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm, labels


def precision_recall_fscore(y_true, y_pred, labels=None, average=None):
    """Per-class (average=None) or 'macro'/'micro' precision, recall, F1.
    Returns (precision, recall, f1, support) as arrays (or floats when averaged)."""
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    if average == "macro":
        return float(prec.mean()), float(rec.mean()), float(f1.mean()), int(support.sum())
    if average == "micro":
        t = tp.sum()
        p = t / (t + fp.sum()) if t + fp.sum() else 0.0
        r = t / (t + fn.sum()) if t + fn.sum() else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return float(p), float(r), float(f), int(support.sum())
    return prec, rec, f1, support


def f1_score(y_true, y_pred, average="macro"):
    return precision_recall_fscore(y_true, y_pred, average=average)[2]


def classification_report(y_true, y_pred, labels=None):
    """Human-readable per-class table + accuracy and macro-F1."""
    prec, rec, f1, sup = precision_recall_fscore(y_true, y_pred, labels)
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    lines = [f"{'class':>14} {'prec':>8} {'recall':>8} {'f1':>8} {'support':>8}"]
    for i, l in enumerate(labels):
        lines.append(f"{str(l):>14} {prec[i]:>8.2f} {rec[i]:>8.2f} {f1[i]:>8.2f} {sup[i]:>8}")
    acc = accuracy_score(y_true, y_pred)
    lines.append(f"\n{'accuracy':>14} {acc:>8.2f}    macro-F1 {f1.mean():.2f}")
    return "\n".join(lines)
