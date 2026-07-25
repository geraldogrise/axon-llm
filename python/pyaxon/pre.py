"""scikit-learn / pandas style preprocessing (normalization, scalers, split).

Works on NumPy arrays. Convert to/from Tensor with `ax.from_numpy` / `.numpy()`.
"""

import math
from collections import Counter

import numpy as np

from ._textclf import tokenize


class TfidfVectorizer:
    """Vectorizes text into TF-IDF (unigrams + optionally bigrams) -- strong
    features for topic classification. `fit`/`transform`/`fit_transform`."""

    def __init__(self, max_features=3000, ngram=1, min_df=2):
        self.max_features = max_features
        self.ngram = ngram
        self.min_df = min_df
        self.vocab_ = {}
        self.idf_ = None

    def _terms(self, text):
        toks = tokenize(text)
        if self.ngram >= 2:
            toks = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        return toks

    def fit(self, texts):
        df = Counter()
        for t in texts:
            df.update(set(self._terms(t)))
        # keep the most frequent terms (with df >= min_df)
        terms = [w for w, c in df.most_common() if c >= self.min_df][: self.max_features]
        self.vocab_ = {w: i for i, w in enumerate(terms)}
        n = len(texts)
        self.idf_ = np.array([math.log((1 + n) / (1 + df[w])) + 1.0 for w in terms])
        return self

    def transform(self, texts):
        rows = np.zeros((len(texts), len(self.vocab_)), dtype=np.float64)
        for i, t in enumerate(texts):
            counts = Counter(self._terms(t))
            for w, c in counts.items():
                j = self.vocab_.get(w)
                if j is not None:
                    rows[i, j] = c
            norm = np.linalg.norm(rows[i]) or 1.0
            rows[i] /= norm  # normalize TF per document
        rows *= self.idf_    # apply IDF
        # re-normalize L2 (standard TF-IDF)
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return rows / norms

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

    # --- variants that operate on COUNTS (bags), allowing the text to be discarded ---
    def fit_counts(self, counters):
        """Fit from a list of Counter(word->count) (text already discarded)."""
        df = Counter()
        for c in counters:
            df.update(set(c))
        terms = [w for w, cnt in df.most_common() if cnt >= self.min_df][: self.max_features]
        self.vocab_ = {w: i for i, w in enumerate(terms)}
        n = len(counters)
        self.idf_ = np.array([math.log((1 + n) / (1 + df[w])) + 1.0 for w in terms])
        return self

    def transform_counts(self, counters):
        rows = np.zeros((len(counters), len(self.vocab_)), dtype=np.float64)
        for i, c in enumerate(counters):
            for w, cnt in c.items():
                j = self.vocab_.get(w)
                if j is not None:
                    rows[i, j] = cnt
            norm = np.linalg.norm(rows[i]) or 1.0
            rows[i] /= norm
        rows *= self.idf_
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return rows / norms


class StandardScaler:
    """Standardize features to mean 0 and std 1: z = (x - mean) / std."""

    def __init__(self, eps=1e-8):
        self.mean_ = None
        self.std_ = None
        self.eps = eps

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / (self.std_ + self.eps)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return np.asarray(X) * (self.std_ + self.eps) + self.mean_


class MinMaxScaler:
    """Scale each feature to the [0, 1] range (or a given [min, max])."""

    def __init__(self, feature_range=(0.0, 1.0)):
        self.min_ = None
        self.max_ = None
        self.range = feature_range

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        span = np.where(self.max_ > self.min_, self.max_ - self.min_, 1.0)
        scaled = (X - self.min_) / span
        lo, hi = self.range
        return scaled * (hi - lo) + lo

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def normalize(X, norm="l2", axis=1, eps=1e-12):
    """Normalize samples to unit norm (l1, l2 or max) -- like sklearn.normalize."""
    X = np.asarray(X, dtype=np.float64)
    if norm == "l1":
        n = np.abs(X).sum(axis=axis, keepdims=True)
    elif norm == "max":
        n = np.abs(X).max(axis=axis, keepdims=True)
    else:  # l2
        n = np.sqrt((X * X).sum(axis=axis, keepdims=True))
    return X / (n + eps)


def one_hot(y, num_classes=None):
    """Convert integer labels to a one-hot matrix (n, num_classes)."""
    y = np.asarray(y, dtype=np.int64).ravel()
    if num_classes is None:
        num_classes = int(y.max()) + 1
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def train_test_split(X, y, test_size=0.25, seed=0, shuffle=True):
    """Split (X, y) into train and test. Returns X_train, X_test, y_train, y_test."""
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    n_test = int(round(n * test_size))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
