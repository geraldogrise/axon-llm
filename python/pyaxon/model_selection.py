"""Model selection utilities (scikit-learn style): K-fold and cross-validation."""

import numpy as np


class KFold:
    """K-fold splitter. Yields (train_index, test_index) arrays."""

    def __init__(self, n_splits=5, shuffle=True, seed=0):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed

    def split(self, X):
        n = len(X)
        idx = np.arange(n)
        if self.shuffle:
            np.random.default_rng(self.seed).shuffle(idx)
        folds = np.array_split(idx, self.n_splits)
        for i in range(self.n_splits):
            test = folds[i]
            train = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])
            yield train, test


def cross_val_score(estimator_factory, X, y, cv=5, seed=0):
    """K-fold cross-validation. `estimator_factory` is a no-arg callable returning a
    fresh model with fit()/score(). Returns the array of per-fold scores.

    Ex.: cross_val_score(lambda: ax.ml.LogisticRegression(), X, y, cv=5)
    """
    X, y = np.asarray(X), np.asarray(y)
    scores = []
    for train, test in KFold(cv, shuffle=True, seed=seed).split(X):
        model = estimator_factory().fit(X[train], y[train])
        scores.append(model.score(X[test], y[test]))
    return np.asarray(scores, dtype=float)
