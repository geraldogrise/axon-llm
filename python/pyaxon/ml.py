"""Classic machine learning models, scikit-learn style.

- LinearRegression      (normal equation via NumPy)
- LogisticRegression    (dogfood: pyaxon nn.Linear + cross_entropy + Adam)
- GaussianNB            (Gaussian Naive Bayes)
- KMeans                (clustering, Lloyd)
- KNeighborsClassifier  (k nearest neighbors)
- DecisionTreeClassifier(CART with Gini impurity)

Familiar API: fit(X, y) / predict(X) / score(X, y). Works on NumPy arrays.
"""

import numpy as np

from ._axon import (cross_entropy, from_numpy, is_grad_enabled, nn, optim, set_grad_enabled,
                    softmax, tensor)


def _to_tensor(X):
    """Convert a NumPy matrix into a pyaxon Tensor (fast, via from_numpy)."""
    return from_numpy(np.ascontiguousarray(X, dtype=np.float32))


class _ClassifierMixin:
    def score(self, X, y):
        """Mean accuracy."""
        return float((self.predict(X) == np.asarray(y)).mean())


# ---------------------------------------------------------------------------
# Linear regression (normal equation / least squares)
# ---------------------------------------------------------------------------
class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])  # bias column
        coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        self.coef_ = coef[:-1]
        self.intercept_ = coef[-1]
        return self

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def score(self, X, y):
        """Coefficient of determination R^2."""
        y = np.asarray(y, dtype=np.float64)
        yp = self.predict(X)
        ss_res = float(((y - yp) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Logistic regression -- trained with pyaxon itself (autograd + Adam)
# ---------------------------------------------------------------------------
class LogisticRegression(_ClassifierMixin):
    def __init__(self, epochs=300, lr=0.1, seed=0, weight_decay=0.0, batch_size=0):
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.weight_decay = weight_decay
        self.batch_size = batch_size    # 0 = full-batch; >0 = mini-batch SGD
        self.classes_ = None
        self._lin = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y).astype(np.float64)
        n, n_features = X.shape
        n_classes = len(self.classes_)

        self._lin = nn.Linear(n_features, n_classes, seed=self.seed)
        opt = optim.Adam(self._lin.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if not self.batch_size or self.batch_size >= n:
            # full-batch: one big forward/backward per epoch
            Xt = _to_tensor(X)
            yt = tensor(y_idx.tolist())
            for _ in range(self.epochs):
                loss = cross_entropy(self._lin(Xt), yt)
                opt.zero_grad(); loss.backward(); opt.step()
            return self

        # mini-batch SGD: lower peak memory (never materializes the full X @ W graph)
        # and usually faster convergence on large, high-dimensional data.
        rng = np.random.default_rng(self.seed)
        bs = self.batch_size
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                Xb = _to_tensor(X[idx])
                yb = tensor(y_idx[idx].tolist())
                loss = cross_entropy(self._lin(Xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        prev = is_grad_enabled()
        set_grad_enabled(False)
        try:
            return np.asarray(softmax(self._lin(_to_tensor(X))).numpy())
        finally:
            set_grad_enabled(prev)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ---------------------------------------------------------------------------
# Gaussian Naive Bayes
# ---------------------------------------------------------------------------
class GaussianNB(_ClassifierMixin):
    def __init__(self, eps=1e-9):
        self.eps = eps

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.theta_, self.var_, self.priors_ = [], [], []
        for c in self.classes_:
            Xc = X[y == c]
            self.theta_.append(Xc.mean(axis=0))
            self.var_.append(Xc.var(axis=0) + self.eps)
            self.priors_.append(len(Xc) / len(X))
        self.theta_ = np.array(self.theta_)
        self.var_ = np.array(self.var_)
        self.priors_ = np.array(self.priors_)
        return self

    def _joint_log_likelihood(self, X):
        jll = []
        for i in range(len(self.classes_)):
            log_prior = np.log(self.priors_[i])
            ll = -0.5 * np.sum(
                np.log(2.0 * np.pi * self.var_[i]) + (X - self.theta_[i]) ** 2 / self.var_[i],
                axis=1,
            )
            jll.append(log_prior + ll)
        return np.array(jll).T  # (n, n_classes)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.classes_[self._joint_log_likelihood(X).argmax(axis=1)]


# ---------------------------------------------------------------------------
# K-Means (unsupervised clustering)
# ---------------------------------------------------------------------------
class KMeans:
    def __init__(self, n_clusters=3, iters=100, seed=0, n_init=10):
        self.n_clusters = n_clusters
        self.iters = iters
        self.seed = seed
        self.n_init = n_init  # restarts (keep the one with lowest inertia)
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _dist(self, X, C):
        return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)

    def _init_pp(self, X, rng):
        """k-means++ initialization (spread-out centers) -- more robust."""
        n = len(X)
        centers = [X[rng.integers(n)]]
        d2 = ((X - centers[0]) ** 2).sum(axis=1)  # squared dist to nearest center
        for _ in range(1, self.n_clusters):
            total = d2.sum()
            if total <= 0:
                centers.append(X[rng.integers(n)])
            else:
                # pick the next center proportional to squared distance (favors far)
                idx = int(rng.choice(n, p=d2 / total))
                centers.append(X[idx])
            d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(axis=1))
        return np.array(centers)

    def _single_fit(self, X, rng):
        C = self._init_pp(X, rng)
        labels = self._dist(X, C).argmin(axis=1)
        for _ in range(self.iters):
            newC = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else C[k]
                for k in range(self.n_clusters)
            ])
            new_labels = self._dist(X, newC).argmin(axis=1)
            C = newC
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
        inertia = float(self._dist(X, C)[np.arange(len(X)), labels].sum())
        return C, labels, inertia

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            C, labels, inertia = self._single_fit(X, rng)
            if best is None or inertia < best[2]:
                best = (C, labels, inertia)
        self.cluster_centers_, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._dist(X, self.cluster_centers_).argmin(axis=1)


# ---------------------------------------------------------------------------
# k nearest neighbors (classification)
# ---------------------------------------------------------------------------
class KNeighborsClassifier(_ClassifierMixin):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = []
        for x in X:
            d = ((self.X_ - x) ** 2).sum(axis=1)
            idx = np.argsort(d)[: self.k]
            vals, counts = np.unique(self.y_[idx], return_counts=True)
            out.append(vals[counts.argmax()])
        return np.array(out)


# ---------------------------------------------------------------------------
# Decision tree (CART, Gini impurity)
# ---------------------------------------------------------------------------
class DecisionTreeClassifier(_ClassifierMixin):
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    @staticmethod
    def _gini(y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1.0 - float((p * p).sum())

    def _majority(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[counts.argmax()]

    def _build(self, X, y, depth):
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or len(np.unique(y)) == 1):
            return {"leaf": self._majority(y)}
        n, n_feat = X.shape
        best, best_gini = None, np.inf
        for feat in range(n_feat):
            for thr in np.unique(X[:, feat]):
                left = X[:, feat] <= thr
                if left.sum() == 0 or left.sum() == n:
                    continue
                g = (left.sum() * self._gini(y[left])
                     + (~left).sum() * self._gini(y[~left])) / n
                if g < best_gini:
                    best_gini, best = g, (feat, thr)
        if best is None:
            return {"leaf": self._majority(y)}
        feat, thr = best
        left = X[:, feat] <= thr
        return {
            "feat": feat, "thr": thr,
            "left": self._build(X[left], y[left], depth + 1),
            "right": self._build(X[~left], y[~left], depth + 1),
        }

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.tree_ = self._build(X, y, 0)
        return self

    def _predict_one(self, x, node):
        while "leaf" not in node:
            node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
        return node["leaf"]

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_one(x, self.tree_) for x in X])


# ---------------------------------------------------------------------------
# PCA (dimensionality reduction via SVD)
# ---------------------------------------------------------------------------
class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        k = self.n_components
        self.components_ = vt[:k]
        var = s ** 2 / max(len(X) - 1, 1)
        self.explained_variance_ratio_ = (var / var.sum())[:k]
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return np.asarray(Z, dtype=np.float64) @ self.components_ + self.mean_


# ---------------------------------------------------------------------------
# Linear SVM (hinge loss, one-vs-rest, subgradient descent)
# ---------------------------------------------------------------------------
class LinearSVC(_ClassifierMixin):
    def __init__(self, epochs=300, lr=0.01, C=1.0, seed=0):
        self.epochs = epochs
        self.lr = lr
        self.C = C            # inverse regularization strength
        self.seed = seed
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n, d = X.shape
        self._W = np.zeros((len(self.classes_), d))
        self._b = np.zeros(len(self.classes_))
        rng = np.random.default_rng(self.seed)
        for c, cls in enumerate(self.classes_):              # one-vs-rest
            t = np.where(y == cls, 1.0, -1.0)
            w, b = rng.normal(0, 0.01, d), 0.0
            for _ in range(self.epochs):
                margin = t * (X @ w + b)
                mask = margin < 1                            # points inside the margin
                grad_w = w - self.C * (X[mask] * t[mask, None]).sum(axis=0)
                grad_b = -self.C * t[mask].sum()
                w -= self.lr / n * grad_w
                b -= self.lr / n * grad_b
            self._W[c], self._b[c] = w, b
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=np.float64) @ self._W.T + self._b

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]


# ---------------------------------------------------------------------------
# Random Forest (bagging of decision trees + feature subsampling)
# ---------------------------------------------------------------------------
class RandomForestClassifier(_ClassifierMixin):
    def __init__(self, n_estimators=20, max_depth=6, max_features="sqrt", seed=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n, d = X.shape
        if self.max_features == "sqrt":
            self._mf = max(1, int(np.sqrt(d)))
        elif isinstance(self.max_features, int):
            self._mf = min(self.max_features, d)
        else:
            self._mf = d
        rng = np.random.default_rng(self.seed)
        self.trees_ = []
        for _ in range(self.n_estimators):
            rows = rng.integers(0, n, n)                     # bootstrap sample
            feats = rng.choice(d, self._mf, replace=False)   # random feature subset
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X[np.ix_(rows, feats)], y[rows])
            self.trees_.append((tree, feats))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        votes = np.array([tree.predict(X[:, feats]) for tree, feats in self.trees_])
        out = []
        for j in range(X.shape[0]):
            vals, counts = np.unique(votes[:, j], return_counts=True)
            out.append(vals[counts.argmax()])
        return np.array(out)
