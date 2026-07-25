"""Demonstrates pyaxon's classic ML models (scikit-learn style).

Preprocessing (pyaxon.pre) + models (pyaxon.ml), all with the fit/predict/score API.
"""

import numpy as np

import pyaxon as ax


def blobs(seed=0, n=80):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [6.0, 6.0], [0.0, 6.0]])
    X = np.vstack([c + rng.normal(0, 0.6, (n, 2)) for c in centers])
    y = np.concatenate([np.full(n, i) for i in range(3)])
    return X, y


X, y = blobs()
Xtr, Xte, ytr, yte = ax.pre.train_test_split(X, y, test_size=0.3, seed=1)

# Normalization (StandardScaler): fit on train, apply to both.
scaler = ax.pre.StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

print("=== Classification (3 classes) — test accuracy ===")
for name, model in [
    ("LogisticRegression", ax.ml.LogisticRegression(epochs=300, lr=0.1)),
    ("GaussianNB", ax.ml.GaussianNB()),
    ("KNeighbors (k=5)", ax.ml.KNeighborsClassifier(k=5)),
    ("DecisionTree", ax.ml.DecisionTreeClassifier(max_depth=4)),
]:
    model.fit(Xtr_s, ytr)
    print(f"  {name:22s}: {100 * model.score(Xte_s, yte):.1f}%")

print("\n=== Linear regression ===")
rng = np.random.default_rng(0)
Xr = rng.normal(0, 1, (200, 3))
yr = 2.0 * Xr[:, 0] - 1.5 * Xr[:, 1] + 0.5 * Xr[:, 2] + 4.0 + rng.normal(0, 0.05, 200)
reg = ax.ml.LinearRegression().fit(Xr, yr)
print(f"  coefficients ~[2, -1.5, 0.5]: {np.round(reg.coef_, 2)}")
print(f"  R² = {reg.score(Xr, yr):.4f}")

print("\n=== KMeans (unsupervised) ===")
km = ax.ml.KMeans(n_clusters=3, seed=0).fit(X)
print(f"  inertia = {km.inertia_:.1f} | centers found:")
for c in np.round(km.cluster_centers_, 1):
    print("   ", c)
