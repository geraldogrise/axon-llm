# pyaxon — Classic ML and preprocessing (scikit-learn / pandas style)

Beyond deep learning, pyaxon brings classic machine learning models and
normalization utilities, with a familiar `fit` / `predict` / `score` API.
They operate on NumPy arrays (convert with `ax.from_numpy` / `.numpy()`).

> Honesty: this is **not** full parity with scikit-learn/pandas/numpy (those are huge
> libraries). It is a solid, tested set of the most-used components.

## Preprocessing — `pyaxon.pre`

| Class / function | What it does |
|-----------------|-----------|
| `StandardScaler` | standardizes to mean 0, std 1 (`fit`/`transform`/`fit_transform`/`inverse_transform`) |
| `MinMaxScaler`   | scales to `[0,1]` (or a given range) |
| `normalize(X, norm)` | normalizes samples to unit norm (`l1`/`l2`/`max`) |
| `one_hot(y, k)`  | integer labels → one-hot matrix |
| `train_test_split(X, y, test_size, seed)` | splits into train/test |

## Models — `pyaxon.ml`

| Model | Method | Note |
|--------|--------|------------|
| `LinearRegression` | normal equation (least squares) | `score` = R² |
| `LogisticRegression` | **trained with pyaxon itself** (`nn.Linear` + `cross_entropy` + `Adam`) | multiclass; `predict_proba` |
| `GaussianNB` | Gaussian Naive Bayes | analytic |
| `KMeans` | Lloyd + **k-means++** + `n_init` restarts | `cluster_centers_`, `labels_`, `inertia_` |
| `KNeighborsClassifier` | k nearest neighbors | majority vote |
| `DecisionTreeClassifier` | CART with Gini impurity | `max_depth`, `min_samples_split` |

## Example

```python
import numpy as np
import pyaxon as ax

X, y = ...  # your data (NumPy)
Xtr, Xte, ytr, yte = ax.pre.train_test_split(X, y, test_size=0.3, seed=1)

scaler = ax.pre.StandardScaler().fit(Xtr)
Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

clf = ax.ml.LogisticRegression(epochs=300, lr=0.1).fit(Xtr, ytr)
print("accuracy:", clf.score(Xte, yte))

reg = ax.ml.LinearRegression().fit(Xr, yr)
print("R²:", reg.score(Xr, yr))

km = ax.ml.KMeans(n_clusters=3).fit(X)
print("centers:", km.cluster_centers_)
```

See `examples/ml_demo.py` for a complete demonstration.

> `pyaxon.pre` and `pyaxon.ml` are loaded **on demand** (NumPy is only imported
> when you access them). The pyaxon core keeps working without NumPy.
