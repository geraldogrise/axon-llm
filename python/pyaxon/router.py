"""axon-lang hierarchical router: area -> sector -> subsector (free nesting).

**Compartmentalized** architecture: each internal node has a lightweight
classifier (incremental WordNB) trained only among its children. When routing,
only the current node's classifier is consulted and only the chosen branch is
descended -- activating the minimal fraction of the system.

**Incremental / online**: `partial_fit(text, path)` ingests one text, updates the
classifiers along the path and can discard the text afterwards (it does not keep
the raw corpus -- only the learned counts). Ideal for the "download -> train ->
delete" flow.
"""

import json
from collections import Counter

from ._axon import load as _ax_load
from ._axon import nn as _nn
from ._axon import save as _ax_save
from ._textclf import WordNB, tokenize


class _Node:
    def __init__(self):
        self.children = {}   # label -> _Node
        self.clf = None      # WordNB among the children (None at a leaf)


class HierarchicalRouter:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.root = _Node()
        self.last_activated_ = 0

    def partial_fit(self, text, path):
        """Ingest ONE labeled text (path [area, sector, subsector, ...]),
        updating the classifier at each level. The text can be discarded
        afterwards -- only the counts remain."""
        node = self.root
        for label in path:
            if node.clf is None:
                node.clf = WordNB(self.alpha)
            node.clf.partial_fit(text, label)
            node.children.setdefault(label, _Node())
            node = node.children[label]
        return self

    def fit(self, examples):
        """Convenience: examples = list of (text, path)."""
        for text, path in examples:
            self.partial_fit(text, path)
        return self

    def route(self, text, max_depth=None):
        """Descend the hierarchy activating only each path node's classifier."""
        path, node, activated = [], self.root, 0
        while node.children and (max_depth is None or len(path) < max_depth):
            label = node.clf.predict(text)
            activated += 1
            if label not in node.children:  # safety
                label = next(iter(node.children))
            path.append(label)
            node = node.children[label]
        self.last_activated_ = activated
        return path

    def route_verbose(self, text, max_depth=None):
        steps, node, activated = [], self.root, 0
        while node.children and (max_depth is None or len(steps) < max_depth):
            proba = node.clf.predict_proba(text)
            label = max(proba, key=proba.get)
            if label not in node.children:
                label = next(iter(node.children))
            steps.append((label, round(proba.get(label, 1.0), 3)))
            activated += 1
            node = node.children[label]
        self.last_activated_ = activated
        return steps

    # --- persistence (saves the "training": the classifiers' counts) ---
    def _dump(self, node):
        return {
            "clf": None if node.clf is None else node.clf.to_dict(),
            "children": {lbl: self._dump(ch) for lbl, ch in node.children.items()},
        }

    def _load(self, d):
        node = _Node()
        if d["clf"] is not None:
            node.clf = WordNB.from_dict(d["clf"])
        node.children = {lbl: self._load(ch) for lbl, ch in d["children"].items()}
        return node

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"alpha": self.alpha, "tree": self._dump(self.root)}, f)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.alpha = d["alpha"]
        self.root = self._load(d["tree"])
        return self


# ---------------------------------------------------------------------------
# HIGH-ACCURACY router: TF-IDF + Logistic Regression per node.
# ---------------------------------------------------------------------------
class _LinNode:
    def __init__(self):
        self.children = {}
        self.bags = []       # accumulated [Counter] (text already discarded)
        self.labels = []     # child label for each bag
        self.vec = None      # fitted TfidfVectorizer
        self.clf = None      # LogisticRegression
        self.only = None     # label of the single child (no choice)


class LinearRouter:
    """Hierarchical router with TF-IDF + Logistic Regression at each node (the model
    that reached ~94% area accuracy). Keeps the "discard the text" flow: in
    `partial_fit` only the (compact) **bag of words** is kept; `fit()` trains the
    classifiers; the raw text is never stored.
    """

    def __init__(self, ngram=2, max_features=5000, min_df=2, epochs=500, lr=0.05,
                 weight_decay=1e-4, balance=True, batch_size=0):
        self.ngram = ngram
        self.max_features = max_features
        self.min_df = min_df
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.balance = balance   # subsample each class to the minimum (avoids collapse)
        self.batch_size = batch_size   # 0 = full-batch; >0 = mini-batch (faster, less RAM)
        self.root = _LinNode()
        self.last_activated_ = 0

    def _bag(self, text):
        toks = tokenize(text)
        if self.ngram >= 2:
            toks = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        return Counter(toks)

    def partial_fit(self, text, path):
        """Ingest a text: store the bag at each path node and discard the text."""
        bag = self._bag(text)
        node = self.root
        for label in path:
            node.children.setdefault(label, _LinNode())
            node.bags.append(bag)
            node.labels.append(label)
            node = node.children[label]
        return self

    def fit(self):
        """Train the classifier (TF-IDF + LogReg) at each node, from the bags."""
        from . import ml, pre

        import numpy as np

        def train(node):
            if node.children:
                if len(set(node.labels)) > 1:
                    bags, labels = node.bags, node.labels
                    if self.balance:  # subsample each class to the minimum (balances)
                        by = {}
                        for b, y in zip(node.bags, node.labels):
                            by.setdefault(y, []).append(b)
                        m = min(len(v) for v in by.values())
                        bags, labels = [], []
                        for y, bs in by.items():
                            for b in bs[:m]:
                                bags.append(b)
                                labels.append(y)
                    vec = pre.TfidfVectorizer(self.max_features, self.ngram, self.min_df)
                    vec.fit_counts(bags)
                    X = vec.transform_counts(bags)
                    clf = ml.LogisticRegression(epochs=self.epochs, lr=self.lr,
                                                weight_decay=self.weight_decay,
                                                batch_size=self.batch_size)
                    clf.fit(X, np.array(labels))
                    node.vec, node.clf = vec, clf
                else:
                    node.only = node.labels[0]
                # Do NOT free the bags: LinearRouter re-trains in batch at each fit()
                for ch in node.children.values():
                    train(ch)
        train(self.root)
        return self

    def route(self, text):
        bag = self._bag(text)
        node, path, activated = self.root, [], 0
        while node.children:
            if node.clf is None:
                label = node.only
            else:
                X = node.vec.transform_counts([bag])
                label = str(node.clf.predict(X)[0])
                activated += 1
            if label not in node.children:
                label = next(iter(node.children))
            path.append(label)
            node = node.children[label]
        self.last_activated_ = activated
        return path

    def route_multi(self, text, k=2, threshold=0.0):
        """MULTI-SUBJECT routing (mixture-of-experts style): returns the `k`
        most likely paths (or all with prob. >= threshold), each with its weight.
        For interdisciplinary questions (e.g. physics + mathematics), it activates
        more than one expert -- but still only the relevant ones, not the whole net."""
        bag = self._bag(text)
        node = self.root
        if node.clf is None:  # single area
            return [(self.route(text), 1.0)]
        proba = node.clf.predict_proba(node.vec.transform_counts([bag]))[0]
        scored = sorted(zip([str(c) for c in node.clf.classes_], proba),
                        key=lambda x: x[1], reverse=True)
        chosen = [(a, p) for a, p in scored if p >= threshold] if threshold > 0 else scored[:k]
        chosen = chosen or scored[:1]

        results = []
        for area, w in chosen:
            path, cur = [area], node.children[area]
            while cur.children:  # complete the path (argmax) within the area
                if cur.clf is None:
                    lbl = cur.only
                else:
                    lbl = str(cur.clf.predict(cur.vec.transform_counts([bag]))[0])
                    if lbl not in cur.children:
                        lbl = next(iter(cur.children))
                path.append(lbl)
                cur = cur.children[lbl]
            results.append((path, float(w)))
        return results

    # --- persistence (JSON structure + LogReg weights in a binary file) ---
    def save(self, path):
        tensors = []

        def dump(node):
            clf = None
            if node.clf is not None:
                clf = {"classes": [str(c) for c in node.clf.classes_],
                       "vocab": node.vec.vocab_, "idf": list(map(float, node.vec.idf_))}
                for p in node.clf._lin.parameters():
                    tensors.append(p)
            return {"clf": clf, "only": node.only,
                    "children": {lbl: dump(ch) for lbl, ch in node.children.items()}}

        tree = dump(self.root)
        meta = {"ngram": self.ngram, "max_features": self.max_features, "min_df": self.min_df}
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "tree": tree}, f)
        if tensors:
            _ax_save(tensors, path + ".w")

    def load(self, path):
        import numpy as np

        from . import ml, pre
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.ngram = d["meta"]["ngram"]
        self.max_features = d["meta"]["max_features"]
        self.min_df = d["meta"]["min_df"]
        tensors = []

        def build(dd):
            node = _LinNode()
            node.only = dd["only"]
            if dd["clf"] is not None:
                vec = pre.TfidfVectorizer(self.max_features, self.ngram, self.min_df)
                vec.vocab_ = dd["clf"]["vocab"]
                vec.idf_ = np.array(dd["clf"]["idf"])
                clf = ml.LogisticRegression()
                clf.classes_ = np.array(dd["clf"]["classes"])
                clf._lin = _nn.Linear(len(vec.vocab_), len(clf.classes_))
                for p in clf._lin.parameters():
                    tensors.append(p)
                node.vec, node.clf = vec, clf
            node.children = {lbl: build(ch) for lbl, ch in dd["children"].items()}
            return node

        self.root = build(d["tree"])
        if tensors:
            _ax_load(tensors, path + ".w")  # fills the weights in-place (same order)
        return self
