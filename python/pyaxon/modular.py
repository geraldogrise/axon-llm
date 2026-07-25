"""Modular router: add knowledge to ONE area without retraining the others.

The flat LinearRouter retrains every node on each `fit()` because a single logistic
regression learns the boundaries between ALL classes at once. This module splits the
problem so adding data to, say, física only retrains física:

- **area gate** (`WordNB`, incremental): text -> area. Naive Bayes accumulates per-class
  word counts, so `add()` updates only the relevant area's counts -- no retraining.
- **per-area experts** (`LinearRouter`, accurate): each area has its OWN sub-router for
  its setor/subsetor hierarchy. Retraining física's expert never touches biologia's.

So the workflow becomes: `add(text, path)` (incremental) -> `fit(dirty_only=True)`
(retrains only the areas that received new data). Adding an area is O(that area), not
O(everything).
"""

import json
import os

from ._textclf import WordNB
from .router import LinearRouter


class ModularRouter:
    def __init__(self, alpha=1.0, ngram=2, max_features=5000, epochs=250, lr=0.05,
                 min_df=2, balance=True, batch_size=0):
        self.area_gate = WordNB(alpha)          # incremental area-level classifier
        self.experts = {}                        # area -> LinearRouter (sub-hierarchy)
        self._dirty = set()                      # areas with unfit new data
        self._cfg = dict(ngram=ngram, max_features=max_features, epochs=epochs, lr=lr,
                         min_df=min_df, balance=balance, batch_size=batch_size)

    def _expert(self, area):
        if area not in self.experts:
            self.experts[area] = LinearRouter(**self._cfg)
        return self.experts[area]

    # --- ingestion (incremental) -------------------------------------------
    def add(self, text, path):
        """Ingest one labeled text. Updates the area gate incrementally and stores the
        bag in the area's expert. Only that area becomes 'dirty' (needs refit)."""
        area = path[0]
        self.area_gate.partial_fit(text, area)   # incremental -- no retrain
        sub = list(path[1:])
        if sub:
            self._expert(area).partial_fit(text, sub)
            self._dirty.add(area)
        return self

    # --- training (modular) ------------------------------------------------
    def fit(self, areas=None, dirty_only=True):
        """Train experts. By default retrains ONLY the areas that got new data
        (`self._dirty`). Pass `areas=[...]` to force specific ones, or
        dirty_only=False to retrain all. The area gate needs no training."""
        targets = set(areas) if areas is not None else (self._dirty if dirty_only
                                                         else set(self.experts))
        for area in targets:
            if area in self.experts:
                self.experts[area].fit()
        self._dirty -= targets
        return self

    # --- routing -----------------------------------------------------------
    def route(self, text, threshold=0.0):
        """Route to area -> subsector. If `threshold` > 0 and the top area confidence
        is below it, returns [] (meaning "don't know / out of domain") instead of
        forcing a guess -- the robustness gate against confidently-wrong answers."""
        proba = self.area_gate.predict_proba(text)
        if not proba:
            return []
        area = max(proba, key=proba.get)
        if threshold > 0 and proba[area] < threshold:
            return []                                  # abstain: not confident enough
        exp = self.experts.get(area)
        if exp is not None and exp.root.children:
            return [area] + exp.route(text)
        return [area]

    def route_multi(self, text, k=2, threshold=0.0):
        """Return the top-k subsector paths within the chosen area (mixture-of-experts).
        Lets retrieval search more than one subsector and merge -- fixes cases where the
        area is right but the subsector is ambiguous (e.g. 'useState' -> hooks vs
        ecossistema). Returns [(path, weight), ...]."""
        proba = self.area_gate.predict_proba(text)
        if not proba:
            return []
        area = max(proba, key=proba.get)
        exp = self.experts.get(area)
        if exp is None or not exp.root.children:
            return [([area], 1.0)]
        return [([area] + p, w) for p, w in exp.route_multi(text, k=k, threshold=threshold)]

    def confidence(self, text):
        """Confidence (0..1) of the top area -- use to decide 'não sei' or to gate the
        generation layer (only answer when confident)."""
        proba = self.area_gate.predict_proba(text)
        return max(proba.values()) if proba else 0.0

    def route_proba_area(self, text):
        return self.area_gate.predict_proba(text)

    # --- persistence (per-area files -> unchanged areas are not rewritten) --
    def save(self, prefix):
        """Save to `prefix.gate.json` + one `prefix.<area>.json` per expert.
        Only rewrite what changed by calling save again -- experts are independent."""
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
        with open(prefix + ".gate.json", "w", encoding="utf-8") as f:
            json.dump({"alpha": self.area_gate.alpha, "nb": self.area_gate.to_dict(),
                       "cfg": self._cfg, "areas": sorted(self.experts)}, f)
        for area, exp in self.experts.items():
            exp.save(f"{prefix}.{_safe(area)}.json")
        from . import manifest
        manifest.write_manifest(prefix, model="ModularRouter", config=self._cfg,
                                counts={"areas": sorted(self.experts),
                                        "vocab": len(self.area_gate.vocab_)})
        return self

    def save_area(self, prefix, area):
        """Persist a single area's expert (+ the gate). Use after adding to one area."""
        with open(prefix + ".gate.json", "w", encoding="utf-8") as f:
            json.dump({"alpha": self.area_gate.alpha, "nb": self.area_gate.to_dict(),
                       "cfg": self._cfg, "areas": sorted(self.experts)}, f)
        if area in self.experts:
            self.experts[area].save(f"{prefix}.{_safe(area)}.json")
        return self

    def load(self, prefix):
        with open(prefix + ".gate.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        self.area_gate = WordNB.from_dict(d["nb"])
        self._cfg = d.get("cfg", self._cfg)
        self.experts = {}
        for area in d.get("areas", []):
            path = f"{prefix}.{_safe(area)}.json"
            if os.path.exists(path):
                self.experts[area] = LinearRouter(**self._cfg).load(path)
        return self


def _safe(name):
    import re
    return re.sub(r"[^\w.-]+", "_", str(name)).strip("_") or "area"
