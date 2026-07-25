"""Model/data versioning: a manifest written alongside every saved model.

Robustness needs reproducibility -- knowing *which data and config* produced a given
model. `write_manifest` records a version, timestamp, the training config, dataset
counts and a content hash, so you can tell two model versions apart and trace a model
back to its inputs.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

SCHEMA_VERSION = 1


def data_hash(items):
    """Stable hash of an iterable of strings (e.g. the training texts/paths)."""
    h = hashlib.sha256()
    for x in items:
        h.update(str(x).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def write_manifest(path, *, model, config=None, counts=None, data_fingerprint=None,
                   notes=None):
    """Write `<path>.manifest.json` describing a saved model.

    - model: a short name/kind of the model (e.g. "ModularRouter").
    - config: the hyperparameters dict.
    - counts: dict of dataset sizes (e.g. {"lessons": 854, "by_area": {...}}).
    - data_fingerprint: a hash from `data_hash(...)` tying the model to its data.
    """
    manifest = {
        "schema": SCHEMA_VERSION,
        "model": model,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config or {},
        "counts": counts or {},
        "data_fingerprint": data_fingerprint,
        "notes": notes,
    }
    mpath = path + ".manifest.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return mpath


def read_manifest(path):
    mpath = path if path.endswith(".manifest.json") else path + ".manifest.json"
    if not os.path.exists(mpath):
        return None
    with open(mpath, "r", encoding="utf-8") as f:
        return json.load(f)
