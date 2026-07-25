"""Sparse, memory-light, append-friendly knowledge base for retrieval.

Keeps everything sparse (scipy.sparse) so it scales to 100k+ passages without the
dense-matrix OOM. Two retrieval modes:

- **keyword** (default): sparse TF-IDF cosine -- fast, exact term match.
- **semantic** (`build(dim=200)`): truncated SVD (LSA) computed **on the sparse
  matrix** via scipy's `svds`, so it stays memory-light. Matches meaning, not just
  words (fixes "fotossíntese" retrieving a "plantas" passage that only shares a word).

Also **source-aware**: each passage carries a source ("lesson" vs "wiki"); retrieval
boosts curated lessons over noisier wiki text, so the clean content wins ties.
"""

import gzip
import json
import math
from collections import Counter

import numpy as np

from ._textclf import tokenize
from .rag import split_chunks


class SparseKB:
    def __init__(self, ngram=1, min_df=1):
        self.ngram = ngram
        self.min_df = min_df
        self.paths = []
        self.texts = []
        self.sources = []        # per-passage source tag ("lesson", "wiki", ...)
        self.vocab_ = {}
        self.idf_ = None
        self.dim = 0             # 0 = keyword mode; >0 = semantic (LSA) mode
        self._matrix = None      # sparse csr (n, vocab), L2-normalized
        self._emb = None         # dense (n, dim) LSA embeddings (semantic mode)
        self._components = None  # (dim, vocab) LSA components

    def _terms(self, text):
        toks = tokenize(text)
        if self.ngram >= 2:
            toks = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        return toks

    def add_document(self, text, path, source="lesson"):
        for chunk in split_chunks(text):
            self.texts.append(chunk)
            self.paths.append(list(path))
            self.sources.append(source)
        return self

    def build(self, dim=0):
        """Build the sparse index. dim>0 also computes an LSA (semantic) projection
        via truncated SVD on the sparse matrix (memory-safe, no densification)."""
        from scipy import sparse

        docs_terms = [Counter(self._terms(t)) for t in self.texts]
        df = Counter()
        for c in docs_terms:
            df.update(c.keys())
        terms = [w for w, c in df.items() if c >= self.min_df]
        self.vocab_ = {w: i for i, w in enumerate(sorted(terms))}
        n, v = len(self.texts), len(self.vocab_)
        self.idf_ = np.zeros(v, dtype=np.float32)
        for w, i in self.vocab_.items():
            self.idf_[i] = math.log((1 + n) / (1 + df[w])) + 1.0

        indptr, indices, data = [0], [], []
        for c in docs_terms:
            row = {}
            for w, cnt in c.items():
                j = self.vocab_.get(w)
                if j is not None:
                    row[j] = cnt * self.idf_[j]
            norm = math.sqrt(sum(x * x for x in row.values())) or 1.0
            for j, x in row.items():
                indices.append(j)
                data.append(x / norm)
            indptr.append(len(indices))
        self._matrix = sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32), np.asarray(indices), np.asarray(indptr)),
            shape=(n, v))

        self.dim = dim
        self._emb = self._components = None
        if dim and n > 2 and v > 2:
            from scipy.sparse.linalg import svds
            k = max(1, min(dim, min(n, v) - 1))
            # SVD on the SPARSE matrix -> low memory (never densifies the n x v matrix)
            u, s, vt = svds(self._matrix.astype(np.float64), k=k)
            emb = (u * s).astype(np.float32)                 # (n, k) doc embeddings
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._emb = emb / norms
            self._components = vt.astype(np.float32)          # (k, vocab)
        return self

    def _sparse_vec(self, text):
        from scipy import sparse

        c = Counter(self._terms(text))
        idx, val = [], []
        for w, cnt in c.items():
            j = self.vocab_.get(w)
            if j is not None:
                idx.append(j)
                val.append(cnt * self.idf_[j])
        norm = math.sqrt(sum(x * x for x in val)) or 1.0
        val = [x / norm for x in val]
        return sparse.csr_matrix((val, ([0] * len(idx), idx)),
                                 shape=(1, len(self.vocab_)), dtype=np.float32)

    def _sims(self, query, alpha=0.5):
        """Similarity of `query` to every passage. HYBRID by default: combines keyword
        cosine (exact-term match) with semantic LSA cosine (meaning/paraphrase). alpha
        weights semantic vs keyword (0=keyword only, 1=semantic only)."""
        q = self._sparse_vec(query)
        kw = np.asarray((self._matrix @ q.T).todense()).ravel()   # keyword cosine
        if self._emb is None:
            return kw
        qe = np.asarray(q @ self._components.T).ravel()           # sparse @ dense
        nrm = np.linalg.norm(qe) or 1.0
        sem = self._emb @ (qe / nrm)                              # semantic cosine
        return (1 - alpha) * kw + alpha * sem                     # hybrid

    def retrieve(self, query, path_prefix=None, top_k=3, lesson_boost=1.4):
        sims = self._sims(query).copy()
        if lesson_boost != 1.0 and self.sources:
            for i, src in enumerate(self.sources):
                if src == "lesson":
                    sims[i] *= lesson_boost      # prioritize curated lessons over wiki
        candidates = np.arange(len(self.texts))
        if path_prefix:
            pref = list(path_prefix)
            sel = [i for i in candidates if self.paths[i][:len(pref)] == pref]
            candidates = np.asarray(sel) if sel else candidates
        order = candidates[np.argsort(sims[candidates])[::-1][:top_k]]
        return [(self.texts[i], self.paths[i], float(sims[i])) for i in order]

    def answer(self, query, router=None, top_k=3, multi=0):
        if router is None:
            return None, self.retrieve(query, None, top_k)
        if multi and multi > 1 and hasattr(router, "route_multi"):
            paths = [p for p, _ in router.route_multi(query, k=multi)]
            pool = []
            for prefix in paths:
                pool += self.retrieve(query, prefix, top_k)
            seen, merged = set(), []
            for chunk, cpath, score in sorted(pool, key=lambda x: x[2], reverse=True):
                if chunk[:80] in seen:
                    continue
                seen.add(chunk[:80])
                merged.append((chunk, cpath, score))
                if len(merged) >= top_k:
                    break
            return paths, merged
        prefix = router.route(query)
        return [prefix], self.retrieve(query, prefix, top_k)

    # --- persistence (gzip JSON of texts+paths+sources; index rebuilt on load) ---
    def save(self, path):
        payload = {"ngram": self.ngram, "min_df": self.min_df, "dim": self.dim,
                   "paths": self.paths, "texts": self.texts, "sources": self.sources}
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "wb") as f:
            f.write(raw)
        return self

    def load(self, path):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rb") as f:
            d = json.loads(f.read().decode("utf-8"))
        self.ngram = d["ngram"]
        self.min_df = d["min_df"]
        self.paths = d["paths"]
        self.texts = d["texts"]
        self.sources = d.get("sources", ["lesson"] * len(self.texts))
        if self.texts:
            self.build(dim=d.get("dim", 0))
        return self
