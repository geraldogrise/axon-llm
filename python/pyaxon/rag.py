"""axon-lang answering stage (RAG -- Retrieval-Augmented Generation).

The router picks the subsector of a question; the KnowledgeBase stores content
chunks indexed by path (area -> sector -> subsector -> ...) and retrieves the most
relevant ones to build the answer. Compartmentalized: it only searches inside the
subsector chosen by the router (fast and precise).

Retrieval uses SEMANTIC embeddings (LSA) -- it matches meaning, not just keywords.
Unlike the router (which discards text), here we KEEP the chunks; that is what lets
the model actually answer, not merely classify.
"""

import json
import re

import numpy as np

from . import compress, embed

# Lines/sections that are noise, not content (references, external links, etc.).
_NOISE = re.compile(r"(refer[êe]ncias|liga[çc][õo]es externas|ver tamb[ée]m|bibliografia|"
                    r"categoria:|ISBN|https?://|www\.|\.com|\.org|\bem ingl[êe]s\b)", re.I)


def clean_text(text):
    """Drop reference/link/navigation noise, keep readable content lines."""
    kept = []
    for line in text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if len(line) < 40:
            continue
        if _NOISE.search(line):
            continue
        letters = sum(c.isalpha() or c.isspace() for c in line)
        if letters / max(len(line), 1) < 0.75:   # too many symbols/numbers -> noise
            continue
        kept.append(line)
    return "\n".join(kept)


def split_chunks(text, max_chars=600, min_chars=80):
    """Split cleaned text into chunks (paragraphs grouped up to max_chars)."""
    text = clean_text(text)
    chunks, cur = [], ""
    for para in [p.strip() for p in text.split("\n") if p.strip()]:
        if len(cur) + len(para) > max_chars and cur:
            chunks.append(cur)
            cur = para
        else:
            cur = (cur + " " + para).strip()
    if len(cur) >= min_chars:
        chunks.append(cur)
    return chunks


class KnowledgeBase:
    """Knowledge base compartmentalized by path (area -> sector -> subsector -> ...),
    with semantic (LSA) retrieval."""

    def __init__(self, dim=200, max_features=6000, ngram=1):
        self.dim = dim
        self.max_features = max_features
        self.ngram = ngram
        self.paths = []        # path for each chunk
        self.texts = []        # chunk text
        self.embedder = None
        self.matrix = None     # (n_chunks, dim) semantic embeddings

    def add_document(self, text, path):
        for chunk in split_chunks(text):
            self.paths.append(list(path))
            self.texts.append(chunk)
        return self

    def build(self):
        """Fit the semantic embedder over all stored chunks."""
        self.embedder = embed.LSAEmbedder(self.dim, self.max_features, self.ngram, min_df=1)
        self.matrix = self.embedder.fit_transform(self.texts)
        return self

    def retrieve(self, query, path_prefix=None, top_k=3):
        """Return the top_k most semantically similar chunks. If path_prefix is given,
        search only inside that subsector (compartmentalized)."""
        q = self.embedder.transform([query])[0]              # (dim,)
        candidates = list(range(len(self.texts)))
        if path_prefix:
            pref = list(path_prefix)
            selected = [i for i in candidates if self.paths[i][:len(pref)] == pref]
            candidates = selected or candidates              # no match -> global search
        sims = self.matrix[candidates] @ q                   # cosine (L2-normalized)
        order = np.argsort(sims)[::-1][:top_k]
        return [(self.texts[candidates[j]], self.paths[candidates[j]], float(sims[j]))
                for j in order]

    def answer(self, query, router=None, top_k=3, multi=0):
        """Route and retrieve. Returns (paths, [(chunk, chunk_path, score), ...]).

        multi>1 uses route_multi (search the `multi` most likely subsectors and merge
        the best chunks) -- fixes ambiguous / interdisciplinary questions."""
        if router is None:
            return None, self.retrieve(query, None, top_k)
        if multi and multi > 1 and hasattr(router, "route_multi"):
            paths = [p for p, _ in router.route_multi(query, k=multi)]
            pool = []
            for prefix in paths:
                pool += self.retrieve(query, prefix, top_k)
            seen, merged = set(), []
            for chunk, cpath, score in sorted(pool, key=lambda x: x[2], reverse=True):
                key = chunk[:80]
                if key in seen:
                    continue
                seen.add(key)
                merged.append((chunk, cpath, score))
                if len(merged) >= top_k:
                    break
            return paths, merged
        prefix = router.route(query)
        return [prefix], self.retrieve(query, prefix, top_k)

    # --- persistence -------------------------------------------------------
    # Plain format (.json): stores chunks + paths; the LSA index is rebuilt on load.
    # Compressed format (.json.gz): gzip + int8-quantized embeddings, so the file is
    # ~4-5x smaller AND load skips the SVD recompute.
    def save(self, path, compressed=None):
        """Save the knowledge base. `compressed` defaults to True when `path` ends
        with `.gz`. The compressed format also persists the quantized index."""
        if compressed is None:
            compressed = path.endswith(".gz")
        if not compressed:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"dim": self.dim, "max_features": self.max_features,
                           "ngram": self.ngram, "paths": self.paths, "texts": self.texts},
                          f, ensure_ascii=False)
            return self

        payload = {"format": "gz-int8", "dim": self.dim,
                   "max_features": self.max_features, "ngram": self.ngram,
                   "paths": self.paths, "texts": self.texts}
        if self.embedder is not None and self.matrix is not None:
            payload["vocab"] = self.embedder.vectorizer.vocab_
            payload["idf"] = [float(v) for v in self.embedder.vectorizer.idf_]
            payload["components_q"] = compress.quantize_int8(self.embedder.components)
            payload["matrix_q"] = compress.quantize_int8(self.matrix)
        compress.save_json_gz(payload, path)
        return self

    def load(self, path):
        compressed = path.endswith(".gz")
        d = compress.load_json_gz(path) if compressed else _read_json(path)
        self.dim = d.get("dim", 200)
        self.max_features = d["max_features"]
        self.ngram = d["ngram"]
        self.paths = d["paths"]
        self.texts = d["texts"]
        if compressed and "matrix_q" in d:
            # Restore the quantized index directly -- no TF-IDF/SVD recompute.
            self.embedder = embed.LSAEmbedder(self.dim, self.max_features, self.ngram)
            self.embedder.vectorizer.vocab_ = d["vocab"]
            self.embedder.vectorizer.idf_ = np.asarray(d["idf"], dtype=np.float64)
            self.embedder.components = compress.dequantize_int8(d["components_q"])
            self.matrix = compress.dequantize_int8(d["matrix_q"])
        elif self.texts:
            self.build()
        return self


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
