"""Semantic embeddings via LSA (TF-IDF + truncated SVD).

Captures latent meaning (synonyms, related topics) far better than raw keyword
matching, without any pretrained neural encoder -- just linear algebra. Each text
is mapped to a dense low-dimensional vector whose cosine similarity reflects
semantic closeness, not just shared words.
"""

import numpy as np

from . import pre


class LSAEmbedder:
    """Latent Semantic Analysis embedder: TF-IDF followed by truncated SVD."""

    def __init__(self, dim=200, max_features=6000, ngram=1, min_df=2):
        self.dim = dim
        self.vectorizer = pre.TfidfVectorizer(max_features, ngram, min_df)
        self.components = None      # (dim, vocab) top right singular vectors

    def fit(self, texts):
        X = self.vectorizer.fit_transform(texts)            # (n, vocab) dense TF-IDF
        k = max(1, min(self.dim, min(X.shape) - 1))
        try:
            from scipy.sparse.linalg import svds
            _, _, vt = svds(X.astype(np.float64), k=k)       # truncated (top-k) SVD
            self.components = vt
        except Exception:
            _, _, vt = np.linalg.svd(X, full_matrices=False)  # fallback
            self.components = vt[:k]
        return self

    def transform(self, texts):
        X = self.vectorizer.transform(texts)                 # (n, vocab)
        emb = X @ self.components.T                           # (n, dim) semantic vectors
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms                                   # L2-normalized (for cosine)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)
