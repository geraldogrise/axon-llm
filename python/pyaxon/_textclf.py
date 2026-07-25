"""Lightweight text classifier via char n-gram profiles (axon-lang base).

Classic technique, robust with little data: represents each class by its character
n-gram frequency profile and classifies by cosine similarity. Cheap and
"compartmentalizable" -- each router node uses one of these only among its children.
"""

import math
import re
import unicodedata
from collections import Counter

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # letter sequences


def clean(text):
    """Lowercase, keep letters, collapse whitespace."""
    text = text.lower()
    # keep letters (accented) and spaces; replace the rest with a space
    out = []
    for ch in text:
        if ch.isalpha() or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def char_ngrams(text, n):
    t = " " + clean(text) + " "
    return [t[i:i + n] for i in range(len(t) - n + 1)]


def profile(text, n=3, top=400):
    """Normalized frequency profile of the `top` most common n-grams."""
    counts = Counter(char_ngrams(text, n))
    total = sum(counts.values()) or 1
    most = counts.most_common(top)
    return {g: c / total for g, c in most}


def cosine(a, b):
    keys = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def tokenize(text):
    """Words (sequences of 2+ letters, lowercased, accented)."""
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) >= 2]


class WordNB:
    """Multinomial Naive Bayes over words -- strong for topic classification and
    **incremental**: accumulates per-class word counts (does not keep texts).

    - `partial_fit(text, label)`: ingests a text and discards it (only counts remain).
    - `fit({label: text})`: convenience (calls partial_fit).
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = []
        self.counts_ = {}   # label -> Counter(word->count)
        self.totals_ = {}   # label -> total word count
        self.vocab_ = set()

    def partial_fit(self, text, label):
        if label not in self.counts_:
            self.counts_[label] = Counter()
            self.totals_[label] = 0
            self.classes_.append(label)
        toks = tokenize(text)
        self.counts_[label].update(toks)
        self.totals_[label] += len(toks)
        self.vocab_.update(toks)
        return self

    def fit(self, class_texts):
        for label, text in class_texts.items():
            self.partial_fit(text, label)
        return self

    def scores(self, text):
        V = len(self.vocab_) or 1
        n = len(self.classes_) or 1
        toks = [t for t in tokenize(text) if t in self.vocab_]
        out = {}
        for c in self.classes_:
            denom = self.totals_[c] + self.alpha * V
            lp = math.log(1.0 / n)  # uniform prior
            cc = self.counts_[c]
            for t in toks:
                lp += math.log((cc.get(t, 0) + self.alpha) / denom)
            out[c] = lp
        return out

    def predict(self, text):
        sc = self.scores(text)
        return max(sc, key=sc.get) if sc else None

    def predict_proba(self, text):
        sc = self.scores(text)
        if not sc:
            return {}
        m = max(sc.values())
        exps = {k: math.exp(v - m) for k, v in sc.items()}
        s = sum(exps.values())
        return {k: v / s for k, v in exps.items()}

    @property
    def labels(self):
        return list(self.classes_)

    def to_dict(self):
        return {"alpha": self.alpha, "classes": self.classes_, "totals": self.totals_,
                "counts": {c: dict(cnt) for c, cnt in self.counts_.items()}}

    @classmethod
    def from_dict(cls, d):
        o = cls(d["alpha"])
        o.classes_ = list(d["classes"])
        o.totals_ = dict(d["totals"])
        o.counts_ = {c: Counter(w) for c, w in d["counts"].items()}
        o.vocab_ = set().union(*[set(w) for w in o.counts_.values()]) if o.counts_ else set()
        return o


class ProfileClassifier:
    """N-gram profile classifier. fit takes {label: concatenated_text}."""

    def __init__(self, n=3, top=400):
        self.n = n
        self.top = top
        self.profiles_ = {}

    def fit(self, class_texts):
        self.profiles_ = {
            label: profile(text, self.n, self.top) for label, text in class_texts.items()
        }
        return self

    def scores(self, text):
        p = profile(text, self.n, self.top)
        return {label: cosine(p, prof) for label, prof in self.profiles_.items()}

    def predict(self, text):
        sc = self.scores(text)
        return max(sc, key=sc.get) if sc else None

    def predict_proba(self, text):
        sc = self.scores(text)
        if not sc:
            return {}
        m = max(sc.values())
        exps = {k: math.exp((v - m) * 8.0) for k, v in sc.items()}  # temperature
        s = sum(exps.values())
        return {k: v / s for k, v in exps.items()}

    @property
    def labels(self):
        return list(self.profiles_)
