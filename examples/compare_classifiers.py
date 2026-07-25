"""Compare the AREA classifier: WordNB (current) vs TF-IDF + Logistic Regression.

Downloads a small, BALANCED set separately (does not touch the collection running in
parallel), trains both on the SAME train split and evaluates on the SAME test split.
"""

import time

import numpy as np

import pyaxon as ax
from pyaxon import corpus
from pyaxon._textclf import WordNB

TRAIN_PER_AREA = 40
TEST_PER_AREA = 12
DELAY = 0.2  # polite (runs alongside the collection)

AREAS = {
    "portugues": ["gramática", "sintaxe", "língua portuguesa", "literatura brasileira",
                  "Machado de Assis", "semântica", "ortografia", "verbo"],
    "matematica": ["matemática", "álgebra", "geometria", "cálculo", "trigonometria",
                   "número primo", "função (matemática)", "estatística"],
    "fisica": ["física", "mecânica clássica", "energia", "termodinâmica", "óptica",
               "eletromagnetismo", "onda", "força"],
    "quimica": ["química", "átomo", "tabela periódica", "reação química", "molécula",
                "ácido", "química orgânica", "ligação química"],
    "biologia": ["biologia", "célula", "genética", "ecologia", "evolução", "DNA",
                 "fotossíntese", "botânica"],
    "historia": ["história do Brasil", "história", "idade média", "renascimento",
                 "segunda guerra mundial", "império romano", "grécia antiga", "idade moderna"],
}


def collect_data():
    """Download balanced articles per area: returns (texts, labels) train/test."""
    tr_x, tr_y, te_x, te_y = [], [], [], []
    for area, queries in AREAS.items():
        got, seen = [], set()
        for q in queries:
            if len(got) >= TRAIN_PER_AREA + TEST_PER_AREA:
                break
            try:
                titles = corpus.wikipedia_search(q, limit=20)
            except Exception:
                continue
            for t in titles:
                if t in seen:
                    continue
                seen.add(t)
                time.sleep(DELAY)
                try:
                    txt = corpus.wikipedia_extract(t)
                except Exception:
                    continue
                if len(txt) >= 400:
                    got.append(txt)
                if len(got) >= TRAIN_PER_AREA + TEST_PER_AREA:
                    break
        tr_x += got[:TRAIN_PER_AREA]; tr_y += [area] * min(TRAIN_PER_AREA, len(got))
        te_x += got[TRAIN_PER_AREA:]; te_y += [area] * max(0, len(got) - TRAIN_PER_AREA)
        print(f"  {area}: {len(got)} articles", flush=True)
    return tr_x, tr_y, te_x, te_y


def main():
    print("downloading balanced set (separate from the collection)...", flush=True)
    tr_x, tr_y, te_x, te_y = collect_data()
    print(f"train={len(tr_x)} test={len(te_x)}\n", flush=True)

    # --- 1) WordNB (the current classifier) ---
    nb = WordNB()
    for txt, area in zip(tr_x, tr_y):
        nb.partial_fit(txt, area)
    acc_nb = np.mean([nb.predict(x) == y for x, y in zip(te_x, te_y)])

    # --- 2) TF-IDF (uni+bigram) + Logistic Regression (regularized) ---
    vec = ax.pre.TfidfVectorizer(max_features=5000, ngram=2, min_df=2).fit(tr_x)
    Xtr, Xte = vec.transform(tr_x), vec.transform(te_x)
    clf = ax.ml.LogisticRegression(epochs=800, lr=0.3)
    clf.lr = 0.3
    # regularization via weight_decay in Adam (avoids overfitting in high dimension)
    import numpy as _np
    yidx = _np.searchsorted(_np.unique(tr_y), tr_y).astype(float)
    clf.classes_ = _np.unique(tr_y)
    clf._lin = ax.nn.Linear(Xtr.shape[1], len(clf.classes_), seed=0)
    opt = ax.optim.Adam(clf._lin.parameters(), lr=0.3, weight_decay=1e-4)
    Xt = ax.from_numpy(Xtr.astype("float32")); yt = ax.tensor(yidx.tolist())
    for _ in range(800):
        loss = ax.cross_entropy(clf._lin(Xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
    acc_lr = clf.score(Xte, np.array(te_y))

    print("=== AREA ACCURACY (same test split) ===")
    print(f"  WordNB (current)          : {100*acc_nb:.0f}%")
    print(f"  TF-IDF + LogReg (pyaxon)  : {100*acc_lr:.0f}%")


if __name__ == "__main__":
    main()
