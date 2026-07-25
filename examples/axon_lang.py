"""axon-lang: compartmentalized pipeline text -> area -> sector -> subsector.

1) Downloads real data from Wikipedia (PT) for a taxonomy (with dedup via manifest).
2) Trains the language identifier (Portuguese) and the hierarchical router.
3) Routes a query activating ONLY the classifiers on the path (efficiency).

The dataset is saved as JSONL and reused (no re-download). Increase `PER_LEAF` to
pull more pages (the user asked for 1000+; start small to validate).
"""

import os
import random

import pyaxon as ax

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "axon_lang_data")
os.makedirs(DATA, exist_ok=True)
DATASET = os.path.join(DATA, "dataset.jsonl")
MANIFEST = os.path.join(DATA, "manifest.json")
PER_LEAF = 6  # articles per subsector (increase for a larger corpus)

# Taxonomy: area -> sector -> subsector (leaf = Wikipedia search query).
TAXONOMY = {
    "matematica": {
        "algebra": {"algebra linear": "álgebra linear", "algebra abstrata": "álgebra abstrata"},
        "teoria dos numeros": {"numeros primos": "número primo", "criptografia": "criptografia"},
        "geometria": {"euclidiana": "geometria euclidiana", "topologia": "topologia"},
    },
    "fisica": {
        "mecanica": {"classica": "mecânica clássica", "quantica": "mecânica quântica"},
        "termodinamica": {"termo": "termodinâmica", "entropia": "entropia (termodinâmica)"},
    },
    "biologia": {
        "genetica": {"dna": "DNA", "hereditariedade": "hereditariedade"},
        "ecologia": {"ecossistema": "ecossistema", "biodiversidade": "biodiversidade"},
    },
}


def get_dataset():
    if os.path.exists(DATASET):
        recs = ax.corpus.read_jsonl(DATASET)
        print(f"cached dataset: {len(recs)} articles")
        return recs
    print("downloading real data from Wikipedia (PT)...")
    recs = ax.corpus.build_wikipedia_dataset(TAXONOMY, per_leaf=PER_LEAF, lang="pt",
                                             manifest_path=MANIFEST)
    ax.corpus.write_jsonl(recs, DATASET)
    print(f"downloaded {len(recs)} articles -> {DATASET}")
    return recs


def main():
    recs = get_dataset()

    # train/test split by record
    random.seed(0)
    random.shuffle(recs)
    n_test = max(1, len(recs) // 5)
    test, train = recs[:n_test], recs[n_test:]

    # train the hierarchical router
    router = ax.router.HierarchicalRouter().fit([(r["text"], r["path"]) for r in train])

    # evaluate per level (area / sector / subsector)
    depth = max(len(r["path"]) for r in test)
    hits = [0] * depth
    exact = 0
    for r in test:
        pred = router.route(r["text"])
        if pred == r["path"]:
            exact += 1
        for lvl in range(min(len(pred), len(r["path"]))):
            if pred[lvl] == r["path"][lvl]:
                hits[lvl] += 1
    levels = ["area", "sector", "subsector"]
    print(f"\ntest ({len(test)} articles) — accuracy per level:")
    for lvl in range(depth):
        name = levels[lvl] if lvl < len(levels) else f"level {lvl+1}"
        print(f"  {name:9s}: {100*hits[lvl]/len(test):.0f}%")
    print(f"  full path: {100*exact/len(test):.0f}%")
    print("  (more data per subsector -> higher accuracy; increase PER_LEAF)")

    # language identifier
    lid = ax.langid.LanguageIdentifier().fit()

    print("\n--- routing queries (only the path is activated) ---")
    queries = [
        "quero calcular o determinante e os autovalores de uma matriz",
        "como funciona a distribuição dos números primos e a fatoração",
        "qual a energia de uma partícula no modelo quântico do átomo",
        "a transferência de calor e a variação de entropia em um gás",
        "a herança genética dos alelos e o papel do DNA",
    ]
    for q in queries:
        if not lid.is_portuguese(q):
            print(f"[{lid.predict(q)}] non-Portuguese, ignored:", q[:40])
            continue
        path = router.route(q)
        print(f"pt | {' > '.join(path)}  (activated {router.last_activated_} classifiers)")
        print(f"     \"{q}\"")


if __name__ == "__main__":
    main()
