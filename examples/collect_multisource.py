"""axon-lang: knowledge collection from MULTIPLE sources, classifying everything.

Sources: Wikipedia + Wikibooks (educational) + Wikisource (literature) + PDFs (books).
Flow: download -> train the router -> **delete the text immediately** -> write the URL to the CSV.
Interleaves the subjects (curriculum) and is resumable (never re-downloads the same source).
"""

import os

import pyaxon as ax
from pyaxon import collect

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "axon_lang_data", "multi")
CKPT = os.path.join(OUT, "router.json")
TARGET = 60

# Taxonomy by area/sector (leaf = search query). Portuguese first.
TAXO = {
    "portugues": {
        "gramatica": ["gramática", "sintaxe", "morfologia (linguística)", "classes de palavras"],
        "linguistica": ["língua portuguesa", "semântica", "fonologia"],
    },
    "matematica": {"basico": ["matemática", "álgebra", "geometria", "número"]},
    "fisica": {"basico": ["física", "energia", "mecânica clássica"]},
    "biologia": {"basico": ["biologia", "célula", "genética"]},
}
# Literature (complete works) comes better from Wikisource:
LIT = {"portugues": {"literatura": ["Machado de Assis", "José de Alencar", "Aluísio Azevedo"]}}

# PDFs of books/handouts (the user can extend). Each one already classified:
PDF_SPECS = [
    (["matematica", "basico"], "https://arxiv.org/pdf/1802.06915"),  # e.g.: academic PDF
]


def main():
    os.makedirs(OUT, exist_ok=True)
    router = ax.router.HierarchicalRouter()
    if os.path.exists(CKPT):
        router.load(CKPT)
    col = collect.Collector(OUT)
    print(f"already indexed (no re-download): {col.count} sources")

    # Build specs from several sources and INTERLEAVE (to keep the corpus diverse).
    wiki = collect.build_mw_specs(TAXO, "wikipedia", per_query=4)
    books = collect.build_mw_specs(TAXO, "wikibooks", per_query=3)
    source = collect.build_mw_specs(LIT, "wikisource", per_query=4)

    def interleave(*lists):
        out, i = [], 0
        while any(i < len(l) for l in lists):
            for l in lists:
                if i < len(l):
                    out.append(l[i])
            i += 1
        return out

    specs = interleave(wiki, books, source) + PDF_SPECS
    print(f"candidate specs (wiki+wikibooks+wikisource+pdf): {len(specs)}")

    added = col.stream_train_specs(router, specs, target=TARGET, chunk=4,
                                   checkpoint=CKPT, save_every=20)

    print(f"\ncheckpoint: {os.path.getsize(CKPT)} bytes | CSV: {col.csv_path}")
    # show the diversity of collected sources
    import csv as _csv
    from collections import Counter
    with open(col.csv_path, encoding="utf-8") as f:
        sources = Counter(r["source"] for r in _csv.DictReader(f))
    print("sources in the index:", dict(sources))

    print("\n--- routing (only the model; texts already discarded) ---")
    for q in ["análise sintática e concordância verbal",
              "a obra literária de Machado de Assis",
              "resolver uma equação e calcular a área de um triângulo",
              "a célula e o material genético dos seres vivos"]:
        print(f"  {' > '.join(router.route(q))}  <-  \"{q[:45]}\"")


if __name__ == "__main__":
    main()
