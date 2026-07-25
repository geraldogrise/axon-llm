"""axon-lang: retrain from LOCAL generated content (treinamento_portugues/).

Reads the original Portuguese lessons written under treinamento_portugues/ (one folder
per area/setor/subsetor -- the folder path IS the label), merges them with the passages
already collected from Wikipedia/Wikilivros/Wikisource (rag_multi/kb.json.gz), and
retrains the router + knowledge base. The generated content is **oversampled** so the
weak areas (história, português) get extra weight and their routing accuracy goes up.

Output goes to axon_lang_data/rag_final/ (router.json + kb.json.gz), leaving the raw
collection untouched. Ends with a question-level routing accuracy report.

Env vars: AXON_OVERSAMPLE (repeat each local doc N times for the router, default 3),
          AXON_WIKI_CAP (max Wikipedia chunks per area fed to the router, default 4000).
"""

import glob
import os
import random

import pyaxon as ax
from pyaxon import compress

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOCAL = os.path.join(ROOT, "treinamento_portugues")
WIKI_KB = os.path.join(HERE, "axon_lang_data", "rag_multi", "kb.json.gz")
OUT = os.path.join(HERE, "axon_lang_data", "rag_final")
R_PATH = os.path.join(OUT, "router.json")
KB_PATH = os.path.join(OUT, "kb.json.gz")

OVERSAMPLE = int(os.environ.get("AXON_OVERSAMPLE", 3))
WIKI_CAP = int(os.environ.get("AXON_WIKI_CAP", 4000))
EPOCHS = int(os.environ.get("AXON_EPOCHS", 250))
MAXFEAT = int(os.environ.get("AXON_MAXFEAT", 5000))

QUESTIONS = [
    ("matematica", "o que é o determinante de uma matriz?"),
    ("matematica", "como calcular a derivada de uma função?"),
    ("matematica", "qual a fórmula da equação da reta no plano cartesiano?"),
    ("fisica", "como a energia é transferida na termodinâmica?"),
    ("fisica", "o que dizem as leis de Newton sobre o movimento?"),
    ("fisica", "o que é um campo magnético e a indução eletromagnética?"),
    ("biologia", "qual o papel do DNA na hereditariedade?"),
    ("biologia", "como funciona a fotossíntese nas plantas?"),
    ("biologia", "o que é um ecossistema e a cadeia alimentar?"),
    ("quimica", "o que é uma ligação covalente entre átomos?"),
    ("quimica", "como é organizada a tabela periódica dos elementos?"),
    ("quimica", "o que é um hidrocarboneto na química orgânica?"),
    ("portugues", "o que é análise sintática de uma frase?"),
    ("portugues", "quais são as principais figuras de linguagem?"),
    ("portugues", "quem foi Machado de Assis na literatura brasileira?"),
    ("historia", "o que foram as capitanias hereditárias no brasil colônia?"),
    ("historia", "o que foi a era Vargas?"),
    ("historia", "o que foi a revolução francesa?"),
]


def read_local():
    """Read every .md under treinamento_portugues/; label = folder path parts."""
    docs = []
    for fp in glob.glob(os.path.join(LOCAL, "**", "*.md"), recursive=True):
        rel = os.path.relpath(fp, LOCAL).replace("\\", "/")
        parts = rel.split("/")[:-1]                 # area / setor / subsetor
        if not parts:
            continue
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        if len(text) >= 200:
            docs.append((parts, text))
    return docs


def main():
    os.makedirs(OUT, exist_ok=True)
    router = ax.router.LinearRouter(max_features=MAXFEAT, ngram=2, epochs=EPOCHS, lr=0.05, balance=True)
    kb = ax.rag.KnowledgeBase(max_features=MAXFEAT, ngram=1)

    # 1) local generated lessons (strong, balanced signal) -- oversampled for the router
    local = read_local()
    by_area = {}
    for parts, _ in local:
        by_area[parts[0]] = by_area.get(parts[0], 0) + 1
    print(f"local: {len(local)} lessons | per area: {by_area}", flush=True)
    for parts, text in local:
        kb.add_document(text, parts)                # KB keeps the full lessons
        for _ in range(OVERSAMPLE):                 # router sees each lesson N times
            router.partial_fit(text, parts)

    # 2) merge Wikipedia/Wikilivros/Wikisource passages (breadth), capped per area
    if os.path.exists(WIKI_KB):
        d = compress.load_json_gz(WIKI_KB)
        seen_area = {}
        pairs = list(zip(d["texts"], d["paths"]))
        random.Random(0).shuffle(pairs)
        used = 0
        for text, path in pairs:
            a = path[0]
            if seen_area.get(a, 0) >= WIKI_CAP:
                continue
            seen_area[a] = seen_area.get(a, 0) + 1
            kb.add_document(text, path)             # note: text is already a chunk
            router.partial_fit(text, path)
            used += 1
        print(f"wiki: merged {used} passages | per area: {seen_area}", flush=True)
    else:
        print("wiki: rag_multi/kb.json.gz not found -- training on local only", flush=True)

    print("training router + building base index...", flush=True)
    router.fit()
    kb.build()
    router.save(R_PATH)
    kb.save(KB_PATH)
    print(f"saved: {R_PATH} | {KB_PATH} "
          f"({os.path.getsize(KB_PATH) // 1024} KB) | passages={len(kb.texts)}", flush=True)

    # 3) question-level routing accuracy (the metric that matches real use)
    ok = 0
    print("\n=== routing questions ===", flush=True)
    for area, q in QUESTIONS:
        pr = router.route(q)
        hit = pr[:1] == [area]
        ok += hit
        print(f"  [{'OK' if hit else 'X '}] {area:11} <- {' > '.join(pr)}", flush=True)
    print(f"\nÁREA accuracy on real questions: {ok}/{len(QUESTIONS)} = {ok / len(QUESTIONS):.0%}",
          flush=True)


if __name__ == "__main__":
    main()
