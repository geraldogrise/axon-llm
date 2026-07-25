"""Build the final consolidated axon-lang model -- robust edition.

Uses the robustness stack so it does NOT OOM and finishes quickly (the old dense
LSA + full-batch LogReg is what got killed):
  - ModularRouter  (incremental area gate + per-area experts, mini-batch)
  - SparseKB       (sparse TF-IDF index -- scales, no dense OOM)

Trains on the generated lessons (treinamento_portugues/) plus, optionally, the
Wikipedia/Wikilivros/Wikisource passages already collected in rag_multi/. Saves the
router, the sparse knowledge base and a version manifest to rag_final/, then reports
question-level routing accuracy.

Env: AXON_WIKI_CAP (wiki passages per area for the KB, default 1200; 0 = local only),
     AXON_EPOCHS (expert epochs, default 200), AXON_BATCH (mini-batch, default 256).
"""

import glob
import os
import random

import pyaxon as ax
from pyaxon import compress, manifest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Point at the data repo (sibling of pyaxon) -- no copy inside the code repo.
# Override with AXON_LESSONS_DIR. Requires `git checkout fase-1` in that repo first.
LOCAL = os.environ.get("AXON_LESSONS_DIR",
                       os.path.join(ROOT, "..", "treinamento", "treinamento_portugues"))
WIKI_KB = os.path.join(HERE, "axon_lang_data", "rag_multi", "kb.json.gz")
OUT = os.path.join(HERE, "axon_lang_data", "rag_final")
R_PREFIX = os.path.join(OUT, "router")          # modular -> router.gate.json + router.<area>.json
KB_PATH = os.path.join(OUT, "kb.sparse.json.gz")

WIKI_CAP = int(os.environ.get("AXON_WIKI_CAP", 1200))
EPOCHS = int(os.environ.get("AXON_EPOCHS", 300))     # more epochs -> better subsector routing
BATCH = int(os.environ.get("AXON_BATCH", 256))
LSA_DIM = int(os.environ.get("AXON_LSA_DIM", 200))   # semantic retrieval dimension

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
    docs = []
    for fp in glob.glob(os.path.join(LOCAL, "**", "*.md"), recursive=True):
        parts = os.path.relpath(fp, LOCAL).replace("\\", "/").split("/")[:-1]
        if not parts:
            continue
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        if len(text) >= 200:
            docs.append((parts, text))
    return docs


def main():
    os.makedirs(OUT, exist_ok=True)
    local = read_local()
    per_area = {}
    for parts, _ in local:
        per_area[parts[0]] = per_area.get(parts[0], 0) + 1
    print(f"local: {len(local)} lessons | per area: {per_area}", flush=True)

    # 1) modular router -- incremental gate + per-area experts (mini-batch)
    router = ax.modular.ModularRouter(epochs=EPOCHS, batch_size=BATCH)
    kb = ax.vindex.SparseKB(ngram=1)
    for parts, text in local:
        router.add(text, parts)
        kb.add_document(text, parts)
    print("training experts (per-area, mini-batch)...", flush=True)
    router.fit(dirty_only=False)

    # 2) merge Wikipedia passages into the KB for retrieval breadth (sparse -> safe)
    if WIKI_CAP and os.path.exists(WIKI_KB):
        d = compress.load_json_gz(WIKI_KB)
        pairs = list(zip(d["texts"], d["paths"]))
        random.Random(0).shuffle(pairs)
        seen = {}
        for text, path in pairs:
            a = path[0]
            if seen.get(a, 0) >= WIKI_CAP:
                continue
            seen[a] = seen.get(a, 0) + 1
            kb.texts.append(text)
            kb.paths.append(list(path))
            kb.sources.append("wiki")           # tagged so lessons are prioritized
        print(f"wiki merged into KB: {sum(seen.values())} passages | {seen}", flush=True)

    print(f"building semantic index (LSA dim={LSA_DIM})...", flush=True)
    kb.build(dim=LSA_DIM)

    router.save(R_PREFIX)
    kb.save(KB_PATH)
    manifest.write_manifest(KB_PATH, model="SparseKB",
                            counts={"passages": len(kb.texts), "vocab": len(kb.vocab_)})
    print(f"saved: {R_PREFIX}.* | {KB_PATH} | passages={len(kb.texts)}", flush=True)

    # 3) question-level routing accuracy
    ok = 0
    print("\n=== routing questions ===", flush=True)
    for area, q in QUESTIONS:
        pr = router.route(q)
        hit = pr[:1] == [area]
        ok += hit
        print(f"  [{'OK' if hit else 'X '}] {area:11} <- {' > '.join(pr) or '(vazio)'}", flush=True)
    print(f"\nÁREA accuracy on real questions: {ok}/{len(QUESTIONS)} = {ok/len(QUESTIONS):.0%}",
          flush=True)


if __name__ == "__main__":
    main()
