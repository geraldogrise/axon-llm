"""Build the JavaScript/TypeScript specialist: one expert per FAMILY.

Same modular design as the Java/.NET specialists. Families/experts: core (JS language),
typescript, node, react, angular, vue, meta (Next/Nuxt/Remix), tooling (npm/bundlers).
React/Angular/Vue are their OWN experts (distinct ecosystems). Semantic hybrid retrieval.

Reads treinamento_js/<family>/<subsector>/*.md -> path = [family, subsector].
Saves the modular router + a sparse semantic KB to axon_lang_data/js_experts/.

Env: AXON_JS_DIR (data dir; default local, else sibling repo), AXON_EPOCHS (300),
     AXON_BATCH (256), AXON_LSA_DIM (200).
"""

import glob
import os

import pyaxon as ax
from pyaxon import manifest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_LOCAL = os.path.join(ROOT, "treinamento_js")
_DATA = os.path.join(ROOT, "..", "treinamento", "treinamento_js")
JS_DIR = os.environ.get("AXON_JS_DIR", _LOCAL if os.path.isdir(_LOCAL) else _DATA)
OUT = os.path.join(HERE, "axon_lang_data", "js_experts")
R_PREFIX = os.path.join(OUT, "router")
KB_PATH = os.path.join(OUT, "kb.sparse.json.gz")

EPOCHS = int(os.environ.get("AXON_EPOCHS", 300))
BATCH = int(os.environ.get("AXON_BATCH", 256))
LSA_DIM = int(os.environ.get("AXON_LSA_DIM", 200))

QUESTIONS = [
    ("core", "o que são closures em JavaScript?"),
    ("core", "como funcionam as Promises e o async/await?"),
    ("core", "como manipular o DOM e adicionar eventos?"),
    ("typescript", "o que são generics e utility types em TypeScript?"),
    ("node", "como criar um middleware no Express?"),
    ("react", "como usar o useState e o useEffect no React?"),
    ("angular", "o que é injeção de dependência e services no Angular?"),
    ("vue", "o que é a Composition API e a reatividade no Vue?"),
    ("meta", "o que são Server Components e o App Router no Next.js?"),
    ("tooling", "para que serve o package.json e o npm?"),
]


def read_js():
    docs = []
    for fp in glob.glob(os.path.join(JS_DIR, "**", "*.md"), recursive=True):
        parts = os.path.relpath(fp, JS_DIR).replace("\\", "/").split("/")[:-1]
        if not parts:
            continue
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        if len(text) >= 200:
            docs.append((parts, text))
    return docs


def main():
    os.makedirs(OUT, exist_ok=True)
    docs = read_js()
    if not docs:
        print(f"sem dados em {JS_DIR}", flush=True)
        return
    per_family = {}
    for parts, _ in docs:
        per_family[parts[0]] = per_family.get(parts[0], 0) + 1
    print(f"js: {len(docs)} lessons | families/experts: {per_family}", flush=True)

    router = ax.modular.ModularRouter(epochs=EPOCHS, batch_size=BATCH)
    kb = ax.vindex.SparseKB(ngram=1)
    for parts, text in docs:
        router.add(text, parts)
        kb.add_document(text, parts)
    print(f"training {len(per_family)} family experts (mini-batch)...", flush=True)
    router.fit(dirty_only=False)
    print(f"building semantic index (LSA dim={LSA_DIM})...", flush=True)
    kb.build(dim=LSA_DIM)

    router.save(R_PREFIX)
    kb.save(KB_PATH)
    manifest.write_manifest(R_PREFIX, model="ModularRouter/js",
                            counts={"experts": sorted(per_family), "lessons": len(docs)})
    print(f"saved: {R_PREFIX}.* | {KB_PATH} | passages={len(kb.texts)}", flush=True)

    ok = 0
    print("\n=== JS/TS questions: routing + retrieved answer ===", flush=True)
    for fam, q in QUESTIONS:
        pr = router.route(q)
        hit = pr[:1] == [fam]
        ok += hit
        passages = kb.retrieve(q, path_prefix=pr, top_k=1)
        snip = passages[0][0][:150].replace("\n", " ") if passages else "(nada)"
        print(f"  [{'OK' if hit else 'X '}] {' > '.join(pr):24} | {snip} ...", flush=True)
    print(f"\nFAMILY accuracy: {ok}/{len(QUESTIONS)} = {ok/len(QUESTIONS):.0%}", flush=True)


if __name__ == "__main__":
    main()
