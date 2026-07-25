"""Build the Go specialist: one expert per FAMILY.

Same modular design as the other specialists. Families/experts:
core (linguagem), concorrencia, web (net/http, gin, echo), data (sql, gorm),
ecossistema (modulos, testes, tooling). Each subsector is a subsector inside its family.

Reads <family>/<subsector>/*.md -> path = [family, subsector]. Data lives in
treinamento_go/ (local in this repo, or in the sibling `treinamento` data repo).
Saves the modular router + a sparse semantic KB to axon_lang_data/go_experts/.

Env: AXON_GO_DIR (data dir), AXON_EPOCHS (300), AXON_BATCH (256), AXON_LSA_DIM (200).
"""

import glob
import os

import pyaxon as ax
from pyaxon import manifest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_LOCAL = os.path.join(ROOT, "treinamento_go")
_DATA = os.path.join(ROOT, "..", "treinamento", "treinamento_go")
GO_DIR = os.environ.get("AXON_GO_DIR", _LOCAL if os.path.isdir(_LOCAL) else _DATA)
OUT = os.path.join(HERE, "axon_lang_data", "go_experts")
R_PREFIX = os.path.join(OUT, "router")
KB_PATH = os.path.join(OUT, "kb.sparse.json.gz")

EPOCHS = int(os.environ.get("AXON_EPOCHS", 300))
BATCH = int(os.environ.get("AXON_BATCH", 256))
LSA_DIM = int(os.environ.get("AXON_LSA_DIM", 200))

QUESTIONS = [
    ("core", "como declarar structs e interfaces em Go?"),
    ("core", "como funciona o tratamento de erros idiomático em Go?"),
    ("core", "o que são slices e maps e como usá-los?"),
    ("concorrencia", "como usar goroutines e canais em Go?"),
    ("concorrencia", "como usar select e sync.WaitGroup?"),
    ("web", "como criar um servidor HTTP com net/http?"),
    ("web", "como criar rotas e middleware com o Gin?"),
    ("data", "como usar database/sql com prepared statements?"),
    ("data", "como mapear structs com o GORM?"),
    ("ecossistema", "como usar go modules e o go.mod?"),
    ("ecossistema", "como escrever testes e table-driven tests em Go?"),
]


def read_go():
    docs = []
    for fp in glob.glob(os.path.join(GO_DIR, "**", "*.md"), recursive=True):
        parts = os.path.relpath(fp, GO_DIR).replace("\\", "/").split("/")[:-1]
        if not parts:
            continue
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        if len(text) >= 200:
            docs.append((parts, text))
    return docs


def main():
    os.makedirs(OUT, exist_ok=True)
    docs = read_go()
    if not docs:
        print(f"sem dados em {GO_DIR}", flush=True)
        return
    per_family = {}
    for parts, _ in docs:
        per_family[parts[0]] = per_family.get(parts[0], 0) + 1
    print(f"go: {len(docs)} lessons | families/experts: {per_family}", flush=True)

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
    manifest.write_manifest(R_PREFIX, model="ModularRouter/go",
                            counts={"experts": sorted(per_family), "lessons": len(docs)})
    print(f"saved: {R_PREFIX}.* | {KB_PATH} | passages={len(kb.texts)}", flush=True)

    ok = 0
    print("\n=== Go questions: routing + retrieved answer ===", flush=True)
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
