"""Build the Python specialist: one expert per FAMILY.

Same modular design as the Java/.NET/JS specialists. Families/experts:
core (Python language), web (Flask/FastAPI/Django), data (pandas/numpy),
ml (scikit-learn/PyTorch/TensorFlow), testing (pytest), tooling (env/packages).
Each framework (flask, fastapi, django, pandas, ...) is a subsector inside its family.

Reads <family>/<subsector>/*.md -> path = [family, subsector]. Data lives in the
sibling `treinamento` repo (branch fase-5): ../treinamento/treinamneto_python/.
Saves the modular router + a sparse semantic KB to axon_lang_data/python_experts/.

Env: AXON_PY_DIR (data dir; default sibling repo, else local), AXON_EPOCHS (300),
     AXON_BATCH (256), AXON_LSA_DIM (200).
"""

import glob
import os

import pyaxon as ax
from pyaxon import manifest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# fase-5 data (sibling repo, folder name kept as created: "treinamneto_python").
_DATA = os.path.join(ROOT, "..", "treinamento", "treinamneto_python")
_LOCAL = os.path.join(ROOT, "treinamento_programacao", "python")
PY_DIR = os.environ.get("AXON_PY_DIR", _DATA if os.path.isdir(_DATA) else _LOCAL)
OUT = os.path.join(HERE, "axon_lang_data", "python_experts")
R_PREFIX = os.path.join(OUT, "router")
KB_PATH = os.path.join(OUT, "kb.sparse.json.gz")

EPOCHS = int(os.environ.get("AXON_EPOCHS", 300))
BATCH = int(os.environ.get("AXON_BATCH", 256))
LSA_DIM = int(os.environ.get("AXON_LSA_DIM", 200))

QUESTIONS = [
    ("core", "o que são decorators e como usá-los em Python?"),
    ("core", "como funcionam geradores e a palavra-chave yield?"),
    ("core", "como usar type hints e o módulo typing?"),
    ("web", "como criar uma rota e retornar JSON no Flask?"),
    ("web", "como declarar um endpoint com Pydantic no FastAPI?"),
    ("web", "como definir um model e uma view no Django?"),
    ("data", "como filtrar e agrupar um DataFrame no pandas?"),
    ("data", "como criar e fatiar arrays com o numpy?"),
    ("ml", "como treinar um classificador com o scikit-learn?"),
    ("ml", "como definir uma rede neural com nn.Module no PyTorch?"),
    ("ml", "como montar um modelo Sequential no TensorFlow/Keras?"),
    ("testing", "como escrever um teste com fixtures no pytest?"),
    ("tooling", "para que serve um ambiente virtual e o pip?"),
]


def read_py():
    docs = []
    for fp in glob.glob(os.path.join(PY_DIR, "**", "*.md"), recursive=True):
        parts = os.path.relpath(fp, PY_DIR).replace("\\", "/").split("/")[:-1]
        if not parts:
            continue
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        if len(text) >= 200:
            docs.append((parts, text))
    return docs


def main():
    os.makedirs(OUT, exist_ok=True)
    docs = read_py()
    if not docs:
        print(f"sem dados em {PY_DIR}", flush=True)
        return
    per_family = {}
    for parts, _ in docs:
        per_family[parts[0]] = per_family.get(parts[0], 0) + 1
    print(f"python: {len(docs)} lessons | families/experts: {per_family}", flush=True)

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
    manifest.write_manifest(R_PREFIX, model="ModularRouter/python",
                            counts={"experts": sorted(per_family), "lessons": len(docs)})
    print(f"saved: {R_PREFIX}.* | {KB_PATH} | passages={len(kb.texts)}", flush=True)

    ok = 0
    print("\n=== Python questions: routing + retrieved answer ===", flush=True)
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
