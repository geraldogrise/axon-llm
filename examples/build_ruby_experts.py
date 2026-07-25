"""Build the Ruby specialist: one expert per FAMILY.

Same modular design as the other specialists. Families/experts:
ruby (a linguagem: sintaxe, colecoes, oop, avancado) and rails (o framework:
fundamentos, activerecord, recursos, testes). Each subsector is inside its family.

Reads <family>/<subsector>/*.md -> path = [family, subsector]. Data lives in
treinamento_ruby/ (local in this repo, or in the sibling `treinamento` data repo).
Saves the modular router + a sparse semantic KB to axon_lang_data/ruby_experts/.

Env: AXON_RUBY_DIR (data dir), AXON_EPOCHS (300), AXON_BATCH (256), AXON_LSA_DIM (200).
"""

import glob
import os

import pyaxon as ax
from pyaxon import manifest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_LOCAL = os.path.join(ROOT, "treinamento_ruby")
_DATA = os.path.join(ROOT, "..", "treinamento", "treinamento_ruby")
RUBY_DIR = os.environ.get("AXON_RUBY_DIR", _LOCAL if os.path.isdir(_LOCAL) else _DATA)
OUT = os.path.join(HERE, "axon_lang_data", "ruby_experts")
R_PREFIX = os.path.join(OUT, "router")
KB_PATH = os.path.join(OUT, "kb.sparse.json.gz")

EPOCHS = int(os.environ.get("AXON_EPOCHS", 300))
BATCH = int(os.environ.get("AXON_BATCH", 256))
LSA_DIM = int(os.environ.get("AXON_LSA_DIM", 200))

QUESTIONS = [
    ("ruby", "o que são blocks, procs e lambdas em Ruby?"),
    ("ruby", "como funciona orientação a objetos e mixins em Ruby?"),
    ("ruby", "como usar map, select e reduce em coleções?"),
    ("ruby", "o que é metaprogramação e define_method em Ruby?"),
    ("rails", "como criar um model e uma migration no Rails?"),
    ("rails", "como funciona o Active Record e as associações has_many?"),
    ("rails", "como criar controllers e rotas RESTful no Rails?"),
    ("rails", "como usar validações e callbacks no Active Record?"),
    ("rails", "como escrever testes com RSpec e FactoryBot?"),
]


def read_ruby():
    docs = []
    for fp in glob.glob(os.path.join(RUBY_DIR, "**", "*.md"), recursive=True):
        parts = os.path.relpath(fp, RUBY_DIR).replace("\\", "/").split("/")[:-1]
        if not parts:
            continue
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        if len(text) >= 200:
            docs.append((parts, text))
    return docs


def main():
    os.makedirs(OUT, exist_ok=True)
    docs = read_ruby()
    if not docs:
        print(f"sem dados em {RUBY_DIR}", flush=True)
        return
    per_family = {}
    for parts, _ in docs:
        per_family[parts[0]] = per_family.get(parts[0], 0) + 1
    print(f"ruby: {len(docs)} lessons | families/experts: {per_family}", flush=True)

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
    manifest.write_manifest(R_PREFIX, model="ModularRouter/ruby",
                            counts={"experts": sorted(per_family), "lessons": len(docs)})
    print(f"saved: {R_PREFIX}.* | {KB_PATH} | passages={len(kb.texts)}", flush=True)

    ok = 0
    print("\n=== Ruby/Rails questions: routing + retrieved answer ===", flush=True)
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
