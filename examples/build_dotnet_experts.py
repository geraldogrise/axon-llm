"""Build the .NET/C# specialist: one expert per framework FAMILY.

Same modular design as the Java specialist: each .NET family (core, web, orm, desktop,
testing, runtime) becomes an independent expert; the specific framework (entity-framework,
nhibernate, web-forms, ...) is a subsector inside its family. Semantic (hybrid) retrieval.

Reads treinamento_net/<family>/<subsector>/*.md  ->  path = [family, subsector].
Saves the modular router + a sparse semantic KB to axon_lang_data/dotnet_experts/.

Env: AXON_NET_DIR (data dir; default = sibling data repo), AXON_EPOCHS (default 300),
     AXON_BATCH (default 256), AXON_LSA_DIM (semantic dim, default 200).
"""

import glob
import os

import pyaxon as ax
from pyaxon import manifest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Prefer a freshly generated local folder; otherwise the sibling data repo.
_LOCAL = os.path.join(ROOT, "treinamento_net")
_DATA = os.path.join(ROOT, "..", "treinamento", "treinamento_net")
NET_DIR = os.environ.get("AXON_NET_DIR", _LOCAL if os.path.isdir(_LOCAL) else _DATA)
OUT = os.path.join(HERE, "axon_lang_data", "dotnet_experts")
R_PREFIX = os.path.join(OUT, "router")
KB_PATH = os.path.join(OUT, "kb.sparse.json.gz")

EPOCHS = int(os.environ.get("AXON_EPOCHS", 300))
BATCH = int(os.environ.get("AXON_BATCH", 256))
LSA_DIM = int(os.environ.get("AXON_LSA_DIM", 200))

QUESTIONS = [
    ("core", "como usar LINQ para filtrar uma lista com Where e Select?"),
    ("core", "o que é async e await na programação assíncrona em C#?"),
    ("core", "o que é herança e polimorfismo em C#?"),
    ("web", "como funciona o ciclo de vida de uma página no ASP.NET Web Forms?"),
    ("web", "o que é middleware no ASP.NET Core?"),
    ("web", "o que é um componente Blazor?"),
    ("orm", "como configurar o DbContext no Entity Framework Core?"),
    ("orm", "como mapear uma entidade com NHibernate e ISession?"),
    ("orm", "como usar o Dapper para executar uma query?"),
    ("desktop", "o que é data binding e o padrão MVVM no WPF?"),
    ("testing", "como criar um mock de uma interface com Moq?"),
    ("runtime", "como funciona o garbage collector e as gerações no .NET?"),
]


def read_net():
    docs = []
    for fp in glob.glob(os.path.join(NET_DIR, "**", "*.md"), recursive=True):
        parts = os.path.relpath(fp, NET_DIR).replace("\\", "/").split("/")[:-1]
        if not parts:
            continue
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        if len(text) >= 200:
            docs.append((parts, text))       # parts = [family, subsector]
    return docs


def main():
    os.makedirs(OUT, exist_ok=True)
    docs = read_net()
    if not docs:
        print(f"sem dados em {NET_DIR}", flush=True)
        return
    per_family = {}
    for parts, _ in docs:
        per_family[parts[0]] = per_family.get(parts[0], 0) + 1
    print(f"dotnet: {len(docs)} lessons | families/experts: {per_family}", flush=True)

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
    manifest.write_manifest(R_PREFIX, model="ModularRouter/dotnet",
                            counts={"experts": sorted(per_family), "lessons": len(docs)})
    print(f"saved: {R_PREFIX}.* | {KB_PATH} | passages={len(kb.texts)}", flush=True)

    ok = 0
    print("\n=== .NET questions: routing + retrieved answer ===", flush=True)
    for fam, q in QUESTIONS:
        pr = router.route(q)
        hit = pr[:1] == [fam]
        ok += hit
        passages = kb.retrieve(q, path_prefix=pr, top_k=1)
        snip = passages[0][0][:150].replace("\n", " ") if passages else "(nada)"
        print(f"  [{'OK' if hit else 'X '}] {' > '.join(pr):26} | {snip} ...", flush=True)
    print(f"\nFAMILY accuracy: {ok}/{len(QUESTIONS)} = {ok/len(QUESTIONS):.0%}", flush=True)


if __name__ == "__main__":
    main()
