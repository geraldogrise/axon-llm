"""Build the Azure specialist: one expert per SUBSECTOR (all inside the azure domain).

Same modular design as the other specialists. Each subsector is an expert:
fundamentos, compute, storage, database, networking, identidade-seguranca,
devops-monitoramento, mensageria-integracao.

Reads <subsector>/*.md -> path = [subsector]. Data lives in treinamento_azure/
(local in this repo). Saves the modular router + a sparse semantic KB to
axon_lang_data/azure_experts/.

Env: AXON_AZURE_DIR (data dir), AXON_EPOCHS (300), AXON_BATCH (256), AXON_LSA_DIM (200).
"""

import glob
import os

import pyaxon as ax
from pyaxon import manifest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_LOCAL = os.path.join(ROOT, "treinamento_azure")
_DATA = os.path.join(ROOT, "..", "treinamento", "treinamento_azure")
AZURE_DIR = os.environ.get("AXON_AZURE_DIR", _LOCAL if os.path.isdir(_LOCAL) else _DATA)
OUT = os.path.join(HERE, "axon_lang_data", "azure_experts")
R_PREFIX = os.path.join(OUT, "router")
KB_PATH = os.path.join(OUT, "kb.sparse.json.gz")

EPOCHS = int(os.environ.get("AXON_EPOCHS", 300))
BATCH = int(os.environ.get("AXON_BATCH", 256))
LSA_DIM = int(os.environ.get("AXON_LSA_DIM", 200))

QUESTIONS = [
    ("fundamentos", "o que são resource groups e o Azure Resource Manager?"),
    ("compute", "como criar uma Virtual Machine e um scale set no Azure?"),
    ("compute", "como funciona o App Service e o Azure Functions?"),
    ("storage", "como funciona o Blob Storage e os tiers hot/cool/archive?"),
    ("database", "qual a diferença entre Azure SQL e Cosmos DB?"),
    ("networking", "como configurar uma VNet e Network Security Groups?"),
    ("identidade-seguranca", "o que é o Microsoft Entra ID e RBAC?"),
    ("devops-monitoramento", "como usar Bicep e o Azure Monitor?"),
    ("mensageria-integracao", "qual a diferença entre Service Bus e Event Grid?"),
]


def read_docs():
    docs = []
    for fp in glob.glob(os.path.join(AZURE_DIR, "**", "*.md"), recursive=True):
        parts = os.path.relpath(fp, AZURE_DIR).replace("\\", "/").split("/")[:-1]
        if not parts:
            continue
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        if len(text) >= 200:
            docs.append((parts, text))
    return docs


def main():
    os.makedirs(OUT, exist_ok=True)
    docs = read_docs()
    if not docs:
        print(f"sem dados em {AZURE_DIR}", flush=True)
        return
    per_expert = {}
    for parts, _ in docs:
        per_expert[parts[0]] = per_expert.get(parts[0], 0) + 1
    print(f"azure: {len(docs)} lessons | experts (subsectors): {per_expert}", flush=True)

    router = ax.modular.ModularRouter(epochs=EPOCHS, batch_size=BATCH)
    kb = ax.vindex.SparseKB(ngram=1)
    for parts, text in docs:
        router.add(text, parts)
        kb.add_document(text, parts)
    print(f"training {len(per_expert)} subsector experts (mini-batch)...", flush=True)
    router.fit(dirty_only=False)
    print(f"building semantic index (LSA dim={LSA_DIM})...", flush=True)
    kb.build(dim=LSA_DIM)

    router.save(R_PREFIX)
    kb.save(KB_PATH)
    manifest.write_manifest(R_PREFIX, model="ModularRouter/azure",
                            counts={"experts": sorted(per_expert), "lessons": len(docs)})
    print(f"saved: {R_PREFIX}.* | {KB_PATH} | passages={len(kb.texts)}", flush=True)

    ok = 0
    print("\n=== Azure questions: routing + retrieved answer ===", flush=True)
    for sub, q in QUESTIONS:
        pr = router.route(q)
        hit = pr[:1] == [sub]
        ok += hit
        passages = kb.retrieve(q, path_prefix=pr, top_k=1)
        snip = passages[0][0][:150].replace("\n", " ") if passages else "(nada)"
        print(f"  [{'OK' if hit else 'X '}] {' > '.join(pr):24} | {snip} ...", flush=True)
    print(f"\nSUBSECTOR accuracy: {ok}/{len(QUESTIONS)} = {ok/len(QUESTIONS):.0%}", flush=True)


if __name__ == "__main__":
    main()
