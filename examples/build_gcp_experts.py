"""Build the GCP specialist: one expert per SUBSECTOR (all inside the gcp domain).

Same modular design as the other specialists. Each subsector is an expert:
fundamentos, compute, storage, database, networking, identidade-seguranca,
devops-monitoramento, data-mensageria.

Reads <subsector>/*.md -> path = [subsector]. Data lives in treinamento_gcp/
(local in this repo). Saves the modular router + a sparse semantic KB to
axon_lang_data/gcp_experts/.

Env: AXON_GCP_DIR (data dir), AXON_EPOCHS (300), AXON_BATCH (256), AXON_LSA_DIM (200).
"""

import glob
import os

import pyaxon as ax
from pyaxon import manifest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_LOCAL = os.path.join(ROOT, "treinamento_gcp")
_DATA = os.path.join(ROOT, "..", "treinamento", "treinamento_gcp")
GCP_DIR = os.environ.get("AXON_GCP_DIR", _LOCAL if os.path.isdir(_LOCAL) else _DATA)
OUT = os.path.join(HERE, "axon_lang_data", "gcp_experts")
R_PREFIX = os.path.join(OUT, "router")
KB_PATH = os.path.join(OUT, "kb.sparse.json.gz")

EPOCHS = int(os.environ.get("AXON_EPOCHS", 300))
BATCH = int(os.environ.get("AXON_BATCH", 256))
LSA_DIM = int(os.environ.get("AXON_LSA_DIM", 200))

QUESTIONS = [
    ("fundamentos", "o que são projetos e a resource hierarchy no GCP?"),
    ("compute", "como usar o Compute Engine e o Cloud Run?"),
    ("compute", "como funciona o GKE e o GKE Autopilot?"),
    ("storage", "como funciona o Cloud Storage e as classes de armazenamento?"),
    ("database", "como consultar dados no BigQuery?"),
    ("database", "qual a diferença entre Cloud SQL, Spanner e Firestore?"),
    ("networking", "como configurar uma VPC e firewall rules no GCP?"),
    ("identidade-seguranca", "como funcionam service accounts e IAM no Google Cloud?"),
    ("devops-monitoramento", "como usar Cloud Build e Cloud Monitoring?"),
    ("data-mensageria", "como usar o Pub/Sub e o Dataflow?"),
]


def read_docs():
    docs = []
    for fp in glob.glob(os.path.join(GCP_DIR, "**", "*.md"), recursive=True):
        parts = os.path.relpath(fp, GCP_DIR).replace("\\", "/").split("/")[:-1]
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
        print(f"sem dados em {GCP_DIR}", flush=True)
        return
    per_expert = {}
    for parts, _ in docs:
        per_expert[parts[0]] = per_expert.get(parts[0], 0) + 1
    print(f"gcp: {len(docs)} lessons | experts (subsectors): {per_expert}", flush=True)

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
    manifest.write_manifest(R_PREFIX, model="ModularRouter/gcp",
                            counts={"experts": sorted(per_expert), "lessons": len(docs)})
    print(f"saved: {R_PREFIX}.* | {KB_PATH} | passages={len(kb.texts)}", flush=True)

    ok = 0
    print("\n=== GCP questions: routing + retrieved answer ===", flush=True)
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
