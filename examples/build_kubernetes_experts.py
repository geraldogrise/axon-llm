"""Build the Kubernetes specialist: one expert per FAMILY.

Same modular design as the other specialists (Java/.NET/JS/Python/PHP/Rust/Go/Ruby).
Reads <family>/<subsector>/*.md -> path = [family, subsector]. Data lives in
treinamento_kubernetes/ (local repo, or the sibling `treinamento` data repo).
Saves the modular router + a sparse semantic KB to axon_lang_data/kubernetes_experts/.

Env: AXON_KUBERNETES_DIR (data dir), AXON_EPOCHS (300), AXON_BATCH (256), AXON_LSA_DIM (200).
"""

import glob
import os

import pyaxon as ax
from pyaxon import manifest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_LOCAL = os.path.join(ROOT, "treinamento_kubernetes")
_DATA = os.path.join(ROOT, "..", "treinamento", "treinamento_kubernetes")
DATA_DIR = os.environ.get("AXON_KUBERNETES_DIR", _LOCAL if os.path.isdir(_LOCAL) else _DATA)
OUT = os.path.join(HERE, "axon_lang_data", "kubernetes_experts")
R_PREFIX = os.path.join(OUT, "router")
KB_PATH = os.path.join(OUT, "kb.sparse.json.gz")

EPOCHS = int(os.environ.get("AXON_EPOCHS", 300))
BATCH = int(os.environ.get("AXON_BATCH", 256))
LSA_DIM = int(os.environ.get("AXON_LSA_DIM", 200))

QUESTIONS = [
    ('fundamentos', 'o que e o kubernetes e o modelo declarativo?'),
    ('fundamentos', 'como funciona a arquitetura do cluster control plane e nodes?'),
    ('workloads', 'como funcionam deployments e replicasets?'),
    ('workloads', 'como fazer rolling update e rollback no kubernetes?'),
    ('rede-servicos', 'como os services fazem descoberta e o que e o ingress?'),
    ('rede-servicos', 'como funcionam as network policies e o service mesh?'),
    ('config-storage', 'como usar configmaps secrets e persistent volumes?'),
    ('escalonamento', 'como usar requests limits probes e o autoscaler hpa?'),
    ('seguranca-rbac', 'como funciona o rbac e as service accounts?'),
    ('ecossistema-operacoes', 'como usar helm e fazer troubleshooting de pods?'),
]


def read_docs():
    docs = []
    for fp in glob.glob(os.path.join(DATA_DIR, "**", "*.md"), recursive=True):
        rel = os.path.relpath(fp, DATA_DIR).replace(chr(92), "/")
        parts = rel.split("/")[:-1]
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
        print("sem dados em " + DATA_DIR, flush=True)
        return
    per_family = {}
    for parts, _ in docs:
        per_family[parts[0]] = per_family.get(parts[0], 0) + 1
    print("kubernetes: %d lessons | families: %s" % (len(docs), per_family), flush=True)

    router = ax.modular.ModularRouter(epochs=EPOCHS, batch_size=BATCH)
    kb = ax.vindex.SparseKB(ngram=1)
    for parts, text in docs:
        router.add(text, parts)
        kb.add_document(text, parts)
    print("training %d family experts (mini-batch)..." % len(per_family), flush=True)
    router.fit(dirty_only=False)
    print("building semantic index (LSA dim=%d)..." % LSA_DIM, flush=True)
    kb.build(dim=LSA_DIM)

    router.save(R_PREFIX)
    kb.save(KB_PATH)
    manifest.write_manifest(R_PREFIX, model="ModularRouter/kubernetes",
                            counts={"experts": sorted(per_family), "lessons": len(docs)})
    print("saved: " + R_PREFIX + ".* | " + KB_PATH + " | passages=" + str(len(kb.texts)), flush=True)

    ok = 0
    print("\n=== Kubernetes questions: routing + retrieved answer ===" % (), flush=True)
    for fam, q in QUESTIONS:
        pr = router.route(q)
        hit = pr[:1] == [fam]
        ok += hit
        passages = kb.retrieve(q, path_prefix=pr, top_k=1)
        snip = passages[0][0][:150].replace(chr(10), " ") if passages else "(nada)"
        tag = "OK" if hit else "X "
        print("  [" + tag + "] " + " > ".join(pr).ljust(26) + " | " + snip + " ...", flush=True)
    print("\nFAMILY accuracy: %d/%d = %.0f%%" % (ok, len(QUESTIONS), 100.0*ok/len(QUESTIONS)), flush=True)


if __name__ == "__main__":
    main()
