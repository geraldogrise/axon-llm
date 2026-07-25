"""Build the AWS specialist: one expert per SUBSECTOR (all inside the aws domain).

Same modular design as the other specialists. Each subsector is an expert:
fundamentos, compute, storage, database, networking, seguranca-iam,
devops-monitoramento, mensageria-integracao.

Reads <subsector>/*.md -> path = [subsector]. Data lives in treinamento_aws/
(local in this repo). Saves the modular router + a sparse semantic KB to
axon_lang_data/aws_experts/.

Env: AXON_AWS_DIR (data dir), AXON_EPOCHS (300), AXON_BATCH (256), AXON_LSA_DIM (200).
"""

import glob
import os

import pyaxon as ax
from pyaxon import manifest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_LOCAL = os.path.join(ROOT, "treinamento_aws")
_DATA = os.path.join(ROOT, "..", "treinamento", "treinamento_aws")
AWS_DIR = os.environ.get("AXON_AWS_DIR", _LOCAL if os.path.isdir(_LOCAL) else _DATA)
OUT = os.path.join(HERE, "axon_lang_data", "aws_experts")
R_PREFIX = os.path.join(OUT, "router")
KB_PATH = os.path.join(OUT, "kb.sparse.json.gz")

EPOCHS = int(os.environ.get("AXON_EPOCHS", 300))
BATCH = int(os.environ.get("AXON_BATCH", 256))
LSA_DIM = int(os.environ.get("AXON_LSA_DIM", 200))

QUESTIONS = [
    ("fundamentos", "o que são regiões e zonas de disponibilidade na AWS?"),
    ("compute", "como criar uma instância EC2 e um Auto Scaling Group?"),
    ("compute", "como funciona o AWS Lambda e os triggers?"),
    ("storage", "como funciona o S3 e as classes de armazenamento?"),
    ("database", "qual a diferença entre RDS, Aurora e DynamoDB?"),
    ("networking", "como configurar uma VPC com subnets públicas e privadas?"),
    ("seguranca-iam", "como definir políticas IAM, roles e assume role?"),
    ("devops-monitoramento", "como usar CloudFormation e CloudWatch?"),
    ("mensageria-integracao", "qual a diferença entre SQS e SNS?"),
]


def read_docs():
    docs = []
    for fp in glob.glob(os.path.join(AWS_DIR, "**", "*.md"), recursive=True):
        parts = os.path.relpath(fp, AWS_DIR).replace("\\", "/").split("/")[:-1]
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
        print(f"sem dados em {AWS_DIR}", flush=True)
        return
    per_expert = {}
    for parts, _ in docs:
        per_expert[parts[0]] = per_expert.get(parts[0], 0) + 1
    print(f"aws: {len(docs)} lessons | experts (subsectors): {per_expert}", flush=True)

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
    manifest.write_manifest(R_PREFIX, model="ModularRouter/aws",
                            counts={"experts": sorted(per_expert), "lessons": len(docs)})
    print(f"saved: {R_PREFIX}.* | {KB_PATH} | passages={len(kb.texts)}", flush=True)

    ok = 0
    print("\n=== AWS questions: routing + retrieved answer ===", flush=True)
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
