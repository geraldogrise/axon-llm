"""Build the Java specialist: one expert per framework FAMILY.

Each Java family (spring, core, orm, jakarta, testing, build, frameworks, versoes,
jvm, web) becomes an independent expert in a ModularRouter: adding Spring topics
retrains only the `spring` expert, never `core` or `orm`. The specific framework
(spring-boot, hibernate...) is a subsector inside its family expert.

Reads treinamento_programacao/java/<family>/<framework>/*.md  ->  path = [family, framework].
Saves the router + a sparse KB to axon_lang_data/java_experts/ and reports routing.

Env: AXON_EPOCHS (default 200), AXON_BATCH (mini-batch, default 256).
"""

import glob
import os

import pyaxon as ax
from pyaxon import manifest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Point at the data repo (sibling). Override with AXON_JAVA_DIR. Needs `git checkout fase-2`.
JAVA = os.environ.get("AXON_JAVA_DIR",
                      os.path.join(ROOT, "..", "treinamento", "treinamento_programacao", "java"))
OUT = os.path.join(HERE, "axon_lang_data", "java_experts")
R_PREFIX = os.path.join(OUT, "router")
KB_PATH = os.path.join(OUT, "kb.sparse.json.gz")

EPOCHS = int(os.environ.get("AXON_EPOCHS", 300))
BATCH = int(os.environ.get("AXON_BATCH", 256))
LSA_DIM = int(os.environ.get("AXON_LSA_DIM", 200))   # semantic (hybrid) retrieval

QUESTIONS = [
    ("spring", "o que é injeção de dependência no Spring Boot?"),
    ("spring", "como configurar autenticação com Spring Security e JWT?"),
    ("core", "como usar a Stream API com lambdas e collectors?"),
    ("core", "o que é herança e polimorfismo em Java?"),
    ("orm", "como mapear uma entidade com Hibernate e JPA?"),
    ("jakarta", "o que é um Servlet e o método doGet?"),
    ("testing", "como criar um mock com Mockito no JUnit?"),
    ("build", "para que serve o pom.xml no Maven?"),
    ("versoes", "o que são virtual threads no Java 21?"),
    ("jvm", "como funciona o garbage collector da JVM?"),
    ("frameworks", "o que é o Quarkus e a compilação nativa com GraalVM?"),
    ("web", "para que serve o PrimeFaces com JSF?"),
]


def read_java():
    docs = []
    for fp in glob.glob(os.path.join(JAVA, "**", "*.md"), recursive=True):
        parts = os.path.relpath(fp, JAVA).replace("\\", "/").split("/")[:-1]
        if not parts:
            continue
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        if len(text) >= 200:
            docs.append((parts, text))       # parts = [family, framework]
    return docs


def main():
    os.makedirs(OUT, exist_ok=True)
    docs = read_java()
    per_family = {}
    for parts, _ in docs:
        per_family[parts[0]] = per_family.get(parts[0], 0) + 1
    print(f"java: {len(docs)} lessons | families/experts: {per_family}", flush=True)

    # each family (path[0]) becomes an independent expert in the modular router
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
    manifest.write_manifest(R_PREFIX, model="ModularRouter/java",
                            counts={"experts": sorted(per_family), "lessons": len(docs)})
    print(f"saved: {R_PREFIX}.* | {KB_PATH} | passages={len(kb.texts)}", flush=True)

    ok = 0
    print("\n=== Java questions: routing + retrieved answer ===", flush=True)
    for fam, q in QUESTIONS:
        pr = router.route(q)
        hit = pr[:1] == [fam]
        ok += hit
        passages = kb.retrieve(q, path_prefix=pr, top_k=1)
        snip = passages[0][0][:150].replace("\n", " ") if passages else "(nada)"
        print(f"  [{'OK' if hit else 'X '}] {' > '.join(pr):28} | {snip} ...", flush=True)
    print(f"\nFAMILY accuracy: {ok}/{len(QUESTIONS)} = {ok/len(QUESTIONS):.0%}", flush=True)


if __name__ == "__main__":
    main()
