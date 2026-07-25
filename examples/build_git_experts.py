"""Build the Git specialist: one expert per FAMILY.

Same modular design as the other specialists (Java/.NET/JS/Python/PHP/Rust/Go/Ruby).
Reads <family>/<subsector>/*.md -> path = [family, subsector]. Data lives in
treinamento_git/ (local repo, or the sibling `treinamento` data repo).
Saves the modular router + a sparse semantic KB to axon_lang_data/git_experts/.

Env: AXON_GIT_DIR (data dir), AXON_EPOCHS (300), AXON_BATCH (256), AXON_LSA_DIM (200).
"""

import glob
import os

import pyaxon as ax
from pyaxon import manifest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_LOCAL = os.path.join(ROOT, "treinamento_git")
_DATA = os.path.join(ROOT, "..", "treinamento", "treinamento_git")
DATA_DIR = os.environ.get("AXON_GIT_DIR", _LOCAL if os.path.isdir(_LOCAL) else _DATA)
OUT = os.path.join(HERE, "axon_lang_data", "git_experts")
R_PREFIX = os.path.join(OUT, "router")
KB_PATH = os.path.join(OUT, "kb.sparse.json.gz")

EPOCHS = int(os.environ.get("AXON_EPOCHS", 300))
BATCH = int(os.environ.get("AXON_BATCH", 256))
LSA_DIM = int(os.environ.get("AXON_LSA_DIM", 200))

QUESTIONS = [
    ('fundamentos', 'o que e o git e como funciona o commit e a staging area?'),
    ('fundamentos', 'como configurar nome e email e o gitignore?'),
    ('branches-e-merge', 'como criar branches e resolver conflitos de merge?'),
    ('branches-e-merge', 'qual a diferenca entre merge e rebase?'),
    ('remotos-e-colaboracao', 'como fazer push e pull e configurar o origin?'),
    ('remotos-e-colaboracao', 'como funciona fork e pull request e as tags de release?'),
    ('desfazer-e-historia', 'como desfazer com reset revert e usar o stash?'),
    ('desfazer-e-historia', 'como recuperar commits perdidos com o reflog?'),
    ('avancado', 'como usar submodulos hooks e git bisect?'),
    ('avancado', 'como usar git subtree e o git bundle?'),
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
    print("git: %d lessons | families: %s" % (len(docs), per_family), flush=True)

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
    manifest.write_manifest(R_PREFIX, model="ModularRouter/git",
                            counts={"experts": sorted(per_family), "lessons": len(docs)})
    print("saved: " + R_PREFIX + ".* | " + KB_PATH + " | passages=" + str(len(kb.texts)), flush=True)

    ok = 0
    print("\n=== Git questions: routing + retrieved answer ===" % (), flush=True)
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
