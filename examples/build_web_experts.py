"""Build the Web (fase 12) specialist: one expert per FAMILY.

Front-end de marcação e estilo: HTML, CSS, layout responsivo, e frameworks de
UI (Bootstrap, Angular Material, Tailwind, Material UI, Ant Design, Vuetify,
PrimeNG), Sass e animações. Famílias desenhadas para vocabulário DISTINTO
(lição aprendida: evitar grab-bags/transversais):
html, css-fundamentos, layout-responsivo, bootstrap, angular-material,
tailwind-e-frameworks, sass-scss, animacoes-e-efeitos, material-ui, ant-design,
vuetify, primeng.

Reads <family>/*.md -> path = [family]. Data lives in treinamento_web/ (local
repo, ou o repo de dados irmao ../treinamento/treinamento_web/).
Saves the modular router + a sparse semantic KB to axon_lang_data/web_experts/.

Env: AXON_WEB_DIR (data dir), AXON_EPOCHS (300), AXON_BATCH (256), AXON_LSA_DIM (200).
"""

import glob
import os

import pyaxon as ax
from pyaxon import manifest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_LOCAL = os.path.join(ROOT, "treinamento_web")
_DATA = os.path.join(ROOT, "..", "treinamento", "treinamento_web")
DATA_DIR = os.environ.get("AXON_WEB_DIR", _LOCAL if os.path.isdir(_LOCAL) else _DATA)
OUT = os.path.join(HERE, "axon_lang_data", "web_experts")
R_PREFIX = os.path.join(OUT, "router")
KB_PATH = os.path.join(OUT, "kb.sparse.json.gz")

EPOCHS = int(os.environ.get("AXON_EPOCHS", 300))
BATCH = int(os.environ.get("AXON_BATCH", 256))
LSA_DIM = int(os.environ.get("AXON_LSA_DIM", 200))

QUESTIONS = [
    ('html', 'como estruturar um documento html e usar tags semanticas?'),
    ('html', 'como criar formularios e garantir acessibilidade com aria?'),
    ('css-fundamentos', 'como funcionam os seletores css e a especificidade?'),
    ('css-fundamentos', 'o que e o box model e as unidades rem e em?'),
    ('layout-responsivo', 'como usar flexbox e css grid para o layout?'),
    ('layout-responsivo', 'como fazer media queries e design responsivo mobile-first?'),
    ('bootstrap', 'como usar o sistema de grid e os componentes do bootstrap?'),
    ('angular-material', 'como usar os componentes mat-button e mat-card do angular material?'),
    ('tailwind-e-frameworks', 'como estilizar com as classes utilitarias do tailwind?'),
    ('sass-scss', 'como usar variaveis mixins e aninhamento no sass scss?'),
    ('animacoes-e-efeitos', 'como criar transicoes e animacoes com keyframes no css?'),
    ('material-ui', 'como usar o prop sx e o ThemeProvider dos componentes mui em react?'),
    ('ant-design', 'como usar a table e o form do ant design antd com configprovider?'),
    ('vuetify', 'como usar os componentes v-btn v-card e v-data-table do vuetify em vue?'),
    ('primeng', 'como usar o p-button e a p-table do primeng em angular?'),
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
    print("web: %d lessons | families: %s" % (len(docs), per_family), flush=True)

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
    manifest.write_manifest(R_PREFIX, model="ModularRouter/web",
                            counts={"experts": sorted(per_family), "lessons": len(docs)})
    print("saved: " + R_PREFIX + ".* | " + KB_PATH + " | passages=" + str(len(kb.texts)), flush=True)

    ok = 0
    print("\n=== Web questions: routing + retrieved answer ===", flush=True)
    for fam, q in QUESTIONS:
        pr = router.route(q)
        hit = pr[:1] == [fam]
        ok += hit
        passages = kb.retrieve(q, path_prefix=pr, top_k=1)
        snip = passages[0][0][:150].replace(chr(10), " ") if passages else "(nada)"
        tag = "OK" if hit else "X "
        print("  [" + tag + "] " + " > ".join(pr).ljust(26) + " | " + snip + " ...", flush=True)
    print("\nFAMILY accuracy: %d/%d = %.0f%%" % (ok, len(QUESTIONS), 100.0 * ok / len(QUESTIONS)), flush=True)


if __name__ == "__main__":
    main()
