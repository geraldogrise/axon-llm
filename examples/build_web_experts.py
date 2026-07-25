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

# Uma pergunta por licao (75), nao uma por familia: com 1-2 perguntas por familia a
# acuracia impressa dizia muito pouco sobre as familias de framework, que concentram
# a maior parte do conteudo. Cada entrada abaixo espelha um arquivo de licao.
QUESTIONS = [
    # --- html
    ('html', 'como garantir acessibilidade com atributos aria e navegacao por teclado?'),
    ('html', 'como estruturar um documento html com doctype head e body?'),
    ('html', 'como criar formularios com input label e validacao no html?'),
    ('html', 'quando usar as tags semanticas header nav main section e article?'),
    ('html', 'como inserir links imagens video e audio numa pagina html?'),
    ('html', 'quais metatags usar para seo e o que o html5 moderno trouxe?'),
    ('html', 'como montar tabelas com thead tbody e listas ordenadas e nao ordenadas?'),
    # --- css-fundamentos
    ('css-fundamentos', 'o que e o box model e como funciona border-box e content-box?'),
    ('css-fundamentos', 'como funcionam a cascata a heranca e as variaveis css?'),
    ('css-fundamentos', 'como definir cores em hex rgb e hsl e trabalhar com background?'),
    ('css-fundamentos', 'para que servem pseudo-classes como hover e pseudo-elementos como before?'),
    ('css-fundamentos', 'como funcionam os seletores css e o calculo de especificidade?'),
    ('css-fundamentos', 'como controlar fonte tamanho peso e altura de linha na tipografia?'),
    ('css-fundamentos', 'qual a diferenca entre px rem em vh e porcentagem no css?'),
    # --- layout-responsivo
    ('layout-responsivo', 'como montar layouts com css grid usando grid-template e areas?'),
    ('layout-responsivo', 'como aplicar design responsivo com abordagem mobile-first?'),
    ('layout-responsivo', 'como alinhar e distribuir elementos com flexbox?'),
    ('layout-responsivo', 'como usar media queries e definir breakpoints?'),
    ('layout-responsivo', 'como controlar overflow scroll e dimensionamento de elementos?'),
    ('layout-responsivo', 'como montar padroes de layout comuns como sidebar e cards?'),
    ('layout-responsivo', 'como funcionam position relative absolute fixed sticky e o z-index?'),
    # --- bootstrap
    ('bootstrap', 'como usar os componentes de conteudo do bootstrap como card badge e alert?'),
    ('bootstrap', 'como montar navbar nav tabs e breadcrumb no bootstrap?'),
    ('bootstrap', 'como usar modal dropdown e collapse do javascript do bootstrap?'),
    ('bootstrap', 'como instalar o bootstrap via cdn ou npm e comecar a usar?'),
    ('bootstrap', 'como funciona o sistema de grid de 12 colunas do bootstrap?'),
    ('bootstrap', 'como usar as classes utilitarias e customizar o tema do bootstrap?'),
    # --- angular-material
    ('angular-material', 'como usar mat-button mat-icon e mat-progress-spinner?'),
    ('angular-material', 'como usar mat-form-field mat-input e mat-select em formularios?'),
    ('angular-material', 'como abrir um mat-dialog e configurar temas e overlays?'),
    ('angular-material', 'como instalar o angular material e configurar o material design?'),
    ('angular-material', 'como usar mat-toolbar mat-sidenav e mat-tabs para navegacao?'),
    ('angular-material', 'como exibir dados com mat-table paginator e sort?'),
    # --- tailwind-e-frameworks
    ('tailwind-e-frameworks', 'como escolher entre tailwind bulma e outros frameworks css?'),
    ('tailwind-e-frameworks', 'como aplicar cores tipografia e sombras com classes do tailwind?'),
    ('tailwind-e-frameworks', 'como customizar o tailwind config e extrair componentes com apply?'),
    ('tailwind-e-frameworks', 'o que e a abordagem utility-first do tailwind?'),
    ('tailwind-e-frameworks', 'como controlar layout margem e padding com as classes do tailwind?'),
    ('tailwind-e-frameworks', 'como usar os prefixos responsivos e o dark mode no tailwind?'),
    # --- sass-scss
    ('sass-scss', 'como aninhar seletores no scss e usar o ampersand?'),
    ('sass-scss', 'como usar if each for e maps no scss?'),
    ('sass-scss', 'como usar extend e placeholders para herdar estilos no sass?'),
    ('sass-scss', 'como declarar variaveis no scss e o que ele resolve?'),
    ('sass-scss', 'como criar mixins e funcoes no sass e usar include?'),
    ('sass-scss', 'como organizar o codigo com partials use e forward no sass?'),
    # --- animacoes-e-efeitos
    ('animacoes-e-efeitos', 'como criar animacoes com keyframes e controlar duracao e repeticao?'),
    ('animacoes-e-efeitos', 'como aplicar filtros blur e sombras box-shadow e text-shadow?'),
    ('animacoes-e-efeitos', 'como criar gradientes lineares e radiais e fundos visuais?'),
    ('animacoes-e-efeitos', 'como fazer micro-interacoes sem perder performance na animacao?'),
    ('animacoes-e-efeitos', 'como usar transform translate rotate scale e skew?'),
    ('animacoes-e-efeitos', 'como usar transition para suavizar mudancas de estado?'),
    # --- material-ui
    ('material-ui', 'como usar Button TextField e Typography do mui em react?'),
    ('material-ui', 'como usar o prop sx e o ThemeProvider para estilizar o mui?'),
    ('material-ui', 'como montar formularios com TextField Select e Checkbox no mui?'),
    ('material-ui', 'como instalar o material ui mui num projeto react?'),
    ('material-ui', 'como usar Box Container e Grid para layout no mui?'),
    ('material-ui', 'como usar AppBar Drawer Snackbar e Dialog no mui?'),
    # --- ant-design
    ('ant-design', 'como usar Button Layout e Row Col do ant design?'),
    ('ant-design', 'como usar Modal message notification e Drawer do antd?'),
    ('ant-design', 'como montar formularios com Form Item e validacao no ant design?'),
    ('ant-design', 'como instalar o ant design antd e configurar o ConfigProvider?'),
    ('ant-design', 'como usar Menu Breadcrumb e Tabs no ant design?'),
    ('ant-design', 'como usar a Table do antd com colunas paginacao e filtros?'),
    # --- vuetify
    ('vuetify', 'como usar v-btn v-card e v-icon no vuetify?'),
    ('vuetify', 'como usar v-data-table e configurar o tema do vuetify?'),
    ('vuetify', 'como montar formularios com v-text-field v-select e validacao no vuetify?'),
    ('vuetify', 'como instalar o vuetify num projeto vue?'),
    ('vuetify', 'como usar v-container v-row e v-col no grid do vuetify?'),
    ('vuetify', 'como usar v-app-bar v-navigation-drawer e v-tabs no vuetify?'),
    # --- primeng
    ('primeng', 'como usar p-button p-card e p-inputtext do primeng?'),
    ('primeng', 'como montar formularios com p-dropdown p-calendar e validacao no primeng?'),
    ('primeng', 'como instalar o primeng num projeto angular?'),
    ('primeng', 'como configurar tema e layout no primeng?'),
    ('primeng', 'como usar p-dialog p-menu e p-toast no primeng?'),
    ('primeng', 'como usar a p-table do primeng com paginacao e ordenacao?'),
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
