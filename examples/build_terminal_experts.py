"""Build the Terminal specialist: bash + shell merged into ONE expert.

Bash e shell sao o mesmo dominio partido em dois. Separados eles disputam o mesmo
vocabulario e nenhum dos dois tem termo proprio suficiente: na validacao com 19
experts o bash ficou em 62% (5/8) e errou espalhado (dotnet, go, python), sinal de
falta de vocabulario distintivo e nao de rivalidade com um dominio vizinho.

Juntos viram 9 familias e ~54 licoes, com mais texto por expert e um eixo de
ambiguidade a menos. As familias nao colidem:
  bash  -> fundamentos, controle-de-fluxo, funcoes-e-dados, robustez
  shell -> navegacao-e-arquivos, inspecao-e-busca, pipes-e-redirecionamento,
           processamento-texto, processos-e-sistema

Le <origem>/<familia>/<subsetor>/*.md -> path = [familia, subsetor], varrendo as duas
pastas de dados. Substitui `build_bash_experts.py` + `build_shell_experts.py`: se usar
este, apague bash_experts/ e shell_experts/ do diretorio de experts, senao o
AxonSystem carrega os tres e volta a ambiguidade que a fusao resolve.

Env: AXON_TERMINAL_DIR (raiz que contem treinamento_bash/ e treinamento_shell/),
     AXON_EPOCHS (300), AXON_BATCH (256), AXON_LSA_DIM (200).
"""

import glob
import os

import pyaxon as ax
from pyaxon import manifest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_LOCAL = ROOT
_DATA = os.path.join(ROOT, "..", "treinamento")
_TEM_LOCAL = os.path.isdir(os.path.join(_LOCAL, "treinamento_bash"))
BASE_DIR = os.environ.get("AXON_TERMINAL_DIR", _LOCAL if _TEM_LOCAL else _DATA)
ORIGENS = ["treinamento_bash", "treinamento_shell"]
OUT = os.path.join(HERE, "axon_lang_data", "terminal_experts")
R_PREFIX = os.path.join(OUT, "router")
KB_PATH = os.path.join(OUT, "kb.sparse.json.gz")

EPOCHS = int(os.environ.get("AXON_EPOCHS", 300))
BATCH = int(os.environ.get("AXON_BATCH", 256))
LSA_DIM = int(os.environ.get("AXON_LSA_DIM", 200))

QUESTIONS = [
    # --- vindas do bash
    ('fundamentos', 'o que e o shebang e como usar variaveis e aspas no bash?'),
    ('fundamentos', 'como receber parametros posicionais e ler entrada com read?'),
    ('controle-de-fluxo', 'como usar if test e a comparacao com colchetes duplos?'),
    ('controle-de-fluxo', 'como fazer loops for while e until no bash?'),
    ('funcoes-e-dados', 'como declarar funcoes e usar arrays no bash?'),
    ('funcoes-e-dados', 'como fazer expansao de parametros e aritmetica?'),
    ('robustez', 'como usar set -e pipefail e tratar erros?'),
    ('robustez', 'como usar trap para limpeza e getopts para argumentos?'),
    # --- vindas do shell
    ('navegacao-e-arquivos', 'como navegar com ls cd pwd e caminhos?'),
    ('navegacao-e-arquivos', 'como ajustar permissoes com chmod e chown?'),
    ('inspecao-e-busca', 'como ver arquivos com cat less head e tail?'),
    ('inspecao-e-busca', 'como encontrar arquivos com o find?'),
    ('pipes-e-redirecionamento', 'como usar pipes e redirecionamento de stdout e stderr?'),
    ('processamento-texto', 'como buscar texto com grep e expressoes regulares?'),
    ('processamento-texto', 'como transformar texto com sed e awk?'),
    ('processos-e-sistema', 'como gerenciar processos com ps top e kill?'),
    ('processos-e-sistema', 'como agendar tarefas com cron e acessar por ssh?'),
]


def read_docs():
    docs = []
    for origem in ORIGENS:
        raiz = os.path.join(BASE_DIR, origem)
        if not os.path.isdir(raiz):
            print("aviso: nao encontrei " + raiz, flush=True)
            continue
        achados = 0
        for fp in glob.glob(os.path.join(raiz, "**", "*.md"), recursive=True):
            rel = os.path.relpath(fp, raiz).replace(chr(92), "/")
            parts = rel.split("/")[:-1]
            if not parts:
                continue
            with open(fp, encoding="utf-8") as f:
                text = f.read()
            if len(text) >= 200:
                docs.append((parts, text))
                achados += 1
        print(origem + ": " + str(achados) + " licoes", flush=True)
    return docs


def main():
    os.makedirs(OUT, exist_ok=True)
    docs = read_docs()
    if not docs:
        print("sem dados em " + BASE_DIR, flush=True)
        return
    per_family = {}
    for parts, _ in docs:
        per_family[parts[0]] = per_family.get(parts[0], 0) + 1
    print("terminal: " + str(len(docs)) + " lessons | families/experts: "
          + str(per_family), flush=True)

    router = ax.modular.ModularRouter(epochs=EPOCHS, batch_size=BATCH)
    kb = ax.vindex.SparseKB(ngram=1)
    for parts, text in docs:
        router.add(text, parts)
        kb.add_document(text, parts)
    print("training " + str(len(per_family)) + " family experts (mini-batch)...",
          flush=True)
    router.fit(dirty_only=False)
    print("building semantic index (LSA dim=" + str(LSA_DIM) + ")...", flush=True)
    kb.build(dim=LSA_DIM)

    router.save(R_PREFIX)
    kb.save(KB_PATH)
    manifest.write_manifest(R_PREFIX, model="ModularRouter/terminal",
                            counts={"experts": sorted(per_family), "lessons": len(docs)})
    print("saved: " + R_PREFIX + ".* | " + KB_PATH + " | passages=" + str(len(kb.texts)),
          flush=True)

    ok = 0
    print("\n=== Terminal questions: routing + retrieved answer ===", flush=True)
    for fam, q in QUESTIONS:
        pr = router.route(q)
        hit = pr[:1] == [fam]
        ok += hit
        passages = kb.retrieve(q, path_prefix=pr, top_k=1)
        snip = passages[0][0][:150].replace(chr(10), " ") if passages else "(nada)"
        tag = "OK" if hit else "X "
        print("  [" + tag + "] " + " > ".join(pr).ljust(26) + " | " + snip + " ...",
              flush=True)
    print("\nFAMILY accuracy: %d/%d = %.0f%%" % (ok, len(QUESTIONS),
                                                 100.0 * ok / len(QUESTIONS)), flush=True)


if __name__ == "__main__":
    main()
