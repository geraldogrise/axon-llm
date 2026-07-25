"""axon-lang: builds the large KNOWLEDGE BASE (full hierarchy).

Collects from Wikipedia (deep pagination, balanced per leaf), TRAINS the router
and STORES the passages in the KnowledgeBase (kb.json.gz -- gzip + int8-quantized
index). Saves periodically. Then demonstrates answering with route_multi (searches
2 subsectors when ambiguous).
"""

import os
import sys
import time
from collections import Counter

import pyaxon as ax
from pyaxon import corpus

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "axon_lang_data", "rag")
R_PATH = os.path.join(OUT, "router.json")
KB_PATH = os.path.join(OUT, "kb.json.gz")   # compressed (gzip + int8)

TIME_BUDGET = int(os.environ.get("AXON_BUDGET", 1800))   # default 30 min
PER_LEAF = int(os.environ.get("AXON_PER_LEAF", 12))
DELAY = float(os.environ.get("AXON_DELAY", 0.1))
HOST = "pt.wikipedia.org"

TAXO = {
    "matematica": {
        "algebra": {"linear": ["álgebra linear", "matriz matemática", "determinante", "autovalor"],
                    "abstrata": ["álgebra abstrata", "teoria dos grupos", "anel matemática"]},
        "calculo": {"derivadas": ["cálculo diferencial", "derivada", "limite matemática"],
                    "integrais": ["cálculo integral", "integral", "teorema fundamental do cálculo"]},
        "geometria": {"plana": ["geometria plana", "triângulo", "circunferência"],
                      "analitica": ["geometria analítica", "plano cartesiano", "equação da reta"]},
    },
    "fisica": {
        "mecanica": {"classica": ["mecânica clássica", "leis de Newton", "energia cinética"],
                     "quantica": ["mecânica quântica", "função de onda", "átomo de Bohr"]},
        "termo": {"calor": ["termodinâmica", "calor", "temperatura"],
                  "entropia": ["entropia", "segunda lei da termodinâmica"]},
        "eletro": {"eletricidade": ["eletricidade", "corrente elétrica", "circuito elétrico"],
                   "magnetismo": ["magnetismo", "campo magnético", "indução eletromagnética"]},
    },
    "biologia": {
        "genetica": {"mendel": ["genética", "leis de Mendel", "hereditariedade"],
                     "molecular": ["DNA", "RNA", "código genético", "gene"]},
        "citologia": {"celula": ["célula", "membrana plasmática", "organela"],
                      "energia": ["fotossíntese", "respiração celular", "metabolismo"]},
        "ecologia": {"ecossistemas": ["ecologia", "ecossistema", "cadeia alimentar"],
                     "biomas": ["bioma", "floresta amazônica", "biodiversidade"]},
    },
    "quimica": {
        "geral": {"atomo": ["átomo", "modelo atômico", "tabela periódica"],
                  "ligacoes": ["ligação química", "ligação covalente", "ligação iônica"]},
        "organica": {"hidrocarbonetos": ["química orgânica", "hidrocarboneto", "alcano"],
                     "funcoes": ["função orgânica", "álcool química", "ácido carboxílico"]},
    },
    "portugues": {
        "gramatica": {"sintaxe": ["sintaxe", "análise sintática", "concordância verbal"],
                      "morfologia": ["morfologia linguística", "substantivo", "verbo conjugação"]},
        "literatura": {"autores": ["Machado de Assis", "José de Alencar", "Clarice Lispector"],
                       "movimentos": ["modernismo no Brasil", "romantismo", "realismo literário"]},
    },
    "historia": {
        "brasil": {"colonia": ["brasil colônia", "capitanias hereditárias", "ciclo do ouro"],
                   "republica": ["era Vargas", "ditadura militar no Brasil"]},
        "geral": {"antiga": ["império romano", "grécia antiga"],
                  "moderna": ["revolução francesa", "revolução industrial"]},
    },
}


def iter_leaves(node, prefix=None):
    prefix = prefix or []
    for k, v in node.items():
        p = prefix + [k]
        if isinstance(v, dict):
            yield from iter_leaves(v, p)
        else:
            yield (p, v)


def leaf_stream(path, queries, seen):
    offset = 0
    while offset < 5000:
        for q in queries:
            try:
                titles = corpus.mw_search(HOST, q, limit=30, offset=offset)
            except Exception:
                titles = []
            for t in titles:
                url = corpus.mw_url(HOST, t)
                if url not in seen:
                    yield (path, t, url)
        offset += 30


def main():
    os.makedirs(OUT, exist_ok=True)
    router = ax.router.LinearRouter(max_features=4000, ngram=2, epochs=250, lr=0.05, balance=True)
    kb = ax.rag.KnowledgeBase(max_features=4000, ngram=1)
    seen = set()

    leaves = list(iter_leaves(TAXO))
    gens = {tuple(p): leaf_stream(p, q, seen) for p, q in leaves}
    active = set(gens)
    print(f"{len(leaves)} leaves | collecting knowledge (up to {TIME_BUDGET//60} min)...", flush=True)

    t0 = time.time()
    n = 0
    per_area = Counter()
    target = PER_LEAF * len(leaves)
    while time.time() - t0 < TIME_BUDGET and active and n < target:
        for key in list(active):
            path = list(key)
            pulled = 0
            while pulled < 2:
                try:
                    _, title, url = next(gens[key])
                except StopIteration:
                    active.discard(key)
                    break
                if url in seen:
                    continue
                seen.add(url)
                time.sleep(DELAY)
                try:
                    txt = corpus.wikipedia_extract(title)
                except Exception:
                    continue
                if len(txt) < 400:
                    continue
                router.partial_fit(txt, path)
                kb.add_document(txt, path)
                per_area[path[0]] += 1
                n += 1
                pulled += 1
                if n % 100 == 0:
                    print(f"  {n} articles | {len(kb.texts)} passages | "
                          f"{(time.time()-t0)/60:.0f} min | {dict(per_area)}", flush=True)
                    router.save(R_PATH); kb.save(KB_PATH)   # save progress

    print("training router and building the base index...", flush=True)
    router.fit()
    kb.build()
    router.save(R_PATH)
    kb.save(KB_PATH)
    print(f"\nDONE: {n} articles | {len(kb.texts)} passages | "
          f"kb.json.gz {os.path.getsize(KB_PATH)//1024} KB | {(time.time()-t0)/60:.0f} min", flush=True)

    print("\n=== answering (route_multi corrects ambiguity) ===", flush=True)
    for q in ["qual o papel do DNA na hereditariedade?",
              "o que é o determinante de uma matriz?",
              "como a energia é transferida na termodinâmica?"]:
        paths, passages = kb.answer(q, router=router, top_k=1, multi=2)
        print("QUESTION:", q)
        print("  subsectors consulted:", [" > ".join(c) for c in paths])
        if passages:
            print("  ANSWER:", passages[0][0][:280].replace("\n", " "), "...")
        print(flush=True)


if __name__ == "__main__":
    main()
