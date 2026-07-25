"""axon-lang: builds the KNOWLEDGE BASE from MULTIPLE open sources.

Same idea as build_knowledge_base.py, but collects from three open, high-reputation
MediaWiki projects at once -- **Wikipedia + Wikilivros (Wikibooks) + Wikisource** --
all CC-BY-SA and already supported by the collector. Deep pagination per leaf, balanced
round-robin across areas, discard-after-train, resumable dedup. Trains the router AND
stores the passages (gzip + int8 -> kb.json.gz), then demonstrates answering.

This run **stops only when the data is truly exhausted** -- when every leaf's search
stops returning anything new across all three sources -- with a generous time ceiling as
a safety net (not the primary stop).

Note on other sources: SciELO's search and Domínio Público both return HTTP 403 to
automated clients (they block bots), and SciELO's working API (articlemeta) is a
per-journal metadata harvest, not a topic search -- so neither maps cleanly onto the
area/sector/subsector collection. They are intentionally NOT wired here. Reputable open
HTML/PDF pages you *are* allowed to fetch can still be dropped into SEED_URLS below
(read via corpus.fetch_url_text). See docs/DATA_SOURCES.md.

Env vars: AXON_BUDGET (s, safety ceiling, default 21600), AXON_DELAY (s between
          downloads, default 0.15), AXON_MIN_CHARS (default 350).
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
OUT = os.path.join(HERE, "axon_lang_data", "rag_multi")
R_PATH = os.path.join(OUT, "router.json")
KB_PATH = os.path.join(OUT, "kb.json.gz")            # compressed (gzip + int8)

TIME_BUDGET = int(os.environ.get("AXON_BUDGET", 21600))   # safety ceiling only (6 h)
DELAY = float(os.environ.get("AXON_DELAY", 0.15))
MIN_CHARS = int(os.environ.get("AXON_MIN_CHARS", 350))

# Open MediaWiki projects, in priority order (all CC-BY-SA).
HOSTS = ["pt.wikipedia.org", "pt.wikibooks.org", "pt.wikisource.org"]

# Optional non-MediaWiki seeds (HTML/PDF). Add (path, url) pairs of reputable open
# pages/PDFs here; they are read via corpus.fetch_url_text. Left empty by default
# because these sites have no uniform search API -- see docs/DATA_SOURCES.md.
SEED_URLS = [
    # (["portugues", "literatura", "autores"], "https://www.dominiopublico.gov.br/....pdf"),
]

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
    """Yield (path, host, title, url) candidates, paginating deep across all HOSTS.

    Stops this leaf only when a **full pagination round** (every host x query at the
    current offset) yields no new title -- i.e. the sources are genuinely exhausted --
    or at the hard offset ceiling. This is what makes 'run until the data runs out'
    honest, without grinding empty requests up to the ceiling."""
    offset = 0
    while offset < 10000:
        fresh = 0
        for host in HOSTS:
            for q in queries:
                try:
                    titles = corpus.mw_search(host, q, limit=20, offset=offset)
                except Exception:
                    titles = []
                for t in titles:
                    url = corpus.mw_url(host, t)
                    if url not in seen:
                        fresh += 1
                        yield (path, host, t, url)
        if fresh == 0:      # a whole round brought nothing new -> exhausted
            return
        offset += 20


def main():
    os.makedirs(OUT, exist_ok=True)
    router = ax.router.LinearRouter(max_features=4000, ngram=2, epochs=250, lr=0.05, balance=True)
    kb = ax.rag.KnowledgeBase(max_features=4000, ngram=1)
    seen = set()

    leaves = list(iter_leaves(TAXO))
    gens = {tuple(p): leaf_stream(p, q, seen) for p, q in leaves}
    active = set(gens)
    per_source = Counter()
    print(f"{len(leaves)} leaves x {len(HOSTS)} sources | collecting "
          f"(up to {TIME_BUDGET // 60} min)...", flush=True)

    t0 = time.time()
    n = 0
    per_area = Counter()

    # 1) optional non-MediaWiki seeds first (reputable HTML/PDF)
    for path, url in SEED_URLS:
        if url in seen:
            continue
        seen.add(url)
        try:
            txt = corpus.fetch_url_text(url)
        except Exception:
            continue
        if len(txt) >= MIN_CHARS:
            router.partial_fit(txt, path)
            kb.add_document(txt, path)
            per_area[path[0]] += 1
            per_source["seed"] += 1
            n += 1

    # 2) MediaWiki multi-source, round-robin across areas.
    #    Stops when `active` empties (every leaf exhausted) -- the real "data ran out"
    #    signal -- or at the TIME_BUDGET safety ceiling.
    while active and time.time() - t0 < TIME_BUDGET:
        for key in list(active):
            path = list(key)
            pulled = 0
            while pulled < 2:
                try:
                    _, host, title, url = next(gens[key])
                except StopIteration:
                    active.discard(key)
                    break
                if url in seen:
                    continue
                seen.add(url)
                time.sleep(DELAY)
                try:
                    txt = corpus.mw_extract(host, title)
                except Exception:
                    continue
                if len(txt) < MIN_CHARS:
                    continue
                router.partial_fit(txt, path)
                kb.add_document(txt, path)
                per_area[path[0]] += 1
                per_source[host] += 1
                n += 1
                pulled += 1
                if n % 100 == 0:
                    print(f"  {n} docs | {len(kb.texts)} passages | "
                          f"{(time.time() - t0) / 60:.0f} min | areas={dict(per_area)} | "
                          f"src={dict(per_source)}", flush=True)
                    router.save(R_PATH); kb.save(KB_PATH)

    print("training router and building the base index...", flush=True)
    router.fit()
    kb.build()
    router.save(R_PATH)
    kb.save(KB_PATH)
    print(f"\nDONE: {n} docs | {len(kb.texts)} passages | "
          f"kb.json.gz {os.path.getsize(KB_PATH) // 1024} KB | "
          f"{(time.time() - t0) / 60:.0f} min | sources={dict(per_source)}", flush=True)

    print("\n=== answering (route_multi corrects ambiguity) ===", flush=True)
    for q in ["qual o papel do DNA na hereditariedade?",
              "o que é o determinante de uma matriz?",
              "como a energia é transferida na termodinâmica?"]:
        paths, passages = kb.answer(q, router=router, top_k=1, multi=2)
        print(f"QUESTION: {q}", flush=True)
        print(f"  subsectors consulted: {[' > '.join(p) for p in paths]}", flush=True)
        if passages:
            print(f"  ANSWER: {passages[0][0][:300]} ...", flush=True)


if __name__ == "__main__":
    main()
