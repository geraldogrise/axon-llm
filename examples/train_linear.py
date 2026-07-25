"""axon-lang: training of the FULL HIERARCHY (area -> sector -> subsector).

Stable version (keeps bags, balances, low lr, deep pagination), now with
the full hierarchy for the FINE LEVELS. Measures accuracy at each level.

Two CSVs: downloaded.csv (dedup) and trained.csv (what was trained).
"""

import os
import time
from collections import Counter

import pyaxon as ax
from pyaxon import collect, corpus

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "axon_lang_data", "linear")
CKPT = os.path.join(OUT, "router.json")

TIME_BUDGET = int(os.environ.get("AXON_BUDGET", 3 * 3600))
DELAY = float(os.environ.get("AXON_DELAY", 0.12))
CHUNK = int(os.environ.get("AXON_CHUNK", 2))        # sources per LEAF per round
FIT_EVERY = int(os.environ.get("AXON_FIT_EVERY", 400))
VAL_PER_LEAF = int(os.environ.get("AXON_VAL", 2))
HOST = "pt.wikipedia.org"

# Hierarchy: area -> sector -> subsector -> [Wikipedia searches]
TAXO = {
    "portugues": {
        "gramatica": {
            "morfologia": ["morfologia linguística", "substantivo", "verbo conjugação",
                           "classes de palavras"],
            "sintaxe": ["sintaxe", "análise sintática", "concordância verbal", "oração"],
            "ortografia": ["ortografia da língua portuguesa", "acentuação gráfica", "pontuação"],
        },
        "literatura": {
            "autores": ["Machado de Assis", "José de Alencar", "Clarice Lispector"],
            "movimentos": ["modernismo no Brasil", "romantismo", "realismo literário"],
            "generos": ["poesia", "romance literário", "conto"],
        },
    },
    "matematica": {
        "algebra": {
            "linear": ["álgebra linear", "matriz matemática", "determinante", "espaço vetorial"],
            "abstrata": ["álgebra abstrata", "teoria dos grupos", "anel matemática"],
        },
        "geometria": {
            "plana": ["geometria plana", "triângulo", "circunferência"],
            "analitica": ["geometria analítica", "plano cartesiano", "equação da reta"],
        },
        "calculo": {
            "derivadas": ["cálculo diferencial", "derivada", "limite matemática"],
            "integrais": ["cálculo integral", "integral", "teorema fundamental do cálculo"],
        },
        "numeros": {
            "teoria": ["teoria dos números", "número primo", "divisibilidade"],
            "estatistica": ["estatística", "probabilidade", "média aritmética"],
        },
    },
    "fisica": {
        "mecanica": {
            "classica": ["mecânica clássica", "leis de Newton", "energia cinética"],
            "quantica": ["mecânica quântica", "função de onda", "princípio da incerteza"],
        },
        "termodinamica": {
            "calor": ["termodinâmica", "calor", "temperatura"],
            "entropia": ["entropia", "segunda lei da termodinâmica"],
        },
        "eletromagnetismo": {
            "eletricidade": ["eletricidade", "corrente elétrica", "circuito elétrico"],
            "magnetismo": ["magnetismo", "campo magnético", "indução eletromagnética"],
        },
    },
    "quimica": {
        "geral": {
            "atomo": ["átomo", "modelo atômico", "tabela periódica"],
            "ligacoes": ["ligação química", "ligação covalente", "ligação iônica"],
        },
        "organica": {
            "hidrocarbonetos": ["química orgânica", "hidrocarboneto", "alcano"],
            "funcoes": ["função orgânica", "álcool química", "ácido carboxílico"],
        },
        "fisicoquimica": {
            "reacoes": ["reação química", "cinética química", "equilíbrio químico"],
            "solucoes": ["solução química", "concentração química", "potencial hidrogeniônico"],
        },
    },
    "biologia": {
        "citologia": {
            "celula": ["célula", "membrana plasmática", "organela"],
            "metabolismo": ["metabolismo", "fotossíntese", "respiração celular"],
        },
        "genetica": {
            "hereditariedade": ["genética", "leis de Mendel", "hereditariedade"],
            "molecular": ["DNA", "RNA", "código genético"],
        },
        "ecologia": {
            "ecossistemas": ["ecologia", "ecossistema", "cadeia alimentar"],
            "biomas": ["bioma", "floresta amazônica", "biodiversidade"],
        },
    },
    "historia": {
        "brasil": {
            "colonia": ["brasil colônia", "capitanias hereditárias", "ciclo do ouro"],
            "republica": ["era Vargas", "ditadura militar no Brasil", "república velha"],
        },
        "geral": {
            "antiga": ["idade antiga", "império romano", "grécia antiga"],
            "moderna": ["revolução francesa", "revolução industrial", "idade moderna"],
        },
    },
    "filosofia": {
        "classica": {
            "grega": ["filosofia grega", "Sócrates", "Aristóteles"],
            "moderna": ["filosofia moderna", "René Descartes", "Immanuel Kant"],
        },
        "areas": {
            "logica": ["lógica", "silogismo", "raciocínio lógico"],
            "etica": ["ética", "filosofia política"],
        },
    },
    "geografia": {
        "fisica": {
            "clima": ["climatologia", "aquecimento global", "clima"],
            "relevo": ["relevo", "geomorfologia", "hidrografia"],
        },
        "humana": {
            "populacao": ["demografia", "urbanização", "população mundial"],
            "economica": ["geografia econômica", "globalização", "agricultura"],
        },
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
    """(Almost) infinite stream of (path, url) from a leaf, paginating deep."""
    offset = 0
    while offset < 9000:
        for q in queries:
            try:
                titles = corpus.mw_search(HOST, q, limit=40, offset=offset)
            except Exception:
                titles = []
            for t in titles:
                url = corpus.mw_url(HOST, t)
                if url not in seen:
                    yield (path, url)
        offset += 40


def build_validation(leaves, seen):
    val = []
    for path, queries in leaves:
        picked = 0
        for q in queries:
            if picked >= VAL_PER_LEAF:
                break
            try:
                titles = corpus.mw_search(HOST, q, limit=8)
            except Exception:
                continue
            for t in titles:
                if picked >= VAL_PER_LEAF:
                    break
                url = corpus.mw_url(HOST, t)
                if url in seen:
                    continue
                seen.add(url)
                try:
                    txt = corpus.wikipedia_extract(t)
                except Exception:
                    continue
                if len(txt) >= 400:
                    val.append((txt, path))
                    picked += 1
    return val


def evaluate(router, val):
    """Accuracy per level (area, sector, subsector)."""
    depth = max(len(p) for _, p in val)
    hits = [0] * depth
    for txt, path in val:
        pred = router.route(txt)
        for i in range(min(len(pred), len(path))):
            if pred[i] == path[i]:
                hits[i] += 1
    return [h / len(val) for h in hits]


def main():
    os.makedirs(OUT, exist_ok=True)
    downloaded = collect.Collector(OUT, "downloaded.csv")
    trained = collect.Collector(OUT, "trained.csv")
    router = ax.router.LinearRouter(max_features=4000, ngram=2, epochs=300, lr=0.05,
                                    weight_decay=1e-4, balance=True)

    leaves = list(iter_leaves(TAXO))
    print(f"{len(leaves)} leaves (subsectors) in the hierarchy", flush=True)
    print("building per-leaf validation (untrained)...", flush=True)
    val = build_validation(leaves, downloaded.seen_urls)
    print(f"validation: {len(val)} articles", flush=True)

    gens = {tuple(p): leaf_stream(p, q, downloaded.seen_urls) for p, q in leaves}
    active = set(gens)
    print(f"per-leaf stream started | running for up to ~3h\n", flush=True)

    t0 = time.time()
    ingested = 0
    next_fit = FIT_EVERY
    per_area = Counter()
    while time.time() - t0 < TIME_BUDGET and active:
        for key in list(active):
            path = list(key)
            pulled, attempts = 0, 0
            while pulled < CHUNK and attempts < CHUNK * 8:
                attempts += 1
                try:
                    _, url = next(gens[key])
                except StopIteration:
                    active.discard(key)
                    break
                if url in downloaded.seen_urls:
                    continue
                downloaded.seen_urls.add(url)
                time.sleep(DELAY)
                try:
                    text = downloaded._fetch_text(url)
                except Exception:
                    continue
                if not text or len(text) < 300:
                    continue
                h = corpus._text_hash(text)
                if h in downloaded.seen_hashes:
                    continue
                downloaded.seen_hashes.add(h)
                title = url.rstrip("/").split("/")[-1].replace("_", " ")[:120]
                row = {"id": downloaded._next_id, "source": HOST, "url": url, "title": title,
                       "area": path[0], "sector": path[1] if len(path) > 1 else "",
                       "subsector": path[2] if len(path) > 2 else "", "chars": len(text),
                       "hash": h, "file": "downloaded"}
                downloaded._append_row(row); downloaded._next_id += 1; downloaded.count += 1
                router.partial_fit(text, path)
                del text
                row = dict(row, id=trained._next_id, file="trained")
                trained._append_row(row); trained._next_id += 1; trained.count += 1
                per_area[path[0]] += 1
                ingested += 1
                pulled += 1
                if ingested % 100 == 0:
                    print(f"  collecting... {ingested} | {(time.time()-t0)/60:.0f} min | "
                          f"{dict(per_area)}", flush=True)
        if ingested >= next_fit:
            router.fit()
            acc = evaluate(router, val)
            router.save(CKPT)
            names = ["area", "sector", "subsector"]
            txt = " | ".join(f"{names[i] if i < 3 else i}={a:.0%}" for i, a in enumerate(acc))
            print(f">>> TRAIN: {ingested} sources | {txt} | {(time.time()-t0)/60:.0f} min | saved\n",
                  flush=True)
            next_fit += FIT_EVERY

    router.fit()
    acc = evaluate(router, val)
    router.save(CKPT)
    print(f"\nEND: {ingested} sources | accuracy per level={[f'{a:.0%}' for a in acc]} | "
          f"{(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
