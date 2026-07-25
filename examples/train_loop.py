"""axon-lang: LONG LOOP of collect+train (run ~1h gathering lots of data).

Collects in batches from several sources (download -> train -> delete), with a polite
pause between downloads, saving the checkpoint after each batch. After each batch it
measures the accuracy (area) on a separate VALIDATION set (untrained). Runs until the
pool is exhausted or the source cap is reached. Resumable and with persistent dedup
(never re-downloads).
"""

import os
import sys
import time

import pyaxon as ax
from pyaxon import collect, corpus

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "axon_lang_data", "loop")
CKPT = os.path.join(OUT, "router.json")

MAX_SOURCES = 100000   # high cap: keeps collecting for a long time (~1h or more)
BATCH = 50             # sources per batch (measures accuracy each batch)
DELAY = 0.2            # pause between downloads (polite)
VAL_PER_AREA = 4

# Areas -> many searches (broad coverage, to gather thousands of articles).
TAXO = {
    "portugues": ["gramática", "sintaxe", "morfologia (linguística)", "língua portuguesa",
                  "ortografia", "semântica", "fonologia", "literatura brasileira",
                  "Machado de Assis", "modernismo no Brasil", "romantismo", "verbo",
                  "substantivo", "figura de linguagem", "análise sintática"],
    "matematica": ["matemática", "álgebra", "geometria", "aritmética", "cálculo",
                   "trigonometria", "equação", "número primo", "função (matemática)",
                   "matriz (matemática)", "probabilidade", "estatística", "logaritmo",
                   "teoria dos números", "geometria analítica"],
    "fisica": ["física", "mecânica clássica", "energia", "termodinâmica", "eletromagnetismo",
               "óptica", "mecânica quântica", "relatividade", "força", "movimento",
               "eletricidade", "onda", "gravidade"],
    "quimica": ["química", "átomo", "tabela periódica", "reação química", "molécula",
                "ligação química", "ácido", "base (química)", "química orgânica",
                "elemento químico", "solução (química)"],
    "biologia": ["biologia", "célula", "genética", "ecologia", "evolução", "DNA",
                 "fotossíntese", "sistema imunológico", "botânica", "zoologia",
                 "anatomia humana", "microbiologia"],
    "historia": ["história do Brasil", "história", "idade média", "revolução industrial",
                 "segunda guerra mundial", "império romano", "brasil colônia",
                 "renascimento", "idade moderna", "grécia antiga"],
    "filosofia": ["filosofia", "lógica", "ética", "metafísica", "epistemologia",
                  "filosofia grega", "sócrates", "immanuel kant"],
    "geografia": ["geografia", "clima", "relevo", "geografia do Brasil", "hidrografia",
                  "urbanização", "cartografia"],
}


def build_validation(col, min_chars=400):
    val = []
    for area, queries in TAXO.items():
        picked = 0
        for q in queries:
            if picked >= VAL_PER_AREA:
                break
            try:
                titles = corpus.wikipedia_search(q, limit=8)
            except Exception:
                continue
            for t in titles:
                if picked >= VAL_PER_AREA:
                    break
                url = corpus.mw_url("pt.wikipedia.org", t)
                if url in col.seen_urls or t in col.seen_titles:
                    continue
                col.seen_urls.add(url)
                col.seen_titles.add(t)  # exclude from training
                try:
                    txt = corpus.wikipedia_extract(t)
                except Exception:
                    continue
                if len(txt) >= min_chars:
                    val.append((txt, area))
                    picked += 1
    return val


def interleave(*ls):
    out, i = [], 0
    while any(i < len(l) for l in ls):
        for l in ls:
            if i < len(l):
                out.append(l[i])
        i += 1
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    router = ax.router.HierarchicalRouter()
    if os.path.exists(CKPT):
        router.load(CKPT)
        print("checkpoint loaded (continuing)", flush=True)
    col = collect.Collector(OUT)
    print(f"already indexed: {col.count} sources", flush=True)

    print("building validation (untrained)...", flush=True)
    val = build_validation(col)
    print(f"validation: {len(val)} articles", flush=True)

    print("building source pool (wikipedia + wikibooks + wikisource)...", flush=True)
    specs = interleave(
        collect.build_mw_specs(TAXO, "wikipedia", per_query=120),
        collect.build_mw_specs(TAXO, "wikibooks", per_query=25),
        collect.build_mw_specs({"portugues": {"literatura": TAXO["portugues"][7:11]}},
                               "wikisource", per_query=20),
    )
    print(f"pool: {len(specs)} candidates | starting long collection...", flush=True)

    t0 = time.time()
    it = 0
    while col.count < MAX_SOURCES:
        added = col.stream_train_specs(router, specs, target=BATCH, chunk=6,
                                       checkpoint=CKPT, save_every=25, delay=DELAY, log=None)
        if added == 0:
            print("pool exhausted — collection finished.", flush=True)
            break
        hit = sum(1 for txt, area in val if router.route(txt)[:1] == [area])
        acc = hit / len(val) if val else 0.0
        it += 1
        mins = (time.time() - t0) / 60
        print(f"batch {it}: sources={col.count} | accuracy(area)={acc:.0%} | {mins:.0f} min",
              flush=True)

    print(f"\nEND: {col.count} sources collected in {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
