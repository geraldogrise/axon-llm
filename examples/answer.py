"""axon-lang: ANSWER (RAG). Routes the question -> retrieves subsector passages ->
composes the answer. Builds the router + knowledge base and demonstrates.

Stores the passages (indexed by area -> sector -> subsector) — that is what enables answering.
"""

import os
import sys
import time

import pyaxon as ax
from pyaxon import corpus

try:  # let the Windows console accept accents/symbols
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "axon_lang_data", "rag")
ARTICLES_PER_LEAF = 10
HOST = "pt.wikipedia.org"

# Hierarchy (3 levels) — can have as many sublevels as you want.
TAXO = {
    "matematica": {
        "algebra": {"linear": ["álgebra linear", "matriz matemática", "determinante", "autovalor"]},
        "calculo": {"derivadas": ["derivada", "cálculo diferencial", "limite matemática"]},
    },
    "fisica": {
        "mecanica": {"classica": ["mecânica clássica", "leis de Newton", "energia cinética"]},
        "quantica": {"basica": ["mecânica quântica", "função de onda", "átomo de Bohr"]},
    },
    "biologia": {
        "genetica": {"basica": ["genética", "DNA", "leis de Mendel", "hereditariedade"]},
        "celula": {"basica": ["célula", "fotossíntese", "membrana plasmática"]},
    },
    "portugues": {
        "gramatica": {"sintaxe": ["sintaxe", "análise sintática", "concordância verbal"]},
        "literatura": {"autores": ["Machado de Assis", "modernismo no Brasil", "romantismo"]},
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


def collect_and_build():
    router = ax.router.LinearRouter(max_features=4000, ngram=2, epochs=300, lr=0.05, balance=True)
    kb = ax.rag.KnowledgeBase(max_features=8000, ngram=1)
    seen = set()
    for path, queries in iter_leaves(TAXO):
        got = 0
        for q in queries:
            if got >= ARTICLES_PER_LEAF:
                break
            try:
                titles = corpus.mw_search(HOST, q, limit=12)
            except Exception:
                continue
            for t in titles:
                if got >= ARTICLES_PER_LEAF or t in seen:
                    continue
                seen.add(t)
                try:
                    txt = corpus.wikipedia_extract(t)
                except Exception:
                    continue
                if len(txt) < 400:
                    continue
                router.partial_fit(txt, path)       # train the router (classifies)
                kb.add_document(txt, path)           # STORE the passages (for answering)
                got += 1
                time.sleep(0.08)
        print(f"  {' > '.join(path):40s}: {got} articles", flush=True)
    router.fit()
    kb.build()
    return router, kb


def main():
    os.makedirs(OUT, exist_ok=True)
    r_path = os.path.join(OUT, "router.json")
    kb_path = os.path.join(OUT, "kb.json.gz")   # compressed (gzip + int8)

    if os.path.exists(r_path) and os.path.exists(kb_path):
        print("loading saved router + knowledge base...", flush=True)
        router = ax.router.LinearRouter().load(r_path)
        kb = ax.rag.KnowledgeBase().load(kb_path)
    else:
        print("collecting and building (router + base)...", flush=True)
        router, kb = collect_and_build()
        router.save(r_path)
        kb.save(kb_path)
        print(f"saved: {len(kb.texts)} passages in the base\n", flush=True)

    questions = [
        "o que é o determinante de uma matriz?",
        "como funcionam as leis de Newton na mecânica?",
        "qual o papel do DNA na hereditariedade?",
        "quem foi Machado de Assis na literatura brasileira?",
    ]
    for q in questions:
        path, passages = kb.answer(q, router=router, top_k=1)
        print("=" * 70)
        print("QUESTION:", q)
        print("routed to:", " > ".join(path))
        if passages:
            passage, _, score = passages[0]
            print(f"ANSWER (most relevant passage, score {score:.2f}):")
            print("  " + passage[:400].replace("\n", " ") + "...")
        print()


if __name__ == "__main__":
    main()
