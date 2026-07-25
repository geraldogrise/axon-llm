"""axon-lang: streaming collection with an interleaved curriculum (how a child learns).

Downloads Wikipedia sources interleaving the subjects (Portuguese first, then
mathematics, physics, ...), trains the router on each text and **deletes the text
immediately** — keeping only the URL in the CSV and the knowledge in the checkpoint.
Resumable: run again to fetch the next sources (never re-downloads what is already in the CSV).
"""

import os

import pyaxon as ax

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "axon_lang_data", "stream")
CKPT = os.path.join(OUT, "router.json")
TARGET = 50  # increase to 1000 / 10000 (runs in batches, resumable)

# Curriculum: starts with PORTUGUESE (grammar, literature, linguistics), then the
# other areas. Leaves = Wikipedia searches (basic subsectors first).
CURRICULUM = {
    "portugues": {
        "gramatica": ["gramática", "morfologia (linguística)", "sintaxe",
                      "ortografia da língua portuguesa", "classes de palavras", "verbo"],
        "literatura": ["literatura brasileira", "Machado de Assis", "modernismo no Brasil",
                       "romantismo (Brasil)", "Carlos Drummond de Andrade"],
        "linguistica": ["língua portuguesa", "linguística", "semântica"],
    },
    "matematica": {
        "aritmetica": ["aritmética", "número"],
        "algebra": ["álgebra", "equação"],
        "geometria": ["geometria", "triângulo"],
    },
    "fisica": {"basico": ["física", "energia", "mecânica clássica"]},
    "quimica": {"basico": ["química", "átomo", "tabela periódica"]},
    "biologia": {"basico": ["biologia", "célula"]},
    "historia": {"basico": ["história do Brasil", "história"]},
    "filosofia": {"basico": ["filosofia", "lógica"]},
}


def main():
    os.makedirs(OUT, exist_ok=True)
    router = ax.router.HierarchicalRouter()
    if os.path.exists(CKPT):
        router.load(CKPT)  # resume the saved training
        print("checkpoint loaded (continuing training)")

    col = ax.collect.Collector(OUT)  # existing CSV = persistent dedup
    print(f"already indexed (no re-download): {col.count} sources")

    added = col.stream_train(router, CURRICULUM, target=TARGET, chunk=4,
                             checkpoint=CKPT, save_every=20)

    # Only the URL remains (the text was discarded); the model kept the knowledge.
    print(f"\ntraining checkpoint: {os.path.getsize(CKPT)} bytes (only the model, no texts)")
    print(f"CSV index: {col.csv_path}")
    with open(col.csv_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    print("first CSV lines:")
    for ln in lines[:4]:
        print("  ", ln[:110])

    # Route queries using only the trained model (texts already deleted).
    print("\n--- routing (text already discarded; uses only the model) ---")
    for q in ["análise sintática do sujeito e do predicado na oração",
              "a obra de Machado de Assis no realismo brasileiro",
              "resolver uma equação do segundo grau",
              "a estrutura do átomo e a tabela periódica"]:
        print(f"  {' > '.join(router.route(q))}  <-  \"{q[:45]}\"")


if __name__ == "__main__":
    main()
