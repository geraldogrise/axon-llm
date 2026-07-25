"""Train AxonLM on a text using the high-level pyaxon.lm API.

Training uses forward_batch (3D batch): several windows processed in a single
call per step — much faster than one window at a time.
"""

import time

import pyaxon as ax

# --- Corpus: a nursery rhyme (repetitive structure the model can learn) ---
TEXT = (
    "a barata diz que tem sete saias de filo. "
    "e mentira da barata ela tem e uma so. "
    "ah ah ah ho ho ho ela tem e uma so. "
    "a barata diz que tem um sapato de fivela. "
    "e mentira da barata o que ela tem e uma pele. "
    "ah ah ah ho ho ho o que ela tem e uma pele. "
) * 2

t0 = time.perf_counter()
model, tok, config = ax.lm.train_lm(
    TEXT, vocab_size=80, dim=64, num_heads=4, num_layers=3, context=24,
    epochs=8, batch_size=8, lr=0.002,
    log=lambda m: print("[train]", m, flush=True),
)
dt = time.perf_counter() - t0
print(f"\nTraining (3D batch) completed in {dt:.1f}s")

print("\n--- Generation (prompt = 'a barata') ---")
print(ax.lm.generate(model, tok, "a barata", n_tokens=40, context=config["context"],
                     top_k=5, temperature=0.6))
