# pyaxon CLI

The CLI lets you train, generate, and serve the AxonLM without writing code.

```powershell
$env:PYTHONPATH = "$PWD\python"     # or install with: pip install .
```

## `train` — train an AxonLM

```powershell
python -m pyaxon train --input corpus.txt --out meu_modelo `
    --vocab 100 --dim 64 --heads 4 --layers 3 --context 24 --epochs 8 --lr 0.002
```

Produces a **bundle** with three files:

| File               | Contents                          |
|--------------------|-----------------------------------|
| `meu_modelo.json`  | hyperparameters (config)          |
| `meu_modelo.ckpt`  | model weights (binary)            |
| `meu_modelo.bpe`   | BPE tokenizer (vocabulary+merges) |

## `generate` — generate text from a saved model

```powershell
python -m pyaxon generate --model meu_modelo --prompt "a barata" `
    --tokens 40 --topk 5 --temp 0.7
```

## `serve` — REST API + web demo

```powershell
python -m pyaxon serve --model meu_modelo --port 8000
# -> http://localhost:8000
```

## Use as a library (`pyaxon.lm`)

The same logic is available programmatically:

```python
import pyaxon as ax

model, tok, cfg = ax.lm.train_lm(open("corpus.txt").read(), epochs=8)
print(ax.lm.generate(model, tok, "a barata", n_tokens=40, context=cfg["context"]))

ax.lm.save_bundle("meu_modelo", model, tok, cfg)
model, tok, cfg = ax.lm.load_bundle("meu_modelo")
```

> After `pip install .`, the command is available directly as `pyaxon ...`
> (entry point defined in `pyproject.toml`).
