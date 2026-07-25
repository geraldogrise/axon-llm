# Serving the AxonLM (REST API + web demo)

`examples/serve.py` trains a small AxonLM and exposes it over HTTP, using **only
the Python standard library** (no Flask/dependencies).

## How to run

```powershell
# Make sure the Python extension is built (see docs/BUILD.md)
$env:PYTHONPATH = "$PWD\python"
python examples\serve.py
# -> http://localhost:8000
```

Open the browser at `http://localhost:8000`, type a prompt (e.g., "a barata") and
click **Generate text**. Adjust tokens, temperature, and top-k with the controls.

## Endpoints

| Method | Route        | Description                                          |
|--------|--------------|------------------------------------------------------|
| GET    | `/`          | Demo page (`examples/web/index.html`)                |
| POST   | `/generate`  | Generates text from a prompt (JSON)                  |

### `POST /generate`

Request:

```json
{ "prompt": "a barata", "n_tokens": 40, "temperature": 0.7, "top_k": 5 }
```

Response:

```json
{ "text": "a barata diz que tem e mentira da barata o que ela tem e uma pele. ..." }
```

Example with `curl`:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a barata","n_tokens":30,"temperature":0.7,"top_k":5}'
```

## Notes

- The model is trained **once at startup** (fast, small demo corpus). For
  production, train and save a checkpoint (`ax.save`) and load it here (`ax.load`).
- Generation runs inside `ax.no_grad()` (without building an autograd graph).
- `python examples/serve.py --self-test` trains, generates a sample, and exits — useful for CI.
