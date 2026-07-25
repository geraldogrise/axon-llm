"""Serve AxonLM over a REST API + web demo (Python standard library only).

Trains a small model at startup using pyaxon.lm and exposes it over HTTP.

Endpoints:
  GET  /            -> demo page (examples/web/index.html)
  POST /generate    -> {prompt, n_tokens, temperature, top_k} -> {text}

Usage:
  python examples/serve.py                # train and serve at http://localhost:8000
  python examples/serve.py --self-test    # train, generate a sample and exit (for tests)
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pyaxon as ax
from pyaxon import lm

# Web page bundled inside the pyaxon package.
INDEX_HTML = os.path.join(os.path.dirname(ax.__file__), "web", "index.html")

TEXT = (
    "a barata diz que tem sete saias de filo. "
    "e mentira da barata ela tem e uma so. "
    "ah ah ah ho ho ho ela tem e uma so. "
    "a barata diz que tem um sapato de fivela. "
    "e mentira da barata o que ela tem e uma pele. "
    "ah ah ah ho ho ho o que ela tem e uma pele. "
) * 2

# Train once at startup (small demo corpus).
model, tokenizer, config = lm.train_lm(
    TEXT, vocab_size=80, dim=64, num_heads=4, num_layers=3, context=24, epochs=8,
    log=lambda m: print("[train]", m, flush=True),
)


def generate(prompt, n_tokens=40, temperature=0.7, top_k=5):
    return lm.generate(model, tokenizer, prompt, n_tokens=n_tokens, context=config["context"],
                       top_k=top_k, temperature=temperature)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(INDEX_HTML, "r", encoding="utf-8") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/generate":
            self._send(404, json.dumps({"error": "unknown route"}), "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            text = generate(req.get("prompt", ""), req.get("n_tokens", 40),
                            req.get("temperature", 0.7), req.get("top_k", 5))
            self._send(200, json.dumps({"text": text}), "application/json; charset=utf-8")
        except Exception as e:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(e)}), "application/json")


def main():
    if "--self-test" in sys.argv:
        print("[self-test] generation:", generate("a barata", 30))
        return
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("[serve] AxonLM ready at http://localhost:8000  (Ctrl+C to exit)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
