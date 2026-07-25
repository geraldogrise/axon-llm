"""Robust HTTP server for axon-lang: routing + RAG + optional generation.

A production-shaped server (vs the demo): concurrent (ThreadingHTTPServer), JSON API,
per-request error handling, a /health check, and a confidence gate. Loads the modular
router + sparse knowledge base once and serves:

    GET  /health                 -> {"status":"ok", ...}
    POST /route   {"text": ...}  -> {"path": [...], "confidence": 0.87}
    POST /answer  {"question":..} -> RAG passages, and a generated answer if Ollama is up

Usage:
    python examples/serve_axonlang.py --router path/router --kb path/kb.json.gz [--port 8000]
    # add --gen to also produce fluent answers via Ollama (needs `ollama serve`)
"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pyaxon as ax

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def build_handler(router, kb, use_gen, threshold):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "ok", "router": router is not None,
                                 "kb_passages": len(getattr(kb, "texts", [])),
                                 "generation": use_gen})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                req = self._body()
                if self.path == "/route":
                    text = req.get("text", "")
                    path = router.route(text, threshold=threshold) if threshold \
                        else router.route(text)
                    conf = router.confidence(text) if hasattr(router, "confidence") else None
                    self._json(200, {"path": path, "confidence": conf})
                elif self.path == "/answer":
                    q = req.get("question", "")
                    if use_gen:
                        out = ax.generate.answer(q, kb, router,
                                                 top_k=req.get("top_k", 3),
                                                 multi=req.get("multi", 2),
                                                 threshold=threshold)
                        self._json(200, out)
                    else:
                        paths, passages = kb.answer(q, router=router,
                                                    top_k=req.get("top_k", 3),
                                                    multi=req.get("multi", 2))
                        self._json(200, {"route": paths,
                                         "passages": [{"text": t, "path": p, "score": s}
                                                      for t, p, s in passages]})
                else:
                    self._json(404, {"error": "unknown route"})
            except Exception as e:  # noqa: BLE001 -- never crash the server on one request
                self._json(500, {"error": str(e)})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", required=True, help="prefix of the saved router")
    ap.add_argument("--kb", required=True, help="path to kb.json.gz")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threshold", type=float, default=0.0, help="confidence gate (0..1)")
    ap.add_argument("--gen", action="store_true", help="generate fluent answers via Ollama")
    args = ap.parse_args()

    # modular router if a .gate.json exists, else the flat LinearRouter
    if os.path.exists(args.router + ".gate.json"):
        router = ax.modular.ModularRouter().load(args.router)
    else:
        router = ax.router.LinearRouter().load(args.router)
    kb = (ax.vindex.SparseKB() if args.kb.endswith(".sparse.json.gz")
          else ax.rag.KnowledgeBase()).load(args.kb)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port),
                              build_handler(router, kb, args.gen, args.threshold))
    print(f"[axon-lang] http://localhost:{args.port}  "
          f"(/health /route /answer{' +gen' if args.gen else ''})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
