"""pyaxon command-line interface: train, generate and serve the AxonLM.

Examples:
    python -m pyaxon train --input corpus.txt --out my_model --epochs 10
    python -m pyaxon generate --model my_model --prompt "a barata" --tokens 40
    python -m pyaxon serve --model my_model --port 8000
"""

import argparse
import sys

from . import lm


def cmd_train(args):
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    model, tok, config = lm.train_lm(
        text, vocab_size=args.vocab, dim=args.dim, num_heads=args.heads,
        num_layers=args.layers, context=args.context, epochs=args.epochs,
        lr=args.lr, batch_size=args.batch, seed=args.seed,
        log=lambda m: print("[train]", m, flush=True),
    )
    lm.save_bundle(args.out, model, tok, config)
    print(f"[train] model saved to {args.out}.{{json,ckpt,bpe}}")


def cmd_generate(args):
    model, tok, config = lm.load_bundle(args.model)
    text = lm.generate(model, tok, args.prompt, n_tokens=args.tokens,
                       context=config["context"], top_k=args.topk, temperature=args.temp)
    print(text)


def cmd_serve(args):
    # Late import: http.server is only needed when serving.
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import os

    model, tok, config = lm.load_bundle(args.model)
    # Web page bundled with the package (pyaxon/web/index.html).
    index_html = os.path.join(os.path.dirname(__file__), "web", "index.html")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            b = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path in ("/", "/index.html") and os.path.exists(index_html):
                with open(index_html, "r", encoding="utf-8") as f:
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
                text = lm.generate(model, tok, req.get("prompt", ""),
                                   n_tokens=req.get("n_tokens", 40),
                                   context=config["context"],
                                   top_k=req.get("top_k", 5),
                                   temperature=req.get("temperature", 0.7))
                self._send(200, json.dumps({"text": text}), "application/json; charset=utf-8")
            except Exception as e:  # noqa: BLE001
                self._send(500, json.dumps({"error": str(e)}), "application/json")

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[serve] http://localhost:{args.port}  (Ctrl+C to exit)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


def build_parser():
    p = argparse.ArgumentParser(prog="pyaxon", description="AxonLM -- train, generate and serve")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train an AxonLM on a text file")
    t.add_argument("--input", required=True, help="training text file")
    t.add_argument("--out", required=True, help="output bundle prefix")
    t.add_argument("--vocab", type=int, default=100)
    t.add_argument("--dim", type=int, default=64)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--context", type=int, default=24)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--lr", type=float, default=0.002)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--seed", type=int, default=1)
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("generate", help="generate text from a saved model")
    g.add_argument("--model", required=True, help="bundle prefix")
    g.add_argument("--prompt", default="")
    g.add_argument("--tokens", type=int, default=40)
    g.add_argument("--topk", type=int, default=5)
    g.add_argument("--temp", type=float, default=0.7)
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("serve", help="serve the model over HTTP (REST API + web)")
    s.add_argument("--model", required=True, help="bundle prefix")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()
