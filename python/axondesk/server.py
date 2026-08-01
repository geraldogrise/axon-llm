"""Servidor HTTP do axon-desk.

Liga em 127.0.0.1 numa porta que o sistema escolhe (porta 0) -- assim não há colisão com
nada, e a porta real vai para o Electron pelo aperto de mão no stdout.

Todo pedido exige o token. O SSE aceita o token na query porque `EventSource` não deixa
mandar cabeçalho.
"""

import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import llm, sse


def construir(motor, barramento, token, origem_permitida="*"):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        # O log padrão escreve em stderr a cada pedido; com SSE isso vira enxurrada.
        def log_message(self, *args):
            pass

        # ------------------------------------------------------------ auxiliares
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", origem_permitida)
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, codigo, obj):
            corpo = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(codigo)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(corpo)))
            self._cors()
            self.end_headers()
            self.wfile.write(corpo)

        def _autorizado(self, params):
            cab = self.headers.get("Authorization", "")
            if cab == f"Bearer {token}":
                return True
            return (params.get("token") or [None])[0] == token

        def _corpo(self):
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode("utf-8"))
            except ValueError:
                return {}

        # ------------------------------------------------------------ verbos
        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            u = urlparse(self.path)
            params = parse_qs(u.query)
            if not self._autorizado(params):
                return self._json(401, {"erro": "token inválido"})

            if u.path == "/health":
                return self._json(200, {
                    "estado": motor.estado,
                    "experts": motor.saude(),
                    "ollama": llm.SAUDE.ok(),
                    "modelos": llm.modelos(),
                    "seq": barramento.seq,
                    "em_curso": llm.REGISTRO.em_curso(),
                })

            if u.path == "/events":
                desde = int((params.get("desde") or ["0"])[0])
                return sse.escrever(self, barramento, desde=desde)

            return self._json(404, {"erro": "rota inexistente"})

        def do_POST(self):
            u = urlparse(self.path)
            params = parse_qs(u.query)
            if not self._autorizado(params):
                return self._json(401, {"erro": "token inválido"})

            try:
                if u.path == "/chat":
                    return self._chat(self._corpo())
                if u.path == "/cancel":
                    run_id = self._corpo().get("run_id", "")
                    return self._json(200, {"cancelado": llm.REGISTRO.cancelar(run_id)})
                if u.path == "/shutdown":
                    self._json(200, {"ok": True})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return None
            except Exception as erro:                      # noqa: BLE001
                # Um pedido nunca derruba o servidor.
                return self._json(500, {"erro": f"{type(erro).__name__}: {erro}"})

            return self._json(404, {"erro": "rota inexistente"})

        # ------------------------------------------------------------ chat
        def _chat(self, dados):
            texto = (dados.get("texto") or "").strip()
            if not texto:
                return self._json(400, {"erro": "texto vazio"})
            modelo = dados.get("modelo") or "qwen2.5-coder:7b-instruct"
            run_id = "run_" + uuid.uuid4().hex[:12]

            # Responde já; tudo o mais sai pelo fluxo de eventos. É isso que mantém os
            # dois painéis em ordem e torna a reconexão trivial.
            self._json(200, {"run_id": run_id})
            threading.Thread(target=_responder, daemon=True,
                             args=(motor, barramento, run_id, texto, modelo)).start()
            return None

    return Handler


def _responder(motor, barramento, run_id, texto, modelo):
    """Roteia, recupera e transmite. Roda fora da thread do pedido."""
    barramento.publicar({"tipo": "usuario", "run_id": run_id, "texto": texto})

    expert, passagens = motor.recuperar(texto)
    if expert is None:
        barramento.publicar({
            "tipo": "erro", "run_id": run_id,
            "erro": "nenhum expert carregado ainda" if motor.estado != "pronto"
                    else "nenhum expert respondeu"})
        return

    _, ranking = motor.rotear(texto)
    barramento.publicar({"tipo": "rota", "run_id": run_id, "expert": expert,
                         "ranking": ranking[:5],
                         "passagens": [{"caminho": p["caminho"], "score": p["score"]}
                                       for p in passagens]})

    contexto = "\n\n---\n\n".join(p["texto"] for p in passagens)
    mensagens = [
        {"role": "system",
         "content": "Você responde em português do Brasil, de forma direta e técnica. "
                    "Use o material fornecido. Se ele não cobrir a pergunta, diga isso "
                    "em vez de inventar."},
        {"role": "user", "content": f"MATERIAL:\n{contexto}\n\nPERGUNTA: {texto}"},
    ]

    try:
        for pedaco in llm.chat_stream(mensagens, modelo=modelo, run_id=run_id):
            if pedaco.get("done"):
                barramento.publicar({
                    "tipo": "fim", "run_id": run_id,
                    "tokens_entrada": pedaco["prompt_eval_count"],
                    "tokens_saida": pedaco["eval_count"],
                    "ms_prefill": round(pedaco["prompt_eval_duration"] / 1e6),
                    "ms_geracao": round(pedaco["eval_duration"] / 1e6),
                })
            else:
                barramento.publicar({"tipo": "delta", "run_id": run_id,
                                     "texto": pedaco["delta"]})
    except llm.Cancelado:
        barramento.publicar({"tipo": "cancelado", "run_id": run_id})
    except Exception as erro:                              # noqa: BLE001
        barramento.publicar({"tipo": "erro", "run_id": run_id,
                             "erro": f"{type(erro).__name__}: {erro}"})


def subir(motor, barramento, token, porta=0):
    """Liga o servidor e devolve (servidor, porta_real)."""
    handler = construir(motor, barramento, token)
    servidor = ThreadingHTTPServer(("127.0.0.1", porta), handler)
    servidor.daemon_threads = True
    return servidor, servidor.server_address[1]
