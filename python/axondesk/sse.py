"""Server-Sent Events: um fluxo por sessão, com reconexão a partir do último `seq`.

Escolhido no lugar de WebSocket porque o app só precisa do servidor para o cliente, e
SSE reconecta sozinho no navegador. Escolhido no lugar de stdio JSON-RPC porque dois
painéis transmitem ao mesmo tempo (tokens do chat e saída de comando), e multiplexar isso
na mão sobre stdout é inventar um protocolo que o HTTP já tem.
"""

import json
import queue
import threading


class Barramento:
    """Distribui eventos para todos os clientes conectados, guardando um histórico curto.

    O histórico é o que permite o cliente reconectar e pedir "tudo a partir do seq N" sem
    perder nada que aconteceu enquanto a conexão estava caída.
    """

    def __init__(self, historico=500):
        self._seq = 0
        self._historico = []
        self._max = historico
        self._clientes = []
        self._trava = threading.Lock()

    def publicar(self, evento):
        with self._trava:
            self._seq += 1
            evento = dict(evento, seq=self._seq)
            self._historico.append(evento)
            if len(self._historico) > self._max:
                del self._historico[: len(self._historico) - self._max]
            clientes = list(self._clientes)
        for fila in clientes:
            try:
                fila.put_nowait(evento)
            except queue.Full:
                pass
        return evento["seq"]

    def inscrever(self, desde=0):
        fila = queue.Queue(maxsize=1000)
        with self._trava:
            atrasados = [e for e in self._historico if e["seq"] > desde]
            self._clientes.append(fila)
        for e in atrasados:
            try:
                fila.put_nowait(e)
            except queue.Full:
                break
        return fila

    def desinscrever(self, fila):
        with self._trava:
            if fila in self._clientes:
                self._clientes.remove(fila)

    @property
    def seq(self):
        with self._trava:
            return self._seq


def escrever(handler, barramento, desde=0, intervalo=15.0):
    """Segura a conexão e escreve eventos até o cliente sumir.

    O comentário periódico (`: ping`) existe porque proxies e o próprio Windows derrubam
    conexão ociosa -- e uma resposta em CPU pode passar um minuto sem produzir token.
    """
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()

    fila = barramento.inscrever(desde=desde)
    try:
        while True:
            try:
                evento = fila.get(timeout=intervalo)
            except queue.Empty:
                handler.wfile.write(b": ping\n\n")
                handler.wfile.flush()
                continue

            dados = json.dumps(evento, ensure_ascii=False)
            handler.wfile.write(f"id: {evento['seq']}\n".encode("utf-8"))
            handler.wfile.write(f"event: {evento.get('tipo', 'mensagem')}\n"
                                .encode("utf-8"))
            handler.wfile.write(f"data: {dados}\n\n".encode("utf-8"))
            handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass                                    # cliente fechou a aba; normal
    finally:
        barramento.desinscrever(fila)
