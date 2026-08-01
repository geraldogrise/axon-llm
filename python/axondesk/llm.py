"""Cliente do Ollama com streaming, cancelamento e contagem exata de tokens.

Não substitui o `pyaxon.generate` -- aquele é API pública de uma biblioteca publicada e
fica intocado. Este módulo existe porque o app precisa de três coisas que o `generate`
não tem e não deveria ganhar:

  * streaming (a 10 tokens/s, uma resposta de 300 tokens é meio minuto de tela parada);
  * cancelamento (uma resposta ruim tem que custar 3 s, não 40);
  * `num_ctx` explícito -- sem ele o Ollama aplica o padrão dele, 2048 ou 4096, e trunca
    o contexto em silêncio, mesmo o modelo declarando 32.768.

Usa `/api/chat` em vez de `/api/generate`: há histórico de verdade e o modelo é
*instruct*, então passar pelo template de chat dele importa.
"""

import json
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = "http://localhost:11434"
URL_CHAT = BASE + "/api/chat"
URL_TAGS = BASE + "/api/tags"

# O Ollama descarrega o modelo depois de 5 min ocioso. Recarregar 4,7 GB em CPU custa
# dezenas de segundos e é o que o usuário mais sente entre uma pergunta e outra.
KEEP_ALIVE = "30m"

# O modelo declara 32k, mas contexto grande em CPU custa caro no *prefill* -- que é
# separado da geração e costuma dominar. 8k é o teto útil, não uma meta.
NUM_CTX = 8192


class Cancelado(Exception):
    """Levantada dentro do gerador quando o cliente cancela a geração."""


class Saude:
    """Sonda o Ollama com validade, em vez de a cada resposta.

    O `pyaxon.generate.is_ollama_up` sonda até 1,5 s **toda vez** que alguém pede uma
    resposta -- para descobrir algo que muda talvez uma vez por dia.
    """

    def __init__(self, validade=10.0):
        self.validade = validade
        self._ate = 0.0
        self._ok = False
        self._trava = threading.Lock()

    def ok(self):
        with self._trava:
            agora = time.monotonic()
            if agora < self._ate:
                return self._ok
            self._ok = self._sondar()
            self._ate = agora + self.validade
            return self._ok

    @staticmethod
    def _sondar():
        try:
            with urllib.request.urlopen(URL_TAGS, timeout=1.5) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False


def modelos():
    """Modelos registrados no Ollama. Lista vazia se ele não estiver no ar."""
    try:
        with urllib.request.urlopen(URL_TAGS, timeout=3) as r:
            d = json.loads(r.read().decode("utf-8"))
        return sorted(m["name"] for m in d.get("models", []))
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return []


class Registro:
    """Cancelamento: guarda a resposta HTTP aberta de cada geração em curso.

    Fechar o socket faz o Ollama abortar a geração do lado dele -- é o que torna o
    cancelamento instantâneo em vez de "para de mostrar mas continua gastando CPU".
    """

    def __init__(self):
        self._vivos = {}
        self._trava = threading.Lock()

    def registrar(self, run_id, resp):
        with self._trava:
            self._vivos[run_id] = resp

    def concluir(self, run_id):
        with self._trava:
            self._vivos.pop(run_id, None)

    def cancelar(self, run_id):
        with self._trava:
            resp = self._vivos.pop(run_id, None)
        if resp is None:
            return False
        try:
            resp.close()
        except Exception:                                  # noqa: BLE001
            pass
        return True

    def em_curso(self):
        with self._trava:
            return sorted(self._vivos)


REGISTRO = Registro()
SAUDE = Saude()


def chat_stream(mensagens, *, modelo, run_id=None, num_ctx=NUM_CTX, temperatura=0.3,
                num_predict=1024, formato=None, timeout=600):
    """Gera a resposta em pedaços.

    Devolve, em sequência, dicionários `{"delta": str}` e, por último, um
    `{"done": True, "prompt_eval_count": n, "eval_count": m, ...}` -- os dois contadores
    são a contagem **exata** de tokens, de graça, sem precisar de tokenizador.
    """
    corpo = {
        "model": modelo,
        "messages": mensagens,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperatura,
            "num_predict": num_predict,
        },
    }
    if formato:
        corpo["format"] = formato

    req = urllib.request.Request(
        URL_CHAT, data=json.dumps(corpo).encode("utf-8"),
        headers={"Content-Type": "application/json"})

    resp = urllib.request.urlopen(req, timeout=timeout)
    if run_id:
        REGISTRO.registrar(run_id, resp)
    try:
        # O objeto devolvido já é um arquivo em buffer: iterar dá uma linha NDJSON por
        # vez, sem remontar pedaços na mão.
        for linha in resp:
            linha = linha.strip()
            if not linha:
                continue
            try:
                quadro = json.loads(linha.decode("utf-8"))
            except ValueError:
                continue

            if quadro.get("done"):
                yield {
                    "done": True,
                    "prompt_eval_count": quadro.get("prompt_eval_count", 0),
                    "eval_count": quadro.get("eval_count", 0),
                    "eval_duration": quadro.get("eval_duration", 0),
                    "prompt_eval_duration": quadro.get("prompt_eval_duration", 0),
                    "load_duration": quadro.get("load_duration", 0),
                }
                return

            pedaco = (quadro.get("message") or {}).get("content", "")
            if pedaco:
                yield {"delta": pedaco}
    except (urllib.error.URLError, OSError, ValueError) as erro:
        # Socket fechado por cancelamento chega aqui -- não é falha.
        raise Cancelado(str(erro)) from erro
    finally:
        if run_id:
            REGISTRO.concluir(run_id)
        try:
            resp.close()
        except Exception:                                  # noqa: BLE001
            pass


def aquecer(modelo):
    """Carrega o modelo na memória do Ollama, para a primeira pergunta não pagar isso."""
    try:
        for _ in chat_stream([{"role": "user", "content": "oi"}],
                             modelo=modelo, num_predict=1, timeout=300):
            pass
        return True
    except Exception as erro:                              # noqa: BLE001
        print(f"[axondesk] aquecimento falhou: {erro}", file=sys.stderr, flush=True)
        return False
