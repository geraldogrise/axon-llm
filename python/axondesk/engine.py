"""Carrega os experts e monta o contexto de uma pergunta.

Duas decisões que valem a explicação:

**Contorna o `AxonSystem.answer`.** Ele descarta o texto das passagens (`system.py:137`
guarda só `path` e `score`), chama a geração sem streaming e embute o prompt em português.
O `Expert.retrieve` devolve `(texto, caminho, score)` -- é o texto que a interface precisa
para citar e o prompt precisa para fundamentar.

**Carrega em segundo plano.** `AxonSystem.load` leva 145 s medidos com 19 experts. O
servidor sobe em milissegundos e informa o progresso; quem perguntar antes de terminar é
atendido com o que já estiver carregado.
"""

import os
import sys
import threading
import time

_ax = None


def importar_pyaxon():
    """Importa o `pyaxon` **na thread principal**, antes de o servidor aceitar pedidos.

    Não é otimização: é correção. Importar aqui dentro da thread de carga trava o
    servidor por completo. A thread que atende o primeiro pedido precisa importar peças
    internas do `http.client`/`email.parser`, e fica presa atrás da importação do
    `pyaxon`, que arrasta numpy e scipy. O sintoma é cruel -- a porta aceita conexão e
    nenhuma resposta sai, sem erro nenhum no log.

    Custa ~1,1 s, uma vez.
    """
    global _ax
    if _ax is None:
        raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if raiz not in sys.path:
            sys.path.insert(0, raiz)
        import pyaxon
        _ax = pyaxon
    return _ax


class Motor:
    def __init__(self, raiz_experts, apenas=None):
        self.raiz = raiz_experts
        self.apenas = apenas                 # lista de nomes, ou None para todos
        self.sistema = None
        self.experts = []
        self.estado = "parado"               # parado | carregando | pronto | erro
        self.erro = None
        self.total = 0
        self.carregados = []
        self.inicio = None
        self._trava = threading.Lock()
        self._ouvintes = []                  # callbacks(evento: dict)

    # ---------------------------------------------------------------- eventos
    def ao_progredir(self, callback):
        self._ouvintes.append(callback)

    def _emitir(self, evento):
        for cb in list(self._ouvintes):
            try:
                cb(evento)
            except Exception:                              # noqa: BLE001
                pass

    # ---------------------------------------------------------------- carga
    def carregar_async(self):
        t = threading.Thread(target=self._carregar, daemon=True, name="carga-experts")
        t.start()
        return t

    def _carregar(self):
        self.estado = "carregando"
        self.inicio = time.monotonic()
        try:
            ax = importar_pyaxon()          # já veio pronto da thread principal
            nomes = ax.system.AxonSystem._discover(self.raiz)
            if self.apenas:
                nomes = [n for n in nomes if n in set(self.apenas)]
            self.total = len(nomes)
            self._emitir({"tipo": "boot", "estado": "carregando",
                          "total": self.total, "carregados": []})

            experts = []
            for nome in nomes:
                t0 = time.monotonic()
                e = ax.system.Expert.load(os.path.join(self.raiz, nome), nome)
                experts.append(e)
                self.carregados.append(nome)
                self._emitir({"tipo": "boot", "estado": "carregando",
                              "expert": nome, "passagens": len(e.kb.texts),
                              "segundos": round(time.monotonic() - t0, 1),
                              "total": self.total,
                              "carregados": list(self.carregados)})

            sistema = ax.system.AxonSystem(experts).fit_domain_gate()
            with self._trava:
                self.sistema = sistema
                self.experts = experts
            self.estado = "pronto"
            self._emitir({"tipo": "boot", "estado": "pronto", "total": self.total,
                          "carregados": list(self.carregados),
                          "segundos": round(time.monotonic() - self.inicio, 1)})
        except Exception as erro:                          # noqa: BLE001
            self.estado = "erro"
            self.erro = f"{type(erro).__name__}: {erro}"
            self._emitir({"tipo": "boot", "estado": "erro", "erro": self.erro})
            print(f"[axondesk] falha ao carregar experts: {self.erro}",
                  file=sys.stderr, flush=True)

    # ---------------------------------------------------------------- consulta
    def saude(self):
        d = {"estado": self.estado, "total": self.total,
             "carregados": list(self.carregados)}
        if self.inicio:
            d["segundos"] = round(time.monotonic() - self.inicio, 1)
        if self.erro:
            d["erro"] = self.erro
        return d

    def rotear(self, pergunta):
        """(nome_do_expert, ranking) -- ranking com NOMES, não objetos Expert.

        O `AxonSystem.route` devolve objetos `Expert`, que não serializam para JSON.
        """
        with self._trava:
            sistema = self.sistema
        if sistema is None:
            return None, []
        vencedor, ranking = sistema.route(pergunta)
        if vencedor is None:
            return None, []
        return vencedor.name, [(e.name, round(float(s), 4)) for e, s in ranking]

    def recuperar(self, pergunta, top_k=3, multi=2):
        """Passagens **com o texto** -- o que o `AxonSystem.answer` joga fora."""
        with self._trava:
            sistema = self.sistema
        if sistema is None:
            return None, []
        vencedor, _ = sistema.route(pergunta)
        if vencedor is None:
            return None, []
        _, passagens = vencedor.retrieve(pergunta, top_k=top_k, multi=multi)
        return vencedor.name, [
            {"texto": t, "caminho": list(c), "score": round(float(s), 4)}
            for t, c, s in passagens
        ]
