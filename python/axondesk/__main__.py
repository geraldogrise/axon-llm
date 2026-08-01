"""Ponto de entrada: `python -m axondesk`.

Imprime **uma** linha no stdout -- o aperto de mão com porta e token -- e nunca mais
escreve ali. Todo o resto vai para stderr, senão qualquer print solto corrompe o
protocolo que o Electron está lendo.
"""

import argparse
import json
import os
import secrets
import sys
import threading
import time

from . import HANDSHAKE, engine, llm, server, sse


def _pai_vivo(pid):
    """O processo `pid` ainda existe?"""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    import ctypes
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    k = ctypes.windll.kernel32
    h = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    try:
        codigo = ctypes.c_ulong()
        if not k.GetExitCodeProcess(h, ctypes.byref(codigo)):
            return False
        return codigo.value == STILL_ACTIVE
    finally:
        k.CloseHandle(h)


def vigiar_pai(servidor, pid, intervalo=2.0):
    """Encerra quando o Electron morre -- inclusive se for morto sem aviso.

    Sonda o PID em vez de bloquear numa leitura do stdin. A versão com stdin
    **congela o processo inteiro** neste projeto: uma thread parada em
    `sys.stdin.read()` (ou em `os.read` no descritor 0) somada ao `pyaxon` carregado
    faz o servidor aceitar conexões e nunca responder, sem erro nem log.

    Medido, quatro combinações, duas rodadas cada: vigia sem experts responde em 3,8 s;
    experts sem vigia, 4,4 s; **vigia + experts, estouro de tempo nas duas rodadas.**
    Um reprodutor mínimo sem o `pyaxon` não falha, então o gatilho é a convivência com o
    `_axon` (compilado com o OpenMP do MinGW) e não a leitura em si.

    Sondar o PID não bloqueia em I/O nenhuma e cobre os mesmos casos.
    """
    def alvo():
        while True:
            time.sleep(intervalo)
            if not _pai_vivo(pid):
                print(f"[axondesk] pai {pid} sumiu; encerrando",
                      file=sys.stderr, flush=True)
                threading.Thread(target=servidor.shutdown, daemon=True).start()
                return

    threading.Thread(target=alvo, daemon=True, name="vigia-pai").start()


def main(argv=None):
    p = argparse.ArgumentParser(prog="axondesk")
    p.add_argument("--experts", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "examples", "axon_lang_data"),
        help="diretório com as pastas de experts")
    p.add_argument("--apenas", default="", help="lista separada por vírgula (ex.: go_experts)")
    p.add_argument("--porta", type=int, default=0, help="0 = o sistema escolhe")
    p.add_argument("--modelo", default="qwen2.5-coder:7b-instruct")
    p.add_argument("--sem-aquecer", action="store_true")
    p.add_argument("--pai", type=int, default=0,
                   help="PID do processo pai; 0 = descobre sozinho")
    args = p.parse_args(argv)

    apenas = [s.strip() for s in args.apenas.split(",") if s.strip()] or None
    token = secrets.token_hex(16)

    # Tudo o que for pesado de importar, aqui, antes de existir servidor. Se ficar para
    # as threads, a primeira requisição trava atrás da importação do pyaxon e o servidor
    # aceita conexão sem nunca responder -- sem erro no log.
    import email.parser                                    # noqa: F401
    import http.client                                     # noqa: F401
    engine.importar_pyaxon()

    barramento = sse.Barramento()
    motor = engine.Motor(args.experts, apenas=apenas)
    motor.ao_progredir(barramento.publicar)

    servidor, porta = server.subir(motor, barramento, token, porta=args.porta)

    # A partir daqui o stdout é só protocolo.
    print(f"{HANDSHAKE} " + json.dumps(
        {"porta": porta, "token": token, "pid": os.getpid()}), flush=True)

    print(f"[axondesk] ouvindo em 127.0.0.1:{porta}", file=sys.stderr, flush=True)
    print(f"[axondesk] experts em {args.experts}", file=sys.stderr, flush=True)

    vigiar_pai(servidor, args.pai or os.getppid())
    motor.carregar_async()

    if not args.sem_aquecer:
        threading.Thread(target=llm.aquecer, args=(args.modelo,),
                         daemon=True, name="aquecer").start()

    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        servidor.server_close()
        print("[axondesk] encerrado", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
