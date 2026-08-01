"""Execução de comandos para o painel axon-code.

Versão pequena, para ver o mecanismo funcionando: roda o comando, transmite a saída
linha a linha e informa o código de saída. Sem checkpoint de git e sem classificação
ainda -- isso entra quando o painel estiver provado.

O ambiente não interativo não é detalhe: sem ele o `git log` abre o paginador e o
comando trava até o tempo estourar, toda vez.
"""

import os
import subprocess
import sys
import threading

TIMEOUT = 120

# Sem isto, qualquer ferramenta que queira paginar ou perguntar algo trava o painel.
AMBIENTE = {
    "GIT_PAGER": "cat", "PAGER": "cat", "GIT_TERMINAL_PROMPT": "0",
    "GIT_ASKPASS": "echo", "NO_COLOR": "1", "TERM": "dumb", "CI": "1",
    "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
}


def _matar_arvore(proc):
    """No Windows, `kill()` não mata os netos -- e `go test` e `npm` sempre criam netos."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        proc.kill()


def executar(comando, cwd, ao_sair, timeout=TIMEOUT):
    """Roda `comando`, chamando `ao_sair(tipo, dados)` a cada linha e no fim.

    `tipo` é "saida" ou "fim".
    """
    env = dict(os.environ, **AMBIENTE)
    if os.name == "nt":
        argv = ["powershell.exe", "-NoProfile", "-NonInteractive",
                "-ExecutionPolicy", "Bypass", "-Command", comando]
    else:
        argv = ["/bin/sh", "-c", comando]

    try:
        proc = subprocess.Popen(
            argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
    except Exception as erro:                              # noqa: BLE001
        ao_sair("fim", {"codigo": -1, "erro": f"{type(erro).__name__}: {erro}"})
        return

    estourou = {"sim": False}

    def cortar():
        estourou["sim"] = True
        _matar_arvore(proc)

    alarme = threading.Timer(timeout, cortar)
    alarme.start()
    try:
        for linha in proc.stdout:
            ao_sair("saida", {"linha": linha.rstrip("\n")})
        proc.wait()
    finally:
        alarme.cancel()

    ao_sair("fim", {"codigo": proc.returncode,
                    "estourou": estourou["sim"], "timeout": timeout})


def parece_comando(texto):
    """O primeiro token existe como executável? Então roda direto, sem passar pelo LLM.

    `git status` tem que ser instantâneo, não uma ida de 15 s ao modelo.
    """
    texto = texto.strip()
    if not texto:
        return False
    primeiro = texto.split()[0]
    if primeiro in {"cd", "ls", "dir", "echo", "type", "cat", "pwd", "cls", "clear"}:
        return True
    import shutil
    return shutil.which(primeiro) is not None
