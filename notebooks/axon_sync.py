"""Sincroniza os artefatos treinados entre o Drive de um Colab e o git.

O problema que isto resolve: cada sessão do Colab tem o Drive de uma conta, e os
experts, adapters e GGUF acabam espalhados por contas diferentes. O git vira o lugar
único -- sobe de qualquer Colab, baixa em qualquer outro.

Branches órfãs (sem histórico comum com a `main`), pra cada uma pesar só o que guarda:

    branch `experts`    <- MyDrive/axon_experts   router.*.json + kb.sparse.json.gz
    branch `adapters`   <- MyDrive/axon_lora      adapters PEFT + metrica.json
                        <- MyDrive/axon_gguf      convertidos pro Ollama

Uso:

    import axon_sync as sync
    sync.status()                  # o que existe no Drive e no git
    sync.subir("experts")          # Drive -> git
    sync.baixar("lora")            # git -> Drive

Precisa de um `GH_TOKEN` com permissão de escrita: clonar repo público é livre, mas
push exige autenticação. Guarde nos Secrets do Colab (ícone da chave).
"""

import os
import shutil
import subprocess
import sys

REPO_DADOS = "https://github.com/geraldogrise/treinamento.git"
RAIZ_TRABALHO = "/content/sync"

# tipo -> (pasta no Drive, branch no git, o que guarda)
# `lora` e `gguf` dividem a branch `adapters`, cada um na sua subpasta: são as duas
# formas do mesmo artefato (PEFT pro Colab, GGUF pro Ollama) e andam juntos.
ARTEFATOS = {
    "experts": ("axon_experts", "experts",  "router + base de conhecimento"),
    "lora":    ("axon_lora",    "adapters", "adapters LoRA (formato PEFT)"),
    "gguf":    ("axon_gguf",    "adapters", "adapters convertidos pro Ollama"),
}

# O GitHub rejeita arquivos acima de 100 MB e avisa acima de 50 MB. Barramos antes
# de tentar o push: um arquivo grande demais faz o push inteiro falhar depois de já
# ter subido tudo o que vinha antes.
LIMITE_ERRO = 95 * 1024 ** 2
LIMITE_AVISO = 45 * 1024 ** 2


# Nomes de secret aceitos, em ordem. O nome do token no GitHub é só uma etiqueta e não
# importa aqui -- o que conta é como o secret foi chamado no Colab. Acrescente o seu
# nesta lista se usar outro: NOMES_SECRET.insert(0, "meu_nome").
NOMES_SECRET = ["GH_TOKEN", "GITHUB_TOKEN", "github_geraldo_grise", "gh_token"]

_origem_token = None


def _token():
    """Token do GitHub -- obrigatório pra subir (push), dispensável pra baixar."""
    global _origem_token
    for var in ("GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(var):
            _origem_token = _origem_token or f"variável de ambiente {var}"
            return os.environ[var]
    try:
        from google.colab import userdata
    except Exception:
        return None
    for nome in NOMES_SECRET:
        try:
            tok = userdata.get(nome)          # levanta se o secret não existir
        except Exception:
            continue
        if tok:
            os.environ["GH_TOKEN"] = tok
            _origem_token = f"secret '{nome}' do Colab"
            return tok
    return None


def verificar_token():
    """Diz se achou o token e de onde veio. Nunca imprime o valor."""
    tok = _token()
    if tok:
        print(f"token ok -- lido do {_origem_token} ({len(tok)} caracteres)")
        return True
    print("token NÃO encontrado. Você consegue baixar, mas não subir.\n"
          f"Procurei pelos secrets: {', '.join(NOMES_SECRET)}\n"
          "Renomeie o seu secret pra GH_TOKEN, ou rode antes:\n"
          "    import os; from google.colab import userdata\n"
          "    os.environ['GH_TOKEN'] = userdata.get('NOME_DO_SEU_SECRET')")
    return False


def _limpa(texto):
    tok = os.environ.get("GH_TOKEN")
    return texto.replace(tok, "***") if tok and texto else texto


def _url(com_token=False):
    tok = _token() if com_token else None
    if not tok:
        return REPO_DADOS
    return REPO_DADOS.replace("https://", f"https://x-access-token:{tok}@", 1)


def _git(args, cwd, checar=True):
    p = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    if checar and p.returncode != 0:
        raise RuntimeError(_limpa(f"git {' '.join(args)}\n{p.stdout}\n{p.stderr}"))
    return p


def _drive(pasta):
    """Caminho da pasta no Drive, montando o Drive se preciso."""
    if not os.path.ismount("/content/drive"):
        from google.colab import drive
        drive.mount("/content/drive")
    caminho = os.path.join("/content/drive/MyDrive", pasta)
    os.makedirs(caminho, exist_ok=True)
    return caminho


def _checa(tipo):
    if tipo not in ARTEFATOS:
        raise KeyError(f"tipo '{tipo}' não existe. Use: {', '.join(ARTEFATOS)}")
    return ARTEFATOS[tipo]


def _branch_existe(branch):
    p = subprocess.run(["git", "ls-remote", "--heads", _url(), branch],
                       capture_output=True, text=True)
    return bool(p.stdout.strip())


def _preparar(branch, escrita=False):
    """Clone raso da branch em /content/sync/<branch>. Cria a branch órfã se faltar."""
    destino = os.path.join(RAIZ_TRABALHO, branch)
    if os.path.isdir(destino):
        shutil.rmtree(destino)
    os.makedirs(RAIZ_TRABALHO, exist_ok=True)

    url = _url(com_token=escrita)
    if escrita and not _token():
        raise RuntimeError(
            "subir exige um GH_TOKEN com permissão de escrita. Crie o secret GH_TOKEN "
            "nos Secrets do Colab (ícone da chave) e ligue o acesso ao notebook.")

    if _branch_existe(branch):
        _git(["clone", "--depth", "1", "--branch", branch, url, destino], cwd="/content")
    else:
        print(f"branch '{branch}' ainda não existe -- criando", flush=True)
        _git(["clone", "--depth", "1", url, destino], cwd="/content")
        _git(["checkout", "--orphan", branch], cwd=destino)
        _git(["rm", "-rf", "--quiet", "."], cwd=destino, checar=False)
        for resto in os.listdir(destino):
            if resto != ".git":
                caminho = os.path.join(destino, resto)
                shutil.rmtree(caminho) if os.path.isdir(caminho) else os.remove(caminho)

    _git(["config", "user.email", "colab@axon.local"], cwd=destino)
    _git(["config", "user.name", "axon colab"], cwd=destino)
    return destino


def _medir(raiz):
    """(total de bytes, nº de arquivos, os que passam dos limites do GitHub)."""
    total, n, grandes = 0, 0, []
    for base, _, arquivos in os.walk(raiz):
        if ".git" in base.split(os.sep):
            continue
        for nome in arquivos:
            fp = os.path.join(base, nome)
            tam = os.path.getsize(fp)
            total += tam
            n += 1
            if tam > LIMITE_AVISO:
                grandes.append((os.path.relpath(fp, raiz), tam))
    return total, n, sorted(grandes, key=lambda x: -x[1])


def status():
    """O que existe no Drive desta conta e o que já está no git."""
    print(f"{'tipo':<10} {'Drive':<34} {'git':<10} conteúdo")
    print("-" * 84)
    for tipo, (pasta, branch, descricao) in ARTEFATOS.items():
        local = os.path.join("/content/drive/MyDrive", pasta)
        if os.path.isdir(local):
            total, n, _ = _medir(local)
            no_drive = f"{n} arquivos · {total / 1e6:.0f} MB"
        else:
            no_drive = "(vazio)"
        print(f"{tipo:<10} {no_drive:<34} "
              f"{'existe' if _branch_existe(branch) else '—':<10} {descricao}")


def subir(tipo, mensagem=None, forcar=False):
    """Drive -> git. Copia a pasta pra branch do tipo, commita e faz push."""
    pasta, branch, _ = _checa(tipo)
    origem = _drive(pasta)

    total, n, grandes = _medir(origem)
    if n == 0:
        print(f"{pasta} está vazia no Drive desta conta -- nada a subir")
        return
    print(f"{pasta}: {n} arquivos, {total / 1e6:.0f} MB", flush=True)

    bloqueados = [(f, t) for f, t in grandes if t > LIMITE_ERRO]
    if bloqueados and not forcar:
        print("\nO GitHub rejeita arquivos acima de 100 MB. Estes não passam:")
        for f, t in bloqueados:
            print(f"  {t / 1e6:6.0f} MB  {f}")
        raise RuntimeError("push abortado -- remova os arquivos acima ou use "
                           "outro destino (o Hugging Face Hub aceita modelos grandes)")
    for f, t in grandes:
        print(f"  aviso: {t / 1e6:.0f} MB  {f}  (acima dos 50 MB que o GitHub avisa)")

    repo = _preparar(branch, escrita=True)
    destino = os.path.join(repo, pasta)
    if os.path.isdir(destino):
        shutil.rmtree(destino)
    shutil.copytree(origem, destino)

    _git(["add", "-A"], cwd=repo)
    if not _git(["status", "--porcelain"], cwd=repo).stdout.strip():
        print("nada mudou em relação ao que já está no git")
        return

    _git(["commit", "-q", "-m", mensagem or f"sync {tipo}: {n} arquivos"], cwd=repo)
    _git(["push", "-q", "--set-upstream", "origin", branch], cwd=repo)
    print(f"\nsubiu pra branch '{branch}' de {REPO_DADOS}")


def baixar(tipo, sobrescrever=False):
    """git -> Drive. Traz a branch do tipo e copia pra pasta do Drive desta conta."""
    pasta, branch, _ = _checa(tipo)
    if not _branch_existe(branch):
        print(f"branch '{branch}' ainda não existe -- nada foi subido ainda")
        return

    repo = _preparar(branch)
    origem = os.path.join(repo, pasta)
    if not os.path.isdir(origem):
        print(f"a branch '{branch}' existe mas não tem a pasta {pasta}")
        return

    destino = _drive(pasta)
    novos, existentes = 0, 0
    for base, _, arquivos in os.walk(origem):
        rel = os.path.relpath(base, origem)
        alvo = destino if rel == "." else os.path.join(destino, rel)
        os.makedirs(alvo, exist_ok=True)
        for nome in arquivos:
            fp = os.path.join(alvo, nome)
            if os.path.exists(fp) and not sobrescrever:
                existentes += 1
                continue
            shutil.copy2(os.path.join(base, nome), fp)
            novos += 1

    print(f"{novos} arquivos copiados pra {destino}")
    if existentes:
        print(f"{existentes} já existiam e foram mantidos "
              f"(use sobrescrever=True pra trocar)")
