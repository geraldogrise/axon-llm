"""Helpers compartilhados pelos notebooks do Colab.

Um lugar só com o mapa "expert -> branch de dados -> script de build", pra os
notebooks não repetirem essa tabela. Importável sem o `_axon` compilado (não
importa `pyaxon`), então serve tanto no notebook de treinar expert quanto no de
fine-tune.

Uso típico no Colab:

    import sys; sys.path.insert(0, "axon-llm/notebooks")
    import axon_colab as ac

    d = ac.fetch_data("go")            # clona a branch fase-8 e devolve o caminho
    ac.build_expert("go")              # roda examples/build_go_experts.py
    licoes = list(ac.iter_licoes(d))   # pra montar o dataset do QLoRA
"""

import os
import subprocess
import sys

CODE_REPO = "https://github.com/geraldogrise/axon-llm.git"
DATA_REPO = "https://github.com/geraldogrise/treinamento.git"

# expert -> (branch, subpasta dentro do clone, env var do script, script de build, rótulo)
# A subpasta é o nome da pasta como ela existe na branch (conferido com `git ls-tree`).
EXPERTS = {
    "escolar":    ("fase-1",  "treinamento_portugues",         "AXON_LESSONS_DIR", "build_final_model.py",       "escolar (mat/física/bio/química/português/história)"),
    "java":       ("fase-2",  "treinamento_programacao/java",  "AXON_JAVA_DIR",    "build_java_experts.py",      "Java"),
    "dotnet":     ("fase-3",  "treinamento_net",               "AXON_NET_DIR",     "build_dotnet_experts.py",    ".NET / C#"),
    "js":         ("fase-4",  "treinamento_js",                "AXON_JS_DIR",      "build_js_experts.py",        "JavaScript / TypeScript"),
    "python":     ("fase-5",  "treinamento_python",            "AXON_PY_DIR",      "build_python_experts.py",    "Python"),
    "php":        ("fase-6",  "treinamento_php",               "AXON_PHP_DIR",     "build_php_experts.py",       "PHP"),
    "rust":       ("fase-7",  "treinamento_rust",              "AXON_RUST_DIR",    "build_rust_experts.py",      "Rust"),
    "go":         ("fase-8",  "treinamento_go",                "AXON_GO_DIR",      "build_go_experts.py",        "Go"),
    "ruby":       ("fase-9",  "treinamento_ruby",              "AXON_RUBY_DIR",    "build_ruby_experts.py",      "Ruby / Rails"),
    "aws":        ("fase-10", "treinamento_aws",               "AXON_AWS_DIR",     "build_aws_experts.py",       "AWS"),
    "azure":      ("fase-10", "treinamento_azure",             "AXON_AZURE_DIR",   "build_azure_experts.py",     "Azure"),
    "gcp":        ("fase-10", "treinamento_gcp",               "AXON_GCP_DIR",     "build_gcp_experts.py",       "Google Cloud"),
    "oci":        ("fase-10", "treinamento_oci",               "AXON_OCI_DIR",     "build_oci_experts.py",       "Oracle Cloud"),
    "bash":       ("fase-11", "treinamento_bash",              "AXON_BASH_DIR",    "build_bash_experts.py",      "Bash"),
    "docker":     ("fase-11", "treinamento_docker",            "AXON_DOCKER_DIR",  "build_docker_experts.py",    "Docker"),
    "git":        ("fase-11", "treinamento_git",               "AXON_GIT_DIR",     "build_git_experts.py",       "Git"),
    "kubernetes": ("fase-11", "treinamento_kubernetes",        "AXON_KUBERNETES_DIR", "build_kubernetes_experts.py", "Kubernetes"),
    "shell":      ("fase-11", "treinamento_shell",             "AXON_SHELL_DIR",   "build_shell_experts.py",     "Shell"),
    "web":        ("fase-12", "treinamento_web",               "AXON_WEB_DIR",     "build_web_experts.py",       "Web (HTML/CSS)"),
}

# Onde cada script grava o router + KB (relativo a examples/axon_lang_data/).
SAIDA = {k: ("rag_final" if k == "escolar" else f"{k}_experts") for k in EXPERTS}

# Modelos-base testados na T4 grátis (16 GB) via Unsloth em 4-bit.
BASES = {
    "deepseek": "unsloth/deepseek-coder-6.7b-instruct-bnb-4bit",
    "qwen":     "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
    "qwen-3b":  "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",  # folga se a T4 apertar
}


def _run(cmd, **kw):
    """Roda um comando ecoando a saída (o Colab só mostra o que sai na hora)."""
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, capture_output=True, **kw)
    if p.stdout:
        print(p.stdout, flush=True)
    if p.returncode != 0:
        print(p.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"falhou ({p.returncode}): {' '.join(cmd)}")
    return p.stdout


def check(expert):
    if expert not in EXPERTS:
        raise KeyError(f"expert '{expert}' não existe. Disponíveis: {', '.join(sorted(EXPERTS))}")
    return EXPERTS[expert]


def fetch_code(dest="axon-llm"):
    """Clona o repo de código (idempotente)."""
    if not os.path.isdir(dest):
        _run(["git", "clone", "--depth", "1", CODE_REPO, dest])
    return os.path.abspath(dest)


def fetch_data(expert, raiz="dados"):
    """Clona só a branch de dados do expert e devolve a pasta das lições.

    Uma branch por fase e `--depth 1`, então baixa só o que esse expert precisa.
    """
    branch, sub, _, _, _ = check(expert)
    dest = os.path.join(raiz, branch)
    if not os.path.isdir(dest):
        _run(["git", "clone", "--depth", "1", "--branch", branch, DATA_REPO, dest])
    caminho = os.path.join(dest, *sub.split("/"))
    if not os.path.isdir(caminho):
        raise FileNotFoundError(f"'{caminho}' não existe na branch {branch}")
    n = sum(len(fs) for _, _, fs in os.walk(caminho))
    print(f"{expert}: {n} arquivos em {caminho}", flush=True)
    return os.path.abspath(caminho)


def build_expert(expert, repo="axon-llm", extra_env=None):
    """Roda o build_*_experts.py do expert apontando pros dados clonados."""
    _, _, env_var, script, rotulo = check(expert)
    dados = fetch_data(expert)
    env = dict(os.environ)
    env[env_var] = dados
    # O escolar puxaria passagens da Wikipédia de um rag_multi/ que não existe no
    # Colab; 0 = treina só com as lições locais.
    if expert == "escolar":
        env.setdefault("AXON_WIKI_CAP", "0")
    if extra_env:
        env.update(extra_env)
    env["PYTHONPATH"] = os.path.join(os.path.abspath(repo), "python") + os.pathsep + env.get("PYTHONPATH", "")
    print(f"\n=== treinando expert: {rotulo} ===", flush=True)
    _run([sys.executable, os.path.join("examples", script)], cwd=os.path.abspath(repo), env=env)
    saida = os.path.join(os.path.abspath(repo), "examples", "axon_lang_data", SAIDA[expert])
    print(f"artefatos em: {saida}", flush=True)
    return saida


def iter_licoes(pasta, min_chars=200):
    """Percorre <família>/<subsetor>/*.md -> (partes_do_path, titulo, corpo)."""
    for raiz, _, arquivos in os.walk(pasta):
        for nome in sorted(arquivos):
            if not nome.endswith(".md"):
                continue
            fp = os.path.join(raiz, nome)
            partes = os.path.relpath(fp, pasta).replace("\\", "/").split("/")[:-1]
            if not partes:
                continue
            with open(fp, encoding="utf-8") as f:
                texto = f.read().strip()
            if len(texto) < min_chars:
                continue
            titulo = next((l.lstrip("# ").strip() for l in texto.splitlines()
                           if l.startswith("#")), partes[-1].replace("_", " "))
            yield partes, titulo, texto


def _secoes(texto, max_chars):
    """Quebra a lição nas seções '## ' quando ela passa do limite de contexto."""
    if len(texto) <= max_chars:
        return [texto]
    blocos, atual = [], []
    for linha in texto.splitlines(keepends=True):
        if linha.startswith("## ") and atual and sum(map(len, atual)) > max_chars // 3:
            blocos.append("".join(atual))
            atual = []
        atual.append(linha)
    if atual:
        blocos.append("".join(atual))
    return [b[:max_chars] for b in blocos if len(b) >= 200]


def sft_examples(pasta, tokenizer, rotulo, max_chars=6000):
    """Lições .md -> exemplos de chat pro SFT.

    A pergunta cita a linguagem e o subsetor (que vêm do path), não só o título:
    é o que faz o modelo aprender a *responder sobre aquele assunto* em vez de
    completar o texto da lição.
    """
    exemplos = []
    for partes, titulo, corpo in iter_licoes(pasta):
        assunto = " > ".join(p.replace("_", " ") for p in partes)
        for i, bloco in enumerate(_secoes(corpo, max_chars)):
            pergunta = f"Em {rotulo} ({assunto}), explique com exemplos de código: {titulo}"
            if i:
                pergunta += " (continuação)"
            msgs = [{"role": "user", "content": pergunta},
                    {"role": "assistant", "content": bloco.strip()}]
            exemplos.append({"text": tokenizer.apply_chat_template(msgs, tokenize=False)})
    return exemplos


def drive_dir(*partes, montar=True):
    """Caminho no Drive, montando se preciso. Sessão do Colab cai; Drive não."""
    if montar and not os.path.ismount("/content/drive"):
        from google.colab import drive
        drive.mount("/content/drive")
    caminho = os.path.join("/content/drive/MyDrive", *partes)
    os.makedirs(caminho, exist_ok=True)
    return caminho


def salvar_expert(saida, expert, pasta_drive="axon_experts"):
    """Copia router + KB do expert pro Drive. A sessão do Colab cai; o Drive não."""
    import shutil
    dst = drive_dir(pasta_drive, SAIDA[expert])
    total = 0
    for nome in sorted(os.listdir(saida)):
        origem = os.path.join(saida, nome)
        if os.path.isfile(origem):
            shutil.copy2(origem, os.path.join(dst, nome))
            total += os.path.getsize(origem)
    print(f"salvo no Drive: {dst} ({total / 1e6:.1f} MB)", flush=True)
    return dst


def ultimo_checkpoint(pasta):
    """Último `checkpoint-<n>` de um output_dir do Trainer, ou None.

    É o que permite retomar o QLoRA de onde parou quando a sessão do Colab cai
    no meio do treino.
    """
    if not os.path.isdir(pasta):
        return None
    cks = [d for d in os.listdir(pasta)
           if d.startswith("checkpoint-") and d[len("checkpoint-"):].isdigit()]
    if not cks:
        return None
    ultimo = max(cks, key=lambda d: int(d[len("checkpoint-"):]))
    return os.path.join(pasta, ultimo)


def lora_dir(base, expert, pasta_drive="axon_lora"):
    """Pasta do adapter LoRA no Drive: axon_lora/<base>/<expert>/.

    Um adapter por expert e por modelo-base, então DeepSeek e Qwen convivem e
    dá pra comparar os dois no mesmo expert.
    """
    return drive_dir(pasta_drive, base, expert)


def tabela():
    """Imprime o mapa completo (qual branch/script cada expert usa)."""
    print(f"{'expert':<12} {'branch':<9} {'dados':<32} {'script'}")
    for k in sorted(EXPERTS):
        branch, sub, _, script, _ = EXPERTS[k]
        print(f"{k:<12} {branch:<9} {sub:<32} {script}")
