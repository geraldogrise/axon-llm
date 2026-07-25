"""Gera dados de treino automaticamente usando a API do Claude (ou GPT).

Para cada subsetor da taxonomia, o script:
  1) pede ao modelo uma lista de TÓPICOS DISTINTOS (o filtro "distinto" é isto);
  2) gera uma lição escolar original em português para cada tópico novo;
  3) grava em treinamento_portugues/<area>/<setor>/<subsetor>/tema_<slug>.md

É RESUMÍVEL: pula arquivos que já existem e mantém um registro de tópicos já usados
(topics_ledger.json) para nunca repetir assunto entre execuções.

Uso:
    # Claude (Anthropic)
    pip install anthropic
    set ANTHROPIC_API_KEY=sk-ant-...        (Windows: set / PowerShell: $env:ANTHROPIC_API_KEY=)
    python tools/gerar_dados.py

    # Variáveis opcionais:
    #   AXON_PROVIDER=anthropic|openai   (default anthropic)
    #   AXON_MODEL=claude-sonnet-5       (ou gpt-4o para openai)
    #   AXON_N=30                        (tópicos distintos por subsetor)
    #   AXON_DELAY=1.0                   (pausa em s entre chamadas)

Depois de gerar, treine com:  python examples/train_from_local.py
"""

import json
import os
import re
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "treinamento_portugues")
LEDGER = os.path.join(BASE, "topics_ledger.json")

PROVIDER = os.environ.get("AXON_PROVIDER", "anthropic")
MODEL = os.environ.get("AXON_MODEL", "claude-sonnet-5" if PROVIDER == "anthropic" else "gpt-4o")
N_PER_SUB = int(os.environ.get("AXON_N", 30))
DELAY = float(os.environ.get("AXON_DELAY", 1.0))

# Taxonomia: area -> setor -> [subsetores]
TAXO = {
    "matematica": {"algebra": ["linear", "abstrata"], "calculo": ["derivadas", "integrais"],
                   "geometria": ["plana", "analitica"]},
    "fisica": {"mecanica": ["classica", "quantica"], "termo": ["calor", "entropia"],
               "eletro": ["eletricidade", "magnetismo"]},
    "biologia": {"genetica": ["mendel", "molecular"], "citologia": ["celula", "energia"],
                 "ecologia": ["ecossistemas", "biomas"]},
    "quimica": {"geral": ["atomo", "ligacoes"], "organica": ["hidrocarbonetos", "funcoes"]},
    "portugues": {"gramatica": ["sintaxe", "morfologia"], "literatura": ["autores", "movimentos"]},
    "historia": {"brasil": ["colonia", "republica"], "geral": ["antiga", "moderna"]},
}


# --- backend do modelo (Claude ou GPT) --------------------------------------
def _make_client():
    if PROVIDER == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    import openai
    return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


_CLIENT = None


def ask(prompt, max_tokens=1800):
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _make_client()
    if PROVIDER == "anthropic":
        r = _CLIENT.messages.create(model=MODEL, max_tokens=max_tokens,
                                    messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    r = _CLIENT.chat.completions.create(model=MODEL, max_tokens=max_tokens,
                                        messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content


# --- geração ----------------------------------------------------------------
def slug(s):
    s = re.sub(r"[àáâãä]", "a", s.lower())
    s = re.sub(r"[éêẽ]", "e", s); s = re.sub(r"[íî]", "i", s)
    s = re.sub(r"[óôõ]", "o", s); s = re.sub(r"[úû]", "u", s); s = re.sub(r"ç", "c", s)
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:60] or "tema"


def list_topics(area, setor, sub, n, ja_usados):
    evitar = ", ".join(sorted(ja_usados)[:40]) or "(nenhum ainda)"
    p = (f"Liste {n} tópicos DISTINTOS e específicos da matéria de {area}, área {setor}/{sub}, "
         f"para lições escolares em português do Brasil. Um por linha, só o título curto do tópico, "
         f"sem numeração e sem repetir estes que já foram usados: {evitar}. "
         f"Se não existirem {n} tópicos genuinamente distintos, liste menos — não invente repetições.")
    text = ask(p, 700)
    out = []
    for line in text.splitlines():
        t = line.strip().lstrip("-•*0123456789. ").strip()
        if t and slug(t) not in ja_usados and t.lower() not in (x.lower() for x in out):
            out.append(t)
    return out[:n]


def lesson(area, setor, sub, topico):
    p = (f"Escreva uma lição escolar ORIGINAL em português do Brasil sobre \"{topico}\" "
         f"(matéria: {area}, área {setor}/{sub}). De 800 a 1000 palavras, começando com um título "
         f"markdown (#). Conteúdo correto, didático e rico em termos técnicos específicos do tema. "
         f"Escreva só a lição, sem comentários seus.")
    return ask(p, 2000)


def load_ledger():
    if os.path.exists(LEDGER):
        with open(LEDGER, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_ledger(s):
    os.makedirs(BASE, exist_ok=True)
    with open(LEDGER, "w", encoding="utf-8") as f:
        json.dump(sorted(s), f, ensure_ascii=False)


def main():
    ledger = load_ledger()   # slugs de tópicos já gerados (dedup entre execuções)
    total = 0
    for area, setores in TAXO.items():
        for setor, subs in setores.items():
            for sub in subs:
                folder = os.path.join(BASE, area, setor, sub)
                os.makedirs(folder, exist_ok=True)
                usados_no_sub = {slug(x) for x in ledger}
                try:
                    topicos = list_topics(area, setor, sub, N_PER_SUB, usados_no_sub)
                except Exception as e:  # noqa: BLE001
                    print(f"[{area}/{setor}/{sub}] falha ao listar tópicos: {e}", flush=True)
                    continue
                print(f"[{area}/{setor}/{sub}] {len(topicos)} tópicos distintos", flush=True)
                for topico in topicos:
                    sl = slug(topico)
                    fp = os.path.join(folder, f"tema_{sl}.md")
                    if sl in ledger or os.path.exists(fp):
                        continue                       # resume / dedup
                    try:
                        texto = lesson(area, setor, sub, topico)
                    except Exception as e:  # noqa: BLE001
                        print(f"    erro em '{topico}': {e}", flush=True)
                        continue
                    with open(fp, "w", encoding="utf-8") as f:
                        f.write(texto)
                    ledger.add(sl)
                    total += 1
                    print(f"    + {topico}", flush=True)
                    if total % 10 == 0:
                        save_ledger(ledger)
                    time.sleep(DELAY)
    save_ledger(ledger)
    print(f"\nfim: {total} lições novas geradas | ledger: {len(ledger)} tópicos", flush=True)


if __name__ == "__main__":
    main()
