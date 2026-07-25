# FUTURE — o que falta pro AxonLM ficar mais inteligente

Avaliação honesta do estado atual e do roadmap pra evoluir de "assistente de
documentação" pra "copiloto de programação". Serve de guia das próximas melhorias.

---

## O que ele JÁ consegue fazer (de verdade)

O sistema hoje é um **assistente RAG** (`ax.system.AxonSystem`):
**roteia** a pergunta pro expert certo → **recupera** a lição relevante → **responde**.

Funciona bem para:
- **tirar dúvidas conceituais** ("o que é `useState`?", "como funciona o Eloquent?");
- **mostrar exemplos** dos tópicos presentes nas lições (6 domínios: escolar, Java, .NET,
  JS/TS, Python, PHP — ~1.900 lições + fases planejadas em `train/`);
- com **Ollama ligado**, reescrever a resposta de forma fluente **fundamentada na lição**
  (grounded, não inventa).

### Limite honesto
Ele **recupera e reformula conhecimento existente**. Ele **não raciocina** nem **gera
código novo/inédito** sozinho — o AxonLM (o transformer treinado do zero) é pequeno demais
pra isso. A "inteligência" da resposta fluente vem hoje do **LLM externo** (llama via
Ollama), não do AxonLM. O AxonLM/experts fazem o **roteamento + recuperação**; o LLM faz a
**geração**.

---

## O que mais faria diferença — em ordem de impacto

### 1. Modelo de geração é o cérebro (maior ganho, menor esforço)
Trocar o `llama3.2` genérico por um **modelo de código** no Ollama muda tudo:
`qwen2.5-coder`, `deepseek-coder-v2`, `codellama`. Eles geram/debugam código de verdade, e
o RAG dá o contexto correto (reduz alucinação). Só isso já entrega um copiloto utilizável.
> Onde mexer: `python/pyaxon/generate.py` (`DEFAULT_MODEL`) e o `AxonSystem.answer(model=...)`.

### 2. Melhorar a recuperação (o que alimenta o cérebro)
- **Reranking**: recuperar ~20 passagens e reordenar com um cross-encoder → pega o trecho
  *certo*, não só o "parecido".
- **Embeddings reais** (sentence-transformers) além do LSA → captura sentido melhor.
- **Mais contexto**: passar top-3/5 lições ao LLM, não só a primeira (o extractive hoje
  mostra só a #1).
- Ajustar tamanho de chunk e metadados (linguagem/versão) em `SparseKB`.

### 3. Dados melhores, não só mais dados
- **Pares pergunta→resposta** e **erro→solução** (como as pessoas realmente pesquisam).
- **Código que roda** nos exemplos.
- As fases planejadas (Rust/Go/cloud/etc. em `train/`) aumentam a *largura*; Q&A aumenta a
  *profundidade*.

### 4. Verificação (isso deixa "inteligente de verdade")
Fazer o sistema **executar/testar** o código que sugere (rodar, linter, testes) e corrigir
se falhar. Um assistente que *verifica* a própria resposta erra muito menos.

### 5. Conversa e memória
Hoje cada pergunta é isolada. Guardar o **histórico da conversa** (follow-up: "e como faço
isso com async?") deixa muito mais útil.

### 6. Avaliar a qualidade da RESPOSTA, não só do roteamento
Já medimos roteamento (100% família / gate de domínio). Falta medir se a *resposta* ajudou
(conjunto de perguntas reais com gabarito). Sem isso, não dá pra saber se melhorou.

---

## O teto realista (a parte honesta)

Dois caminhos pra "ficar inteligente" — escolher conscientemente:

| Caminho | O que é | Viável? |
|---|---|---|
| **RAG + LLM de código** (recomendado) | Os experts recuperam; um Qwen/DeepSeek-Coder local gera. | ✅ Hoje, no PC/Colab. Melhor custo-benefício. |
| **Treinar o AxonLM pra gerar sozinho** | Escalar o transformer + fine-tune/distillation de um modelo aberto com os dados. | ⚠️ Precisa GPU e muito mais dado; projeto grande (o CUDA já feito é o 1º passo). |

**Resumo:** com o que existe, já é um bom **assistente de estudo/documentação**; com um
modelo de código no Ollama + reranking + top-k, vira um **copiloto de programação** decente
e *offline*. Pra ele "pensar" e escrever código novo com confiança, o caminho prático é
**RAG + um modelo de código aberto** (fine-tunado nos dados depois), **não** escalar o
AxonLM do zero.

---

## Recomendação de próximo passo
Começar por **#1 + #2**: plugar `qwen2.5-coder` no `generate.py` + **reranking e top-k** no
RAG. É o maior salto de qualidade com o que já está pronto, e não quebra nada do existente.

### Ordem sugerida do roadmap "inteligência"
1. Modelo de código no Ollama (`generate.py`).
2. Reranking + top-k + mais contexto no RAG (`vindex.py`/`system.py`).
3. Embeddings reais (sentence-transformers) opcionais além do LSA.
4. Memória de conversa no `AxonSystem`.
5. Verificação/execução de código (sandbox + testes).
6. Dataset de Q&A e erro→solução.
7. Avaliação de qualidade de resposta (não só roteamento).
8. (Longo prazo) fine-tune/distillation de um modelo de código aberto nos dados.

---

## ✅ FEITO (2026-07-22) — 5 experts de ferramentas/DevOps (git, bash, shell, docker, kubernetes)

Criados 5 experts **separados** (um por ferramenta, igual às nuvens: 1 expert = 1 domínio),
no MESMO modelo de arquivos das nuvens (`treinamento_<expert>/<família>/<subsetor…>/*.md`),
na pasta principal do `pyaxon/` e replicados no repo de dados irmão `../treinamento/`.

**Dados moram no repo irmão** `C:\grisecorp\repositories\treinamento\treinamento_<expert>\`
(o usuário moveu as pastas do `pyaxon/` pra lá; o build usa esse caminho via fallback `_DATA`).

**Expandido (2026-07-22): 81 → 167 lições**, equilibrando toda família em ~6 lições
(as famílias magras de 1–3 foram a causa do `evaluate.py` baixo — ver §métricas).

| Expert | Lições | Famílias | Build script | Auto-check | evaluate.py (macro-F1) |
|---|---:|---:|---|---|---|
| `git_experts` | 35 | 6 | `examples/build_git_experts.py` | **9/9 = 100%** | 72,7% / **66,1%** |
| `bash_experts` | 24 | 4 | `examples/build_bash_experts.py` | **7/8 = 88%** | 87,5% / **86,7%** |
| `shell_experts` | 30 | 5 | `examples/build_shell_experts.py` | **9/9 = 100%** | 80,0% / **80,0%** |
| `docker_experts` | 36 | 6 | `examples/build_docker_experts.py` | 5/8 = 62% | 66,7% / **68,9%** |
| `kubernetes_experts` | 42 | 7 | `examples/build_kubernetes_experts.py` | **9/9 = 100%** | 78,6% / **77,6%** |

- **Famílias** — git: fundamentos, branches-e-merge, remotos, desfazer-e-historia, colaboracao,
  avancado · bash: fundamentos, controle-de-fluxo, funcoes-e-dados, robustez · shell:
  navegacao-e-arquivos, inspecao-e-busca, pipes-e-redirecionamento, processamento-texto,
  processos-e-sistema · docker: fundamentos, imagens, runtime-dados, redes, compose,
  producao-seguranca · kubernetes: fundamentos, workloads, rede-servicos, config-storage,
  escalonamento, seguranca-rbac, ecossistema-operacoes.
- **Experts gerados** em `examples/axon_lang_data/{git,bash,shell,docker,kubernetes}_experts/`
  (`router.gate/*.json(.w)`, `kb.sparse.json.gz`, `router.manifest.json`). O `AxonSystem` os
  carrega automaticamente junto dos outros — nenhum código novo precisou.
- **Como rodou** (core `_axon.pyd`/`libaxon.a` já compilados — não recompilar):
  ```powershell
  $env:PYTHONPATH = "C:\grisecorp\repositories\pyaxon\python"
  cd C:\grisecorp\repositories\pyaxon\examples
  foreach ($e in "git","bash","shell","docker","kubernetes") { python "build_${e}_experts.py" }
  ```

### Leitura das métricas (antes vs. depois de encorpar as famílias)
- **Auto-check do build** (rota certa + `SparseKB` recupera a passagem pertinente):
  git 100% · bash 88% · shell 100% · docker 62% · k8s 100%. É um set pequeno de 8–9
  perguntas escolhidas a mão, então é ruidoso — o docker caiu (5/8) porque, com mais
  vocabulário, "containers vs VMs" e "segurança de imagens" roteiam pra vizinhos
  (`redes`/`imagens`); a métrica confiável é o `evaluate.py` abaixo.
- **`tools/evaluate.py` (split estratificado retido) — MELHOROU MUITO ao equilibrar as famílias.**
  Antes cada expert tinha famílias com 1–3 lições → teste retido com ~1 doc/família → 1 erro
  derrubava o F1. Com ~6 por família o teste tem ~2 docs/família e o macro-F1 subiu:

  | Expert | macro-F1 ANTES (81 lições) | macro-F1 DEPOIS (167) |
  |---|---|---|
  | git | 57,8% | **66,1%** |
  | bash | 10,0% | **86,7%** |
  | shell | 30,0% | **80,0%** |
  | docker | 27,8% | **68,9%** |
  | kubernetes | 34,3% | **77,6%** |

  Confirma o diagnóstico do §7 (Ruby): o número baixo era artefato de famílias magras, não
  de qualidade das lições.

### 🔑 DESCOBERTA (2026-07-23): reestruturar famílias >> adicionar conteúdo
Ao tentar levar git e docker a 90%, encontramos que **adicionar mais lições NÃO ajuda quando as
famílias se sobrepõem** — e pode piorar. O git, expandido a 10/família (60 lições), CAIU pra
**49% macro-F1**, porque famílias como `avancado` (grab-bag: submódulos, hooks, LFS, subtree,
bundle, performance...) e `colaboracao` (PR/tags/review, que usa vocabulário de push/remote)
**colidem** com as vizinhas. As nuvens roteiam 93–100% porque "aws" vs "gcp" têm vocabulário
MUITO distinto; dentro de UMA ferramenta as famílias são mais próximas.

**A alavanca real é a DISTINÇÃO SEMÂNTICA das famílias, não o volume.** Reestruturando:

| Expert | famílias antes | famílias depois | macro-F1 |
|---|---|---|---|
| **git** | 6 (fund, branches, remotos, desfazer, **colaboracao**, **avancado**-grabbag) | 5 (fund**+config/perf**, branches, desfazer, **remotos-e-colaboracao** fundidas, avancado enxuto) | 49% → **81,7%** |
| **docker** | 6 (…, **producao-seguranca** transversal) | 5 (producao-seguranca **distribuída**: scan/buildkit/registry→imagens; secrets/observ/boas-práticas→runtime-dados) | 68,9% → **83,3%** |
| **kubernetes** | 7 (netpol DUPLICADA em `seguranca-rbac` e `rede-servicos`) | 7 (netpol consolidada em `rede-servicos`; `seguranca-rbac` só auth/authz) | 77,6% → **85,2%** |

Regra aprendida: **evitar famílias "grab-bag" e transversais** (avancado, producao-seguranca) —
elas roubam docs das vizinhas. Fundir famílias que compartilham vocabulário (colaboracao↔remotos)
e distribuir as transversais pelas famílias coesas. Isso dobrou o macro-F1 sem escrever 1 lição.

### Estado final (2026-07-23)
| Expert | Lições | Famílias | Auto-check | evaluate.py macro-F1 |
|---|---:|---:|---|---|
| `git` | 60 | 5 | **10/10** | **81,7%** |
| `bash` | 24 | 4 | 7/8 | **86,7%** |
| `shell` | 30 | 5 | 9/9 | **80,0%** |
| `docker` | 39 | 5 | 8/9 | **83,3%** |
| `kubernetes` | 42 | 7 | **9/10** | **85,2%** |

Todos em **80–87%** (média ~84%). Chegar a **90% exato** é limitado pelo split retido minúsculo
(2–3 docs/família → 1 erro = ~-16pts); estabilizar em 90% exigiria ~15–18/família (densidade de nuvem).

**Limite da técnica de reestruturação (2026-07-23):** ela só ajuda quando há um problema
ESTRUTURAL (grab-bag, transversal, duplicata) — git/docker/k8s tinham e melhoraram muito.
Onde o overlap é INERENTE, reestruturar não ajuda (ou piora):
- **shell (80%)**: `pipes-e-redirecionamento` é um atrator natural (pipes aparecem em quase todo
  exemplo, ex.: `du | sort | head`). Tentei fundir `inspecao`+`navegacao` → **piorou p/ 68,8%**
  (a família grande virou atrator e puxou `processos` via `ssh/scp`). **Revertido p/ 80%.**
- **bash (86,7%)**: 4 famílias já distintas, só 1 near-miss `fundamentos`→`robustez`. Nada a fundir.
Regra fina: fundir só ajuda se a família resultante continuar DISTINTA das outras; fundir num
bloco grande com vocabulário amplo cria um novo atrator.
Lembrete: a métrica de **produção** (qual ferramenta, via gate) já é ~93–100%; o `evaluate.py` mede
o roteamento fino DENTRO da ferramenta, que a produção nem usa (o `SparseKB` varre o expert todo).

> **Pendente (opcional):** promover no `train/_INDEX.md` a fase 11 (`devops_experts`) — hoje
> planejava agrupar docker/kubernetes/terraform num expert só; foi feito **1 expert por
> ferramenta** (git/bash/shell/docker/kubernetes), conforme pedido ("cada um tem um expert").

---

## ✅ FEITO (2026-07-23) — Fase 12: expert `web` (HTML, CSS, layout + 7 frameworks de UI)

Criado **1 expert** (`web_experts`) cobrindo front-end de marcação/estilo e os frameworks de UI
mais usados, no mesmo modelo de arquivos (`treinamento_web/<família>/*.md`). **Dados no repo
irmão** `C:\grisecorp\repositories\treinamento\treinamento_web\` (o build usa o fallback `_DATA`).

**75 lições em 12 famílias** — desenhadas para vocabulário DISTINTO desde o início (aplicando a
descoberta do §DESCOBERTA: cada framework = família própria com termos próprios):

| Família | Lições | Vocabulário-âncora (distingue das vizinhas) |
|---|---:|---|
| `html` | 7 | tags semânticas, formulários, ARIA, `<meta>`, HTML5 |
| `css-fundamentos` | 7 | seletores, especificidade, box-model, `rem`/`em`, variáveis |
| `layout-responsivo` | 7 | flexbox, grid, media queries, mobile-first, z-index |
| `bootstrap` | 6 | `col-md-*`, `navbar`, `btn`, utilitários, 12 colunas |
| `angular-material` | 6 | `mat-button`, `mat-card`, `MatDialog`, CDK |
| `tailwind-e-frameworks` | 6 | utility-first, `p-4`/`flex`/`bg-*`, dark mode |
| `sass-scss` | 6 | `@mixin`, `@use`, aninhamento, maps, `@each` |
| `animacoes-e-efeitos` | 6 | `@keyframes`, `transition`, `transform`, filtros |
| `material-ui` | 6 | prop `sx`, `<Button>`, `ThemeProvider` (React) |
| `ant-design` | 6 | `antd`, `type="primary"`, `<Table>`, `ConfigProvider`, grid 24-col |
| `vuetify` | 6 | `v-btn`, `v-card`, `v-data-table`, `v-model` (Vue) |
| `primeng` | 6 | `p-button`, `p-table`, `pTemplate`, `MessageService` (Angular) |

- **Métrica** — auto-check **14/15 = 93%** · `evaluate.py` **macro-F1 86,9% / acurácia 87,5%**
  (segundo melhor de todos os experts; média das ferramentas era ~84%).
- **Erros (3 de 24 no split retido)** — todos entre pares INERENTEMENTE próximos, não estruturais:
  `material-ui`↔`vuetify` (ambos Material Design), `ant-design`↔`primeng` (ambos tabela de dados
  corporativa), `html`→`ant-design` (1 doc). Nada a reestruturar — é o teto do overlap genuíno.
- **Build script** `examples/build_web_experts.py` (lê `treinamento_web`, salva em
  `examples/axon_lang_data/web_experts/`). Rodar igual aos outros:
  ```powershell
  $env:PYTHONPATH = "C:\grisecorp\repositories\pyaxon\python"
  python C:\grisecorp\repositories\pyaxon\examples\build_web_experts.py
  # avaliar: $env:AXON_LESSONS_DIR="C:\grisecorp\repositories\treinamento\treinamento_web"; python tools\evaluate.py
  ```

---

## Pipeline de treino de um expert de linguagem (COMO FOI FEITO — não reinventar)

> Registrado pra não perder tempo. Todos os experts de linguagem (Java, .NET, JS,
> Python, PHP, Rust, Go e agora **Ruby**) seguem EXATAMENTE o mesmo fluxo. O código
> já existe; é só rodar.

### 1. Dados (fonte)
Lições em Markdown, uma por conceito, na convenção:
`treinamento_<lang>/<família>/<subsetor>/*.md` → o `path` do doc = `[família, subsetor]`.

- **Ruby**: `treinamento_ruby/` com 2 famílias — `ruby/` (sintaxe, colecoes, oop, strings,
  excecoes, metaprogramacao, modulos-gems) e `rails/` (fundamentos, routing, controllers,
  views, activerecord, queries, recursos, frontend, background-tempo-real, auth, api,
  testes, deploy-performance). **242 lições prontas** (86 ruby + 156 rails).
- O diretório é lido tanto do repo local (`pyaxon/treinamento_ruby/`) quanto do repo de
  dados irmão (`../treinamento/treinamento_ruby/`) — o build tenta o local, senão o irmão.
  (No momento os dois apontam pro mesmo conteúdo; os `.md` já estão nos dois lugares.)
- Filtro: só entram arquivos com `len(text) >= 200` chars.

### 2. Pré-requisito: `pyaxon` importável (compilar o core C++)
O `build_*_experts.py` faz `import pyaxon as ax`, que precisa do `_axon.pyd` compilado.
Ordem (Windows/MinGW, ver `scripts/`):
```powershell
powershell -File scripts\build.ps1          # 1) compila build\libaxon.a (core C++)
powershell -File scripts\build_python.ps1   # 2) gera python\pyaxon\_axon.pyd (pybind11)
```
Requisitos: MSYS2/MinGW em `C:\msys64\mingw64\bin`, `pip install pybind11`.
Depois, rodar com o pacote no path: `cd examples; $env:PYTHONPATH="..\python"; python ...`
(ou instalar o wheel: `pip install -e .`).

### 3. Treinar o expert
```powershell
cd examples
$env:PYTHONPATH = "..\python"
python build_ruby_experts.py
```
O script (`examples/build_ruby_experts.py`, já pronto e idêntico ao `build_go_experts.py`):
1. lê os `.md` → `docs = [(path, texto)]`;
2. `router = ax.modular.ModularRouter(epochs=300, batch_size=256)` — 1 expert por FAMÍLIA;
3. `kb = ax.vindex.SparseKB(ngram=1)` — índice semântico esparso;
4. `router.add` + `kb.add_document` pra cada lição;
5. `router.fit(dirty_only=False)` + `kb.build(dim=200)` (LSA);
6. salva em `examples/axon_lang_data/ruby_experts/`: `router.*.json(.w)`, `router.gate.json`,
   `kb.sparse.json.gz`, `router.manifest.json`;
7. imprime a acurácia de roteamento por família (conjunto `QUESTIONS` no script).

Env vars de tuning: `AXON_RUBY_DIR` (dir de dados), `AXON_EPOCHS` (300), `AXON_BATCH` (256),
`AXON_LSA_DIM` (200).

### 4. Estado atual (2026-07-21) — ✅ TREINADO
- ✅ Lições prontas (242 = 156 rails + 86 ruby) e `build_ruby_experts.py` pronto.
- ✅ **Expert gerado**: `examples/axon_lang_data/ruby_experts/` criado com
  `router.gate.json`, `router.ruby.json(.w)`, `router.rails.json(.w)`,
  `kb.sparse.json.gz` (1648 passages) e `router.manifest.json`.
- ✅ **Auto-check do build: 9/9 = 100%** (rota + trecho recuperado — ver §6).
- ✅ **Avaliação formal (`tools/evaluate.py`, split retido): família 98,0% / macro-F1 97,8%**
  (ver §7 pra subsetores e leitura completa).
- Nota: o `_axon.pyd` (19/07) e `build/libaxon.a` JÁ estavam compilados — não precisou
  recompilar o core. Bastou `$env:PYTHONPATH="python"; python examples/build_ruby_experts.py`.
- Pendente: `ruby_experts` é a **fase 9** no `train/_INDEX.md` (ainda marcada ⏳ GERANDO) —
  mover pra tabela "Já treinados (experts prontos)" com 242 lições / 100%.

### 5. Como o expert entra no sistema
O `ax.system.AxonSystem` carrega todos os `*_experts/` de `axon_lang_data/` e roteia entre
eles pelo gate de domínio (vocabulário). Nenhum código novo é preciso — basta o diretório
`ruby_experts/` existir junto dos outros (go_experts, rust_experts, ...).

### 6. EXEMPLO CONCRETO — comandos que rodaram (copiar e colar)
> ⚠️ **Pegadinha de caminho**: `pyaxon\treinamento_ruby` NÃO resolve como pasta real neste
> ambiente (o build usa o fallback `_DATA`). Os dados de verdade estão no repo irmão:
> `C:\grisecorp\repositories\treinamento\treinamento_ruby`. Use esse caminho em avaliações.

```powershell
# core já compilado (python\pyaxon\_axon.pyd de 19/07 + build\libaxon.a) — NÃO recompilar.
$env:PYTHONPATH = "C:\grisecorp\repositories\pyaxon\python"

# (a) TREINAR + auto-check de resposta (o build já imprime rota + passagem recuperada)
cd C:\grisecorp\repositories\pyaxon\examples
python build_ruby_experts.py
#   -> "ruby: 242 lessons | families/experts: {'rails': 156, 'ruby': 86}"
#   -> salva axon_lang_data/ruby_experts/ (router.gate/ruby/rails.json(.w), kb.sparse.json.gz)
#   -> "=== Ruby/Rails questions: routing + retrieved answer ===" com 9 perguntas
#   -> "FAMILY accuracy: 9/9 = 100%"  <-- ISSO é o "respondeu certo" (rota + trecho da lição)
```

**O "respondeu certo" é o bloco final do build**: para cada pergunta ele imprime
`[OK] familia > subsetor | <primeiros 150 chars da lição recuperada>`. 9/9 corretas — a rota
foi certa E o trecho recuperado é o pertinente (ex.: "procs e lambdas" → `ruby > colecoes`,
"has_many" → `rails > activerecord`, "RSpec/FactoryBot" → `rails > testes`).

### 7. AVALIAÇÃO formal (`tools/evaluate.py`) — split estratificado + métricas
Diferente do auto-check (9 perguntas escolhidas a mão), a avaliação treina um router num
split de treino e mede num teste **retido**, com precisão/recall/F1 por área + matriz de
confusão + macro-F1. Roda em 3 níveis (o nível FAMÍLIA é o que a produção usa de fato):

```powershell
$env:PYTHONPATH = "C:\grisecorp\repositories\pyaxon\python"
$D = "C:\grisecorp\repositories\treinamento\treinamento_ruby"
$env:AXON_LESSONS_DIR = $D;         python tools\evaluate.py   # família: ruby vs rails
$env:AXON_LESSONS_DIR = "$D\ruby";  python tools\evaluate.py   # 7 subsetores de ruby/
$env:AXON_LESSONS_DIR = "$D\rails"; python tools\evaluate.py   # 13 subsetores de rails/
```

**Resultados (2026-07-21):**

| Nível | Split treino/teste | Acurácia | macro-F1 | Leitura |
|---|---|---|---|---|
| **Família (ruby vs rails)** | 192 / 50 | **98,0%** | **97,8%** | ✅ é o que a produção roteia. Só 1 lição `ruby` caiu em `rails`. |
| Subsetores `ruby/` | 65 / 21 | 76,2% | 76,5% | diagnóstico (teste minúsculo, ~3/área). |
| Subsetores `rails/` | 117 / 39 | 71,8% | 69,9% | diagnóstico (teste minúsculo, ~3/área). |

**Por que os subsetores dão "baixo" e por que NÃO é problema:** (1) o teste retido tem ~3
docs por área — 1 erro já derruba muito o F1; (2) o router de subsetor aqui é treinado do
zero só com o split, não é o expert salvo; (3) **a produção não roteia por subsetor**: ela
roteia por FAMÍLIA (98%) e o `SparseKB` recupera a melhor passagem varrendo a família toda,
então confundir subsetores vizinhos não erra a resposta. As confusões são todas entre
**vizinhos semânticos legítimos**:
- `ruby`: `metaprogramacao`↔`oop` (metaprogramação é sobre objetos/classes).
- `rails`: `api`↔`controllers`↔`routing`↔`recursos` (todos giram em torno de requisição HTTP)
  e `views`↔`frontend` (ambos são camada de apresentação).

**Veredito**: export saudável. Roteamento de domínio forte (98%), recuperação pertinente
(9/9 no auto-check). Se um dia quiser subir a separação de subsetores, o caminho é mais
lições por subsetor e/ou bigramas no gate — mas não é bloqueio pra usar o expert.
