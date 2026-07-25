# Plano de treinamento — próximas fases (train/)

Este diretório é só **planejamento** (currículo): o que cada expert precisa cobrir pra
"responder tudo" sobre o domínio. As lições em si são geradas depois (mesma pipeline das
fases 1–6: `família/subsetor/*.md` → `build_*_experts.py` → ModularRouter + SparseKB).

## Já treinados (experts prontos)
| Fase | Domínio | Expert | Lições | Roteamento |
|---|---|---|---:|---|
| 1 | Escolar (mat/física/bio/química/português/história) | `rag_final` | 1.052 | 98,6% |
| 2 | Java (+ Spring, Jakarta, JPA...) | `java_experts` | 362 | 100% |
| 3 | .NET / C# (+ ASP.NET, EF...) | `dotnet_experts` | 284 | 100% |
| 4 | JS/TS (+ Node, React, Angular, Vue, Next) | `js_experts` | 392 | 91,5% |
| 5 | Python (+ Flask, FastAPI, Django, pandas, numpy, sklearn, torch, tf) | `python_experts` | 346 | 100% |
| 6 | PHP (+ Laravel, Symfony, CodeIgniter, WordPress, PDO, Doctrine) | `php_experts` | 237 | 100% |
| 7 | Rust (core, concorrência, web, ecossistema) | `rust_experts` | 130 | 100% |
| **8** | **Go (core, concorrência, web, data, ecossistema)** | **`go_experts`** | **140** | **100%** |

> **Gate de domínio entre os 8 experts: ~93–100%** (média ponderada retrieval + WordNB, `ax.system.AxonSystem`).
> Único ponto cego conhecido: a string **`net/http`** colide com **`.NET`** no nível de token
> (`net`) — perguntas que dizem "golang"/"do Go" roteiam 100%; "servidor web com net/http" sem
> contexto pode cair no `dotnet_experts`. Mitigável com bigramas no gate, se necessário.

## A planejar agora (este diretório)
Recomendação de agrupamento em experts (cada expert = 1 ModularRouter modular).
Legenda: ✅ treinado · ⏳ gerando · ⬜ planejado.

| Fase | Status | Expert sugerido | Domínios (famílias) | Lições | Plano |
|---|---|---|---|---:|---|
| 7 | ✅ | `rust_experts` | Rust (core, concorrência, web, ecossistema) | 130 | [rust.md](rust.md) |
| 8 | ✅ | `go_experts` | Go (core, concorrência, web, data, ecossistema) | 140 | [go.md](go.md) |
| 9 | ⏳ | `ruby_experts` | Ruby + Rails (ruby: 7 subsetores, rails: 13 subsetores) | ~225 | [ruby-on-rails.md](ruby-on-rails.md) |
| 10 | ⬜ | `cloud_experts` | AWS, Azure, GCP, OCI (1 família por nuvem) | ~440 | [aws](aws.md) · [azure](azure.md) · [gcp](gcp.md) · [oci](oci.md) |
| 11 | ⬜ | `devops_experts` | Terraform, Docker, Kubernetes (1 família cada) | ~230 | [terraform](terraform.md) · [docker](docker.md) · [kubernetes](kubernetes.md) |

> **Nuvens**: cada nuvem (AWS/Azure/GCP/OCI) é grande o bastante pra ser um expert próprio.
> Recomendo **1 família por nuvem** dentro de um `cloud_experts` (o gate de domínio já separa
> bem por vocabulário: EC2/S3 vs Blob/Cosmos vs GCS/BigQuery vs OCI/Autonomous). Se preferir,
> pode virar 4 experts separados — o `AxonSystem` roteia entre quantos experts houver.

Total planejado nas fases 7–11: **~1.120 lições**.

## Planos por linguagem (todas as que faltam — 1 arquivo cada)
Levantamento e prioridades em [linguagens-faltantes.md](linguagens-faltantes.md). Cada
linguagem tem seu próprio currículo:

- **Tier 1**: [c](c.md) · [cpp](cpp.md) · [kotlin](kotlin.md) · [swift](swift.md) · [sql](sql.md) · [bash](bash.md)
- **Tier 2**: [dart-flutter](dart-flutter.md) · [scala](scala.md) · [r](r.md) · [elixir](elixir.md) · [objective-c](objective-c.md) · [powershell](powershell.md) · [groovy](groovy.md)
- **Tier 3**: [julia](julia.md) · [matlab](matlab.md) · [haskell](haskell.md) · [clojure](clojure.md) · [fsharp](fsharp.md) · [ocaml](ocaml.md) · [erlang](erlang.md) · [lua](lua.md) · [perl](perl.md) · [solidity](solidity.md) · [zig](zig.md) · [nim](nim.md) · [crystal](crystal.md)
- **Tier 4**: [cobol](cobol.md) · [fortran](fortran.md) · [assembly](assembly.md) · [pascal-delphi](pascal-delphi.md) · [ada](ada.md) · [visual-basic](visual-basic.md) · [abap](abap.md) · [apex](apex.md) · [prolog](prolog.md) · [scheme-racket](scheme-racket.md) · [haxe](haxe.md)

> Bancos de dados, DevOps/observabilidade, mensageria, web-base e mobile (que não são
> "linguagens") ficam listados em [linguagens-faltantes.md](linguagens-faltantes.md) como
> tecnologias a planejar depois.

## Como o gate de domínio lida com o crescimento
`ax.system.AxonSystem` usa **média ponderada (retrieval + WordNB)** pra escolher o expert.
Já validado 100% em 6 domínios; escala pra 11+ porque cada linguagem/nuvem tem vocabulário
característico (`fn/impl/->` Rust; `func/goroutine/chan` Go; `def/end/do` Ruby; nomes de
serviço distintos por nuvem). Adicionar um expert **não** retreina os outros.
