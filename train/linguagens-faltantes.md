# Linguagens e tecnologias que ainda faltam (gap analysis)

Levantamento do que o modelo **ainda não aprendeu**, considerando o que já existe e o que
está planejado neste diretório. Serve pra decidir as próximas fases.

## Situação atual
- **Já aprendido (fases 1–6)**: Escolar · **Java** · **C#/.NET** · **JavaScript/TypeScript** · **Python** · **PHP**
- **Planejado agora (fases 7–11)**: **Rust** · **Go** · **Ruby/Rails** · **AWS · Azure · GCP · OCI** · **Terraform · Docker · Kubernetes**

## Linguagens de programação que faltam

### Tier 1 — alta prioridade (muito populares, sem cobertura)
| Linguagem | Por quê | Ecossistema a cobrir |
|---|---|---|
| **C** | base de tudo, sistemas/embarcado | ponteiros, memória, structs, stdlib, make |
| **C++** | jogos, sistemas, alta performance | OOP, templates, STL, RAII, smart pointers, C++ moderno |
| **Kotlin** | Android oficial, backend (Spring) | core, coroutines, Android, Ktor |
| **Swift** | iOS/macOS | core, opcionais, SwiftUI, concurrency |
| **SQL** (standalone) | universal em dados | DDL/DML, joins, índices, window functions, otimização |
| **Bash/Shell** | automação, DevOps | scripting, pipes, sed/awk, cron |

### Tier 2 — populares em nichos fortes
| Linguagem | Nicho |
|---|---|
| **Dart (+ Flutter)** | apps mobile multiplataforma |
| **Scala** | big data (Spark), JVM funcional |
| **R** | estatística, ciência de dados |
| **Elixir (+ Phoenix)** | sistemas concorrentes/tempo real |
| **Objective-C** | legado Apple |
| **PowerShell** | automação Windows |
| **Groovy** | Gradle, Jenkins |

### Tier 3 — científico / funcional / especializado
Julia (computação científica) · MATLAB (engenharia) · Haskell (funcional puro) · Clojure ·
F# · OCaml · Erlang · Lua (scripting/games/Roblox) · Perl · Solidity (blockchain/smart
contracts) · Zig · Nim · Crystal.

### Tier 4 — legado / muito de nicho
COBOL · Fortran · Assembly (x86/ARM) · Pascal/Delphi · Ada · Visual Basic · ABAP (SAP) ·
Apex (Salesforce) · Prolog · Scheme/Racket · Haxe.

## Frameworks / runtimes que faltam (linguagens já cobertas)
- **JVM**: Spring além do básico, Micronaut, Quarkus, Android (Kotlin/Java)
- **JS/TS**: Svelte, SolidJS, Astro, NestJS, Deno, Bun, Electron, React Native
- **Python**: Polars, FastAPI avançado, LangChain, Hugging Face, Airflow, Streamlit
- **.NET**: Blazor, MAUI, gRPC, SignalR (aprofundar)
- **Ruby**: Sinatra, Hanami

## Bancos de dados (merecem expert próprio ou família `data`)
**Relacionais**: PostgreSQL · MySQL/MariaDB · SQL Server · Oracle DB · SQLite.
**NoSQL**: MongoDB · Redis · Cassandra · DynamoDB · Elasticsearch · Neo4j (grafos) · CouchDB.
**Data/analytics**: BigQuery · Snowflake · ClickHouse · DuckDB.

## DevOps / infra / observabilidade
Git (fundamentos + avançado) · GitHub Actions · GitLab CI · Jenkins · **Ansible** · Puppet ·
Chef · Helm (aprofundar) · **Prometheus + Grafana** · ELK/OpenSearch · Nginx · Apache ·
Linux (administração) · Vault · Consul · Packer · ArgoCD/Flux (GitOps).

## Mensageria / streaming / integração
**Kafka** · RabbitMQ · NATS · gRPC · **GraphQL** · MQTT · Redis Streams · Apache Pulsar.

## Dados / ML / IA (aprofundar além do que já há em Python)
Apache Spark · Hadoop · **Airflow** · dbt · Pandas avançado/Polars · **Hugging Face
Transformers** · LangChain · MLflow · Ray · ONNX · CUDA programming.

## Web / frontend base
**HTML5** · **CSS3** · Sass/SCSS · Tailwind CSS · Bootstrap · Web APIs (fetch, WebSockets,
Web Components) · acessibilidade · SEO técnico.

## Mobile
Flutter (Dart) · React Native · Android nativo (Kotlin) · iOS nativo (Swift) · Jetpack Compose ·
SwiftUI.

## Ordem sugerida das próximas fases (depois de 7–11)
1. **Fundamentos que faltam e são universais**: SQL, Git, Bash/Linux, HTML/CSS.
2. **Linguagens Tier 1**: C, C++, Kotlin, Swift.
3. **Bancos de dados** (PostgreSQL, MongoDB, Redis) como `db_experts`.
4. **DevOps/observabilidade** (Ansible, Prometheus/Grafana, CI/CD, Git avançado).
5. **Mobile** (Flutter, React Native) e **Tier 2** (Dart, Scala, R, Elixir).
6. **Mensageria/dados** (Kafka, GraphQL, Spark, Airflow).
7. Tiers 3–4 conforme necessidade.
