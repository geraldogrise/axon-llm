# Rust — plano de treinamento (o que precisa constar)

> ✅ **TREINADO** (fase 7) — `rust_experts`, 130 lições, roteamento de família 100%.
> Dados em `treinamento/treinamento_rust` (branch fase-7).


**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Rust.
**Expert sugerido**: `rust_experts` (fase 7). **Total estimado**: ~150 lições.
**Convenção**: `treinamento_rust/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ (linguagem) — ~55 lições
### sintaxe (~12)
variáveis e mutabilidade (`let`/`mut`); tipos primitivos e inferência; shadowing; constantes e static; operadores; controle de fluxo (`if`/`else`); `loop`/`while`/`for`; `match`; expressões vs statements; funções e retorno; comentários e doc comments; formatação (`println!`/`format!`).
### ownership (~10)
o modelo de ownership; move e cópia; `Clone` vs `Copy`; borrowing (`&`/`&mut`); as regras do borrow checker; lifetimes básicos; lifetimes em funções e structs; slices; dangling references; ownership em coleções.
### tipos (~12)
structs; tuple structs e unit structs; enums; `Option<T>`; `Result<T, E>`; pattern matching avançado (`if let`/`while let`); generics em funções; generics em structs/enums; traits; trait bounds; trait objects (`dyn`); associated types.
### erros e coleções (~11)
tratamento de erros com `Result`; o operador `?`; `panic!` e quando usar; erros customizados; `Vec<T>`; `HashMap`; `String` vs `&str`; iterators e adapters (`map`/`filter`/`collect`); closures e capturas; `impl Trait`; conversões (`From`/`Into`).
### memória e smart pointers (~10)
`Box<T>`; `Rc<T>` e `RefCell<T>`; `Arc<T>`; interior mutability; `Deref` e `Drop`; ciclos e `Weak`; a stack vs heap no Rust; `unsafe` e ponteiros brutos; FFI básico; alinhamento e layout.

## concorrencia/ — ~20 lições
threads e `spawn`; `move` closures em threads; canais (`mpsc`); `Arc<Mutex<T>>`; `RwLock`; deadlocks e como evitar; `Send`/`Sync`; async/await introdução; runtime Tokio; futures e `.await`; tasks e `tokio::spawn`; canais async; `select!`; streams async; compartilhamento de estado async; padrões de concorrência.

## web/ — ~25 lições
### axum (~9)
setup e handlers; roteamento; extractors; JSON e serde; state compartilhado; middleware (tower); erros e respostas; banco de dados (sqlx); testes.
### actix-web (~8)
setup e app; handlers e rotas; extractors; JSON; state; middleware; erros; websockets.
### rocket (~4) + geral (~4)
Rocket: rotas, guards, forms; geral: CORS, autenticação JWT, arquitetura de API REST, deploy.

## ecossistema/ — ~30 lições
### cargo e projeto (~10)
Cargo e `Cargo.toml`; dependências e crates; workspaces; features; profiles (dev/release); publicar no crates.io; `cargo build/run/test`; documentação (`cargo doc`); scripts de build; versionamento semântico.
### testes (~7)
testes unitários (`#[test]`); testes de integração; `assert!`/`assert_eq!`; testar panics; doc tests; mocking; benchmarks.
### avançado (~13)
macros declarativas (`macro_rules!`); macros procedurais; derive macros; serde deep; trait `Iterator` customizado; `From`/`TryFrom`; pattern typestate; error handling com `anyhow`/`thiserror`; logging (`tracing`); CLI com `clap`; padrões idiomáticos; performance e zero-cost abstractions; interop com C.
