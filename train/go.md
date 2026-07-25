# Go (Golang) — plano de treinamento (o que precisa constar)

> ✅ **TREINADO** (fase 8) — `go_experts`, 140 lições, roteamento de família 100%.
> Dados em `treinamento/treinamento_go` (branch fase-8).


**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Go.
**Expert sugerido**: `go_experts` (fase 8). **Total estimado**: ~140 lições.
**Convenção**: `treinamento_go/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ (linguagem) — ~50 lições
### sintaxe (~12)
pacotes e `main`; variáveis (`var`/`:=`); tipos básicos e zero values; constantes e `iota`; operadores; `if`/`else` e `switch`; `for` (única forma de loop); funções e múltiplos retornos; retornos nomeados; `defer`; comentários e gofmt; ponteiros.
### tipos e structs (~13)
structs; métodos e receivers; embedding (composição); interfaces; interface vazia (`any`); type assertions e type switch; slices; arrays vs slices; maps; strings e runes; `make` vs `new`; conversões de tipo; generics (type parameters).
### funções e erros (~12)
funções variádicas; funções como valores; closures; recursão; tratamento de erros idiomático; `errors.New`/`fmt.Errorf`; wrapping (`%w`) e `errors.Is`/`errors.As`; erros customizados; `panic` e `recover`; múltiplos valores de erro; sentinel errors; boas práticas de erro.
### stdlib essencial (~13)
`fmt`; `strings` e `strconv`; `bytes`; `time`; `os` e `io`; `bufio`; `encoding/json`; `regexp`; `sort`; `math`; `context` básico; `flag`; `log`.

## concorrencia/ — ~22 lições
goroutines; canais (channels); canais buffered vs unbuffered; `select`; direção de canais; fechar canais e `range`; `sync.WaitGroup`; `sync.Mutex`/`RWMutex`; `sync.Once`; `context` para cancelamento e timeout; worker pools; pipelines; fan-in/fan-out; race conditions e o race detector; deadlocks; atomic; padrões de concorrência idiomáticos.

## web/ — ~28 lições
### net/http (~9)
servidor HTTP; handlers e `HandleFunc`; `ServeMux` e roteamento; request e response; query e path params; JSON encode/decode; middleware; arquivos estáticos; cliente HTTP.
### gin (~8)
setup e rotas; grupos de rotas; binding e validação; middleware; JSON e respostas; params e query; upload de arquivos; tratamento de erros.
### echo/fiber e geral (~11)
Echo: rotas, middleware, binding; Fiber: básico; CORS; autenticação JWT; templates HTML; websockets; arquitetura REST; graceful shutdown; rate limiting; deploy.

## data/ — ~18 lições
`database/sql`; drivers (pgx/mysql); prepared statements; transações; connection pooling; `sqlx`; GORM: models e migrations; GORM: CRUD e queries; GORM: relacionamentos; Redis; migrations (golang-migrate); NoSQL (mongo-driver).

## ecossistema/ — ~22 lições
go modules e `go.mod`; `go get`/`go install`; estrutura de projeto e layout padrão; `go build`/`go run`; testes (`testing`); table-driven tests; benchmarks; `go test -cover`; mocking (interfaces); `go vet`/staticcheck; documentação (godoc); build tags; cross-compilation; workspaces (`go.work`); ferramentas (`gopls`); profiling (pprof); logging estruturado (`slog`); CLI com cobra; embedding de arquivos (`embed`); boas práticas idiomáticas (Effective Go).
