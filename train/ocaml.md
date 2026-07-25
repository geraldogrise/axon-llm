# OCaml — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre OCaml.
**Expert sugerido**: família em `functional_experts`. **Total est.**: ~65 lições.
**Convenção**: `treinamento_ocaml/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~28
o que é OCaml; o toplevel (utop); `let` e binding; tipos e inferência; funções; currying; controle de fluxo; recursão; listas; tuplas; records; variantes (tipos algébricos); pattern matching; option type; guards; funções de ordem superior; map/filter/fold; imutabilidade; refs e mutabilidade; arrays; strings.

## tipos-modulos/ — ~22
type inference avançado; polimorfismo paramétrico; variantes polimórficas; GADTs; módulos; signatures; functors (módulos parametrizados); abstract types; o sistema de módulos; exceptions; error handling (Result); labeled/optional arguments; type classes (via módulos); recursive types; mutually recursive.

## avancado-ecossistema/ — ~15
Dune (build); OPAM (pacotes); testes; concorrência (Lwt/Async); efeitos (OCaml 5); domains (paralelismo); interop com C; PPX (metaprogramação); Core/Base (stdlib alternativa); ReasonML/Melange (visão geral); performance; imperativo vs funcional; boas práticas.
