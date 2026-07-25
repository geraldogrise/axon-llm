# F# — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre F# (funcional no .NET).
**Expert sugerido**: família em `dotnet_experts` ou `functional_experts`. **Total est.**: ~75 lições.
**Convenção**: `treinamento_fsharp/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~30
o que é F# (.NET funcional); `let` e imutabilidade; tipos e inferência; funções; currying e aplicação parcial; pipe (`|>`) e composição (`>>`); controle de fluxo; recursão; listas; arrays; sequences; tuplas; records; discriminated unions; pattern matching; active patterns; option type; guards; `match` avançado; unidades de medida.

## funcional-tipos/ — ~25
higher-order functions; closures; funções puras; Result type; error handling funcional; List/Seq/Array modules; map/filter/fold; computation expressions; async workflows; Option module; type providers; generics; interfaces; classes (quando usar); mutabilidade controlada (`mutable`/`ref`); coleções imutáveis (Map/Set).

## dotnet-ecossistema/ — ~20
interop com C#/.NET; usar bibliotecas .NET; ASP.NET com F# (Giraffe/Falco); JSON; testes (Expecto/xUnit); FAKE (build); Paket; scripts (.fsx) e FSI; MailboxProcessor (actors); paralelismo; SAFE stack (web); Fable (F# para JS); domain modeling; railway-oriented programming; boas práticas; comparação com C#.
