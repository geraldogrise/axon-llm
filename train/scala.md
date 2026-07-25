# Scala — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Scala.
**Expert sugerido**: `scala_experts` ou família em `jvm_experts`. **Total est.**: ~110 lições.
**Convenção**: `treinamento_scala/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~40
### sintaxe (~10)
`val` vs `var`; tipos e inferência; expressões; controle de fluxo; `if`/`for`/`while`; for comprehensions; funções e métodos; funções anônimas; string interpolation; Unit e tipos.
### oop (~14)
classes e construtores; case classes; objects e companion objects; traits; herança; mixins; abstract classes; sealed traits; enums (Scala 3); pattern matching; visibilidade; self types; type members; given/using (Scala 3).
### funcional (~16)
imutabilidade; higher-order functions; currying; partial application; closures; funções puras; recursão e tail recursion; `Option`/`Some`/`None`; `Either`; `Try`; monads (conceito); functors; for comprehensions com monads; composição de funções; lazy evaluation; pattern matching avançado.

## coleções-tipos/ — ~25
List/Vector/Set/Map; imutáveis vs mutáveis; operações (map/filter/fold/reduce); flatMap; collections lazy (LazyList); tuplas; generics; variância (covariance/contravariance); type bounds; implicits (Scala 2)/givens (Scala 3); type classes; higher-kinded types; path-dependent types; structural types.

## concorrencia-ecossistema/ — ~30
Futures; ExecutionContext; async; Akka (actors); Akka Streams; Cats (visão geral); Cats Effect (IO); ZIO (visão geral); sbt (build); testes (ScalaTest/MUnit); Spark com Scala; Play Framework; JSON (circe/play-json); interop com Java; macros e metaprogramação; Scala 2 vs Scala 3; boas práticas funcionais.
