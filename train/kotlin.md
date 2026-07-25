# Kotlin — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Kotlin.
**Expert sugerido**: `kotlin_experts` ou família em `jvm_experts`. **Total est.**: ~130 lições.
**Convenção**: `treinamento_kotlin/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~45
### sintaxe (~12)
`val` vs `var`; tipos e inferência; null safety (`?`/`!!`/`?.`); Elvis operator; string templates; controle de fluxo (`if`/`when`); loops e ranges; funções; argumentos nomeados e padrão; `Unit`/`Nothing`; smart casts; comentários e convenções.
### oop (~16)
classes e construtores; propriedades e backing fields; `data class`; herança e `open`; interfaces; classes abstratas; objetos e `companion object`; `sealed class`; enums; nested e inner classes; visibilidade; delegation (`by`); `object` singleton; extension properties; operator overloading; `init` blocks.
### funcional (~10)
funções de ordem superior; lambdas; `it` e receivers; funções de escopo (`let`/`run`/`with`/`apply`/`also`); coleções (`map`/`filter`/`reduce`); sequences; funções inline; closures; higher-order idiomático; imutabilidade.
### recursos (~7)
extension functions; null safety avançado; generics e variância (`in`/`out`); reified types; type aliases; destructuring; operadores customizados.

## coroutines/ — ~18
o que são coroutines; `suspend` functions; `launch` e `async`; coroutine scopes; dispatchers; `withContext`; structured concurrency; cancelamento; `Flow`; operadores de Flow; StateFlow e SharedFlow; channels; exception handling em coroutines; testes de coroutines.

## android/ — ~35
introdução ao Android com Kotlin; Activities e lifecycle; Fragments; Views e layouts; Jetpack Compose (fundamentos); Compose state; Compose layouts; ViewModel; LiveData vs StateFlow; navegação; Room (banco de dados); Retrofit (rede); dependency injection (Hilt); RecyclerView; permissões; WorkManager; DataStore; Material Design; testes no Android; arquitetura (MVVM); Clean Architecture.

## backend-ecossistema/ — ~32
Ktor (servidor e rotas); Ktor (clientes e serialização); Spring Boot com Kotlin; kotlinx.serialization; Gradle Kotlin DSL; testes (JUnit/Kotest); MockK; multiplatform (KMP); interoperabilidade com Java; DSLs em Kotlin; annotations; reflection; contratos e `contract`; build e deploy; boas práticas idiomáticas.
