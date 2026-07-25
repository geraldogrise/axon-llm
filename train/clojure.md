# Clojure — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Clojure (Lisp na JVM).
**Expert sugerido**: família em `functional_experts` ou `jvm_experts`. **Total est.**: ~75 lições.
**Convenção**: `treinamento_clojure/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~30
o que é Clojure (Lisp/JVM); a REPL; sintaxe de S-expressions; prefix notation; tipos de dados; imutabilidade; estruturas de dados persistentes (list/vector/map/set); keywords; símbolos; `def` e `let`; funções (`defn`); funções anônimas (`fn`/`#()`); destructuring; controle de fluxo (`if`/`when`/`cond`); loops (`loop`/`recur`); higher-order functions; `map`/`filter`/`reduce`; threading macros (`->`/`->>`); lazy sequences.

## dados-estado/ — ~25
sequences abstraction; transducers; namespaces; interop com Java; atoms; refs e STM; agents; vars dinâmicas; multimethods; protocols; records; spec (validação de dados); manipulação de mapas; nested update (`update-in`/`assoc-in`); coleções aninhadas; conversões; metadata; polimorfismo.

## macros-ecossistema/ — ~20
macros (o poder do Lisp); `defmacro`; quote/unquote; `macroexpand`; homoiconicidade; Leiningen e deps.edn; testes (clojure.test); concorrência (core.async); channels; ClojureScript (visão geral); Ring/Compojure (web); web APIs; REPL-driven development; tooling; boas práticas; comparação com outras Lisps.
