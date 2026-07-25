# Scheme / Racket — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder sobre Scheme e Racket (Lisp/ensino).
**Expert sugerido**: família em `functional_experts`. **Total est.**: ~45 lições.
**Convenção**: `treinamento_scheme/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~24
o que são Lisp/Scheme/Racket; S-expressions; prefix notation; `define`; tipos de dados; números; símbolos; listas (`cons`/`car`/`cdr`); pares; quote e quasiquote; funções (`lambda`); recursão; recursão de cauda; condicionais (`if`/`cond`/`case`); `let`/`let*`/`letrec`; higher-order functions (map/filter/fold); closures; variádicas; named let.

## avancado-ecossistema/ — ~21
continuations (call/cc); tail calls; macros (define-syntax); hygiene; syntax-rules; estruturas (structs); vetores; hash tables; mutação (set!); streams (lazy); pattern matching (Racket); contracts (Racket); módulos; TCP/web (Racket); DrRacket; SICP (conceitos clássicos); typed Racket; performance; comparação com Common Lisp/Clojure; boas práticas.
