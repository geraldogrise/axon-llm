# Crystal — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Crystal.
**Expert sugerido**: família em `systems_experts`. **Total est.**: ~50 lições.
**Convenção**: `treinamento_crystal/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~25
o que é Crystal (Ruby-like, compilado); sintaxe (parecida com Ruby); variáveis e tipos; type inference; union types; nil handling; operadores; strings e símbolos; controle de fluxo; loops; blocks e procs; métodos; overloading por tipo; tuplas e named tuples; arrays e hashes; ranges; comentários.

## tipos-oop/ — ~15
classes e structs; herança; módulos e mixins; abstract classes; generics; type restrictions; macros (metaprogramação); enums; exceptions; visibilidade; getters/setters (`property`); method_missing; annotations; conversões; type checking em compilação.

## concorrencia-ecossistema/ — ~10
fibers (concorrência leve); channels; async I/O; shards (pacotes); interop com C; testes (spec); web (Kemal/Lucky); performance; compilação; comparação com Ruby; boas práticas.
