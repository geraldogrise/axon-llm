# Nim — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Nim.
**Expert sugerido**: família em `systems_experts`. **Total est.**: ~50 lições.
**Convenção**: `treinamento_nim/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~25
o que é Nim (compilado, Python-like); indentação e sintaxe; variáveis (`var`/`let`/`const`); tipos; inferência; operadores; strings; controle de fluxo; loops; funções (`proc`); parâmetros e retorno; result implícito; overloading; iterators; closures; tuplas; seq e arrays; comentários; case (variantes).

## tipos-avancado/ — ~15
objects (OOP); herança; methods e dispatch; enums; distinct types; generics; concepts; templates; macros (metaprogramação); AST; ref e ponteiros; gerenciamento de memória (ARC/ORC); exceptions; pragmas; conversões.

## ecossistema/ — ~10
Nimble (pacotes); compilar para C/C++/JS; interop com C; testes (unittest); performance; async/await; std lib; FFI; boas práticas; comparação com Python/Rust.
