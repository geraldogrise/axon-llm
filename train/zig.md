# Zig — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Zig.
**Expert sugerido**: família em `systems_experts`. **Total est.**: ~55 lições.
**Convenção**: `treinamento_zig/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~25
o que é Zig (systems/sem C); variáveis (`const`/`var`); tipos; inteiros e floats; operadores; controle de fluxo (`if`/`while`/`for`); switch; funções; opcionais (`?`); error unions (`!`); error handling (`try`/`catch`); defer e errdefer; structs; enums; unions; arrays e slices; strings; ponteiros; comptime (avaliação em compilação); comentários.

## memoria-avancado/ — ~18
gerenciamento manual de memória; allocators; `std.mem.Allocator`; alocação e liberação; sem hidden control flow; comptime avançado; generics via comptime; tipos genéricos; testing (`test` blocks); build.zig; interop com C (`@cImport`); undefined behavior detection; packed structs; alinhamento; SIMD; async (visão geral).

## ecossistema/ — ~12
o build system; cross-compilation (destaque do Zig); usar Zig como compilador C; standard library; tratamento de erros idiomático; padrões de alocação; performance; sem macros (comptime no lugar); pacotes; debugging; comparação com Rust/C; boas práticas.
