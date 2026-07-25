# Lua — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Lua (scripting/games/embutido).
**Expert sugerido**: `lua_experts` ou família em `scripting_experts`. **Total est.**: ~60 lições.
**Convenção**: `treinamento_lua/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~28
o que é Lua e onde é usada; variáveis e tipos; nil e booleanos; números e strings; operadores; concatenação (`..`); controle de fluxo (`if`/`elseif`); loops (`for`/`while`/`repeat`); funções; múltiplos retornos; argumentos variádicos (`...`); escopo (`local`); closures; recursão; strings (biblioteca string); patterns (Lua patterns); comentários.

## tables-oop/ — ~18
tables (a estrutura central); arrays com tables; dicionários; metatables; metamethods (`__index`/`__add`); OOP com tables; herança; a biblioteca table; iteradores (`pairs`/`ipairs`); custom iterators; coroutines; pcall (tratamento de erros); módulos e `require`; namespaces; deep vs shallow copy.

## aplicacoes-ecossistema/ — ~14
embedding Lua em C; a C API; LuaJIT; Lua no Roblox; Lua no Neovim (config); Love2D (games); OpenResty/Nginx; Redis scripts; garbage collection; performance; sandbox e segurança; LuaRocks (pacotes); debugging; boas práticas.
