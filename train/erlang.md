# Erlang — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Erlang.
**Expert sugerido**: família em `elixir_experts` (mesma BEAM) ou expert próprio. **Total est.**: ~70 lições.
**Convenção**: `treinamento_erlang/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~28
o que é Erlang e a BEAM; o shell; tipos de dados; átomos; variáveis (imutáveis); pattern matching; o operador `=`; listas; tuplas; maps; strings e binaries; funções; funções anônimas (`fun`); guards; controle de fluxo (`case`/`if`); recursão; list comprehensions; módulos; higher-order functions; records.

## concorrencia-otp/ — ~28
o modelo de atores; processos (`spawn`); mensagens (`!`/`receive`); links e monitors; timeouts; registered processes; concorrência massiva; fault tolerance ("let it crash"); supervisores; gen_server; gen_statem; gen_event; behaviours; OTP applications; supervision trees; ETS/DETS; Mnesia (banco distribuído); distribuição (nós); hot code swapping.

## avancado-ecossistema/ — ~14
rebar3 (build); tratamento de erros; exceptions (`throw`/`catch`); testes (EUnit/Common Test); dializer (types); NIFs (interop C); ports; logging; releases; observabilidade; performance; padrões OTP; comparação com Elixir; telecom e sistemas de tempo real; boas práticas.
