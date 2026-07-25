# Elixir (+ Phoenix) — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder sobre Elixir e o framework Phoenix.
**Expert sugerido**: `elixir_experts`. **Total est.**: ~110 lições.
**Convenção**: `treinamento_elixir/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~35
o que é Elixir e a BEAM/Erlang VM; tipos básicos; átomos; imutabilidade; pattern matching; o operador `=` (match); listas e tuplas; keyword lists; maps; structs; funções (anônimas e nomeadas); módulos; `pipe` (`|>`); guards; controle de fluxo (`case`/`cond`/`if`); recursão; comprehensions (`for`); Enum; Stream (lazy); protocols; behaviours; sigils; strings e binaries; charlists; ranges.

## concorrencia-otp/ — ~28
o modelo de atores; processos (`spawn`); mensagens (`send`/`receive`); links e monitors; Agent; Task; GenServer; Supervisor e árvores de supervisão; Registry; DynamicSupervisor; estado em processos; fault tolerance ("let it crash"); OTP applications; ETS (armazenamento); GenStage; Flow; distribuição (nós); hot code reloading.

## phoenix/ — ~32
introdução ao Phoenix; estrutura do projeto; roteamento; controllers; views e templates (HEEx); layouts; Ecto (schemas); Ecto (migrations); Ecto (queries e changesets); Ecto (associações); contexts; Plug; LiveView (fundamentos); LiveView (eventos e estado); Phoenix Channels (websockets); PubSub; autenticação; forms e validação; APIs JSON; uploads; testes; deploy (releases).

## ecossistema/ — ~15
Mix (build tool); dependências (Hex); testes (ExUnit); doctests; formatação (`mix format`); Dialyzer (types); documentation (ExDoc); Nerves (IoT, visão geral); Nx (numérico, visão geral); Broadway; Oban (jobs); telemetria; boas práticas; comparação com Erlang.
