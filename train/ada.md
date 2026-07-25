# Ada — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Ada (sistemas críticos).
**Expert sugerido**: família em `systems_experts` ou `legacy_experts`. **Total est.**: ~50 lições.
**Convenção**: `treinamento_ada/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~24
o que é Ada e onde é usada (aviônica/defesa); estrutura de um programa; tipos fortes; subtypes; declaração de variáveis; operadores; controle de fluxo (if/case); loops; procedures; functions; parâmetros (in/out/in out); packages (spec e body); arrays; records; enumerations; strings; ranges e constraints; overloading; atributos.

## tipagem-avancado/ — ~16
o sistema de tipos forte; derived types; constrained types; discriminants; variant records; access types (ponteiros); generics; exceptions; contracts (pre/postconditions - Ada 2012); tasking (concorrência); protected objects; rendezvous; type invariants; safety features.

## seguranca-ecossistema/ — ~10
SPARK (verificação formal); Ada para sistemas críticos; real-time; GNAT (compilador); Alire (pacotes); interop com C; certificação (DO-178); boas práticas; comparação com C/Rust; segurança e confiabilidade.
