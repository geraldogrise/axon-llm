# Perl — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Perl.
**Expert sugerido**: família em `scripting_experts`. **Total est.**: ~60 lições.
**Convenção**: `treinamento_perl/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~28
sintaxe e `use strict`/`warnings`; escalares (`$`); arrays (`@`); hashes (`%`); contexto (escalar vs lista); operadores; strings e interpolação; controle de fluxo; loops; funções (`sub`); argumentos (`@_`); referências; estruturas de dados complexas; escopo (`my`/`local`); `qw`; heredocs; sigils; truthiness; special variables (`$_`/`@ARGV`).

## regex-texto/ — ~18
expressões regulares (a força do Perl); match (`=~`); substituição (`s///`); grupos e captura; modificadores; `tr///`; split e join; processamento de texto; leitura de arquivos; escrita; slurp mode; one-liners (`perl -e`); manipulação de linhas; parsing; named captures; lookahead/lookbehind.

## modulos-avancado/ — ~14
módulos e `use`; CPAN; criar módulos; OOP (bless); Moose/Moo (OOP moderno); packages; exceções (eval/die); testes (Test::More); DBI (banco de dados); processar CSV/JSON; web (Mojolicious/Dancer, visão geral); references avançadas; closures; boas práticas (Perl moderno); comparação Perl 5 vs Raku.
