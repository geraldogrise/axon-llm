# Bash / Shell — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre shell scripting (Bash) e Linux CLI.
**Expert sugerido**: `shell_experts` ou família em `devops_experts`. **Total est.**: ~90 lições.
**Convenção**: `treinamento_bash/<família>/<subsetor>/*.md` → path = [família, subsetor].

## fundamentos/ — ~14
o que é o shell (bash/sh/zsh); o terminal e o prompt; comandos básicos (ls/cd/pwd); navegação no filesystem; caminhos absolutos e relativos; criar/mover/copiar/remover (mkdir/cp/mv/rm); permissões (chmod/chown); usuários e grupos; ver arquivos (cat/less/head/tail); ajuda (man/--help); histórico e atalhos; variáveis de ambiente; PATH.

## scripting/ — ~22
o shebang (`#!/bin/bash`); executar scripts; variáveis; aspas (simples/duplas); parâmetros posicionais (`$1`/`$@`); condicionais (`if`/`test`/`[[ ]]`); comparações numéricas e de string; `case`; loops (`for`/`while`/`until`); funções; return e exit codes; `read` (entrada); arrays; aritmética (`$(( ))`); expansão de variáveis; substituição de comando (`$( )`); expansão de chaves; here-documents; parâmetros com valores padrão; `getopts` (argumentos); debugging (`set -x`).

## texto-pipes/ — ~20
pipes (`|`); redirecionamento (`>`/`>>`/`<`); stderr (`2>`); `grep` (básico e regex); `sed` (substituição e edição); `awk` (processamento de colunas); `cut`; `sort`; `uniq`; `tr`; `wc`; `find` (busca de arquivos); `xargs`; `tee`; expressões regulares; globbing; `diff`; JSON com `jq`; processamento de logs; one-liners.

## sistema-avancado/ — ~22
processos (`ps`/`top`/`kill`); jobs e background (`&`/`jobs`/`fg`/`bg`); `nohup` e `disown`; cron e agendamento; systemd (visão geral); variáveis de ambiente e `.bashrc`/`.profile`; aliases; sinais e traps (`trap`); subshells; `source`; código de saída e `$?`; `&&`/`||`; SSH e execução remota; tar e compressão; wget/curl; gerenciamento de pacotes (apt/yum); montagem de discos; monitoramento; segurança de scripts; boas práticas (shellcheck).
