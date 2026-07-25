# C — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre a linguagem C.
**Expert sugerido**: família `c` (em `systems_experts` com C++/Rust, ou expert próprio). **Total est.**: ~110 lições.
**Convenção**: `treinamento_c/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~45
### sintaxe (~12)
estrutura de um programa e `main`; tipos primitivos e `sizeof`; variáveis e constantes; operadores; `printf`/`scanf` e formatação; condicionais (`if`/`switch`); loops (`for`/`while`/`do-while`); funções e protótipos; escopo e classes de armazenamento; `typedef`; enums; preprocessador (`#define`/`#include`).
### ponteiros e memória (~16)
ponteiros (o conceito); aritmética de ponteiros; ponteiros e arrays; ponteiro para ponteiro; alocação dinâmica (`malloc`/`free`); `calloc`/`realloc`; memory leaks e dangling pointers; passagem por referência; ponteiros para função; arrays multidimensionais; strings como ponteiros; stack vs heap; `const` e ponteiros; void pointers; segmentation faults; boas práticas de memória.
### tipos compostos (~9)
structs; struct e ponteiros; unions; bitfields; arrays de structs; structs aninhadas; alinhamento e padding; enums avançados; typedef com structs.
### strings e I/O (~8)
strings e `char[]`; `string.h` (strcpy/strcmp/strlen/strcat); manipulação manual; entrada segura (`fgets`); arquivos (`fopen`/`fclose`); ler/escrever arquivos; `fprintf`/`fscanf`; buffers.

## stdlib-ferramentas/ — ~30
`stdlib.h`; `math.h`; `string.h`; `ctype.h`; `time.h`; `stdio.h` avançado; `errno` e tratamento de erros; `assert.h`; argumentos de linha de comando (`argc`/`argv`); variáveis de ambiente; alocadores customizados; macros avançadas; compilação separada e headers; `extern`/`static`; make e Makefiles; gcc/clang flags; debugging com gdb; valgrind; linking; bibliotecas estáticas e dinâmicas.

## avancado/ — ~35
gerenciamento manual de memória (padrões); estruturas de dados (lista ligada); pilha e fila; árvores; hash tables; recursão; ponteiros de função e callbacks; manipulação de bits; endianness; alocação alinhada; programação de sistemas (syscalls); processos (`fork`/`exec`); threads (pthreads); mutexes e sincronização; sockets (rede); sinais; memória compartilhada; `setjmp`/`longjmp`; undefined behavior; C89 vs C99 vs C11 vs C17; segurança (buffer overflow); embarcado (visão geral); interoperabilidade; boas práticas.
