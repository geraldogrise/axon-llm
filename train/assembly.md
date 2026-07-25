# Assembly — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Assembly (x86-64 e ARM).
**Expert sugerido**: família em `systems_experts`. **Total est.**: ~65 lições.
**Convenção**: `treinamento_assembly/<família>/<subsetor>/*.md` → path = [família, subsetor].

## fundamentos/ — ~18
o que é Assembly e linguagem de máquina; arquitetura de CPU; registradores; a memória e endereçamento; sintaxe (Intel vs AT&T); instruções básicas (MOV); aritmética (ADD/SUB/MUL/DIV); a stack (PUSH/POP); flags; comparação (CMP); saltos (JMP/JE/JNE); loops; labels; diretivas do assembler; montagem e linkagem (nasm/gas); sistema numérico (hex/binário).

## x86-64/ — ~24
registradores x86-64 (RAX/RBX/...); modos de endereçamento; instruções de movimentação; operações lógicas (AND/OR/XOR); shifts e rotates; a stack frame; chamadas de função (calling conventions); System V ABI; passagem de parâmetros; retorno de valores; syscalls (Linux); acesso à memória; arrays em assembly; strings; ponteiros; SSE/AVX (SIMD); interrupções; inline assembly (em C).

## arm-avancado/ — ~23
arquitetura ARM; registradores ARM; instruções ARM/Thumb; ARM vs x86; AArch64; endereçamento ARM; calling convention ARM; embarcado; otimização de código; entendendo output de compilador; reverse engineering (básico); debugging (gdb); disassembly; bootloaders (visão geral); performance; segurança (shellcode/ROP conceitos); comparação de arquiteturas; boas práticas.
