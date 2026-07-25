# Fortran — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Fortran (computação científica/HPC).
**Expert sugerido**: família em `legacy_experts` ou `scientific_experts`. **Total est.**: ~50 lições.
**Convenção**: `treinamento_fortran/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~24
o que é Fortran e HPC; estrutura de um programa; tipos (INTEGER/REAL/COMPLEX); declaração de variáveis; IMPLICIT NONE; operadores; atribuição; controle de fluxo (IF/SELECT CASE); loops (DO); arrays; array slicing; operações de array (element-wise); funções intrínsecas; subroutines; functions; parâmetros (INTENT); módulos; I/O (READ/WRITE/FORMAT); parameters e constantes.

## numerico-avancado/ — ~16
arrays multidimensionais; álgebra linear; MATMUL e funções de array; alocação dinâmica (ALLOCATABLE); pointers; derived types; interfaces; operator overloading; recursão; precisão (KIND); manipulação de matrizes; BLAS/LAPACK (visão geral); formatação numérica; arquivos.

## hpc-ecossistema/ — ~10
Fortran 77 vs 90 vs modern (2008/2018); OpenMP (paralelismo); MPI (visão geral); coarrays; compiladores (gfortran/ifort); performance e otimização; interop com C; boas práticas; migração de código legado.
