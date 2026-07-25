# Julia — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Julia (computação científica).
**Expert sugerido**: família em `data_science_experts`. **Total est.**: ~70 lições.
**Convenção**: `treinamento_julia/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~28
o que é Julia (JIT/LLVM); variáveis e tipos; sistema de tipos; inferência; operadores; strings; controle de fluxo; loops e comprehensions; funções; funções anônimas; múltiplos retornos; broadcasting (`.`); tipos abstratos e concretos; structs; parametric types; multiple dispatch (o coração da linguagem); métodos; type stability; conversão e promoção.

## numerico-dados/ — ~24
arrays e matrizes; indexação; álgebra linear; operações vetorizadas; DataFrames.jl; ler/escrever dados (CSV.jl); Plots.jl; estatística (Statistics/Distributions); otimização; equações diferenciais (DifferentialEquations.jl); interpolação; FFT; números aleatórios; performance de arrays; views vs cópias; missing values.

## avancado-ecossistema/ — ~18
pacotes (Pkg); ambientes e Project.toml; macros e metaprogramação; expressões (`:()`); performance e benchmarking (BenchmarkTools); paralelismo; distributed computing; GPU (CUDA.jl); interop (PyCall/ccall); testes (Test); módulos; Flux.jl (deep learning); JuMP (otimização); Pluto/Jupyter; boas práticas; comparação com Python/R.
