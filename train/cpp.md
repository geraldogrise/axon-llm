# C++ — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre C++.
**Expert sugerido**: família `cpp` (em `systems_experts` ou expert próprio). **Total est.**: ~140 lições.
**Convenção**: `treinamento_cpp/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~40
### sintaxe (~12)
diferenças de C; tipos e `auto`; referências (`&`); `const` e `constexpr`; namespaces; I/O com streams (`cin`/`cout`); condicionais e loops; range-based for; funções e overloading; argumentos padrão; `nullptr`; inicialização uniforme (`{}`).
### oop (~16)
classes e objetos; construtores e destrutores; encapsulamento; herança; polimorfismo e funções virtuais; classes abstratas; sobrecarga de operadores; construtor de cópia; regra dos 3/5/0; herança múltipla; virtual destructors; friend; static members; `this`; slicing; RTTI e `dynamic_cast`.
### memória (~12)
ponteiros e referências; `new`/`delete`; smart pointers (`unique_ptr`); `shared_ptr` e `weak_ptr`; RAII; move semantics; rvalue references; `std::move`; copy vs move; memory leaks; ponteiros brutos vs smart; alocadores.

## stl/ — ~30
containers sequenciais (`vector`/`array`/`deque`/`list`); containers associativos (`map`/`set`/`unordered_map`); container adapters (`stack`/`queue`/`priority_queue`); iteradores; algoritmos (`sort`/`find`/`transform`); `std::string`; `string_view`; lambdas; `std::function`; functors; `std::pair`/`tuple`; `optional`/`variant`/`any`; ranges (C++20); `span`; `chrono`; regex; `std::algorithm` avançado.

## templates-generico/ — ~25
templates de função; templates de classe; especialização; template parameters; variadic templates; SFINAE; type traits; concepts (C++20); metaprogramação; CRTP; template template parameters; perfect forwarding; `decltype`/`declval`; fold expressions; policy-based design.

## avancado/ — ~35
concorrência (`std::thread`); mutex e locks; `std::atomic`; condition variables; futures e promises; `async`; coroutines (C++20); exceções e `noexcept`; tratamento de erros; namespaces e ODR; compilação e linkagem; CMake; módulos (C++20); undefined behavior; otimização e performance; cache-friendly code; padrões de projeto em C++; C++11/14/17/20/23 (evolução); interop com C; debugging; boas práticas modernas (Core Guidelines).
