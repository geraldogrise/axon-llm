# Haskell — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Haskell (funcional puro).
**Expert sugerido**: família em `functional_experts`. **Total est.**: ~80 lições.
**Convenção**: `treinamento_haskell/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~30
o que é programação funcional pura; GHCi; tipos e assinaturas; inferência de tipos; funções; currying; funções de ordem superior; pattern matching; guards; `let`/`where`; recursão; listas; list comprehensions; lazy evaluation; ranges infinitos; tuplas; `Maybe`; tipos algébricos (`data`); type classes; instâncias; `deriving`; polimorfismo; composição de funções (`.`); aplicação (`$`).

## typeclasses-abstracoes/ — ~28
type classes fundamentais (Eq/Ord/Show); Functor; Applicative; Monad; a monad `Maybe`; a monad `Either`; a monad de lista; `do` notation; a monad IO; State monad; Reader/Writer; monad transformers; Foldable; Traversable; Semigroup e Monoid; kinds; type families; GADTs; newtype; functor laws; monad laws.

## avancado-ecossistema/ — ~22
IO e efeitos; manipulação de exceções; records; módulos; Cabal e Stack (build); Hackage; testes (HUnit/QuickCheck); property-based testing; parsers (Parsec/Megaparsec); lenses; concorrência (STM); performance e strictness; `seq` e `$!`; profiling; template Haskell; extensões de linguagem (GHC); web (Servant/Yesod, visão geral); boas práticas; comparação com outras funcionais.
