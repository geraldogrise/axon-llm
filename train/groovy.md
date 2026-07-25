# Groovy — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Groovy.
**Expert sugerido**: família em `jvm_experts`. **Total est.**: ~65 lições.
**Convenção**: `treinamento_groovy/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~28
o que é Groovy (na JVM); sintaxe e diferenças de Java; tipagem dinâmica e `def`; opcional typing; strings (GString e interpolação); operadores (Elvis `?:`, safe `?.`, spaceship); listas; maps; ranges; closures; controle de fluxo; loops; funções e métodos; truthiness; multi-line strings; regex nativo; comentários.

## oop-dinamico/ — ~20
classes e propriedades; POGOs; construtores; herança; traits; operator overloading; metaprogramação; metaClass; `methodMissing`/`propertyMissing`; categorias; AST transformations; `@ToString`/`@EqualsAndHashCode`; `@Builder`; delegation; mixins; dynamic dispatch; interop com Java; GDK (métodos adicionais).

## ecossistema/ — ~17
Gradle (build scripts em Groovy); Jenkins pipelines (Jenkinsfile); Spock (testes); DSLs em Groovy; Grails (framework web, visão geral); processar JSON (JsonSlurper); processar XML (XmlSlurper); scripts e automação; arquivos e I/O; execução de comandos; Groovlets; templating; comparação com Kotlin; boas práticas.
