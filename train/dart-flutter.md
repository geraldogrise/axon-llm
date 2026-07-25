# Dart + Flutter — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder sobre a linguagem Dart e o framework Flutter.
**Expert sugerido**: `dart_experts`. **Total est.**: ~130 lições.
**Convenção**: `treinamento_dart/<família>/<subsetor>/*.md` → path = [família, subsetor].

## dart/ (linguagem) — ~45
### sintaxe (~12)
variáveis (`var`/`final`/`const`); tipos e inferência; null safety; operadores; string interpolation; controle de fluxo; loops; funções; arrow functions; argumentos nomeados e opcionais; `late`; cascade (`..`).
### oop (~16)
classes e construtores; named constructors; factory constructors; herança; mixins; interfaces (implicit); abstract classes; getters/setters; static members; enums; extension methods; operator overloading; generics; `covariant`; sealed classes; records.
### async-coleções (~17)
Future e async/await; Stream; async generators; error handling; collections (List/Map/Set); collection methods (map/where/fold); spread e collection if/for; iterables; `Iterable` customizado; isolates (concorrência); typedef; callable classes; pattern matching; destructuring; libraries e imports.

## flutter/ (framework) — ~85
### fundamentos (~18)
o que é o Flutter; widgets (tudo é widget); StatelessWidget; StatefulWidget; BuildContext; hot reload; MaterialApp e Scaffold; layout (Container/Row/Column); alinhamento e flex; Stack e Positioned; Text e estilos; imagens e assets; botões; padding e margin; constraints; árvore de widgets; keys.
### estado-navegacao (~20)
setState; lifting state up; InheritedWidget; Provider; Riverpod; Bloc/Cubit; GetX; navegação (Navigator); rotas nomeadas; go_router; passar dados entre telas; deep linking; state management comparado; ChangeNotifier; ValueNotifier; Consumer.
### ui-avancada (~24)
ListView e builders; GridView; formulários e validação; TextField; gestos e GestureDetector; animações (implicit); AnimationController; Hero; custom painters; themes; responsividade; SliverAppBar; TabBar; Drawer; BottomNavigationBar; dialogs e bottom sheets; Material 3; Cupertino (iOS); fontes e ícones; SafeArea.
### dados-plataforma (~23)
HTTP e APIs (http/dio); JSON serialization; persistência (shared_preferences); SQLite (sqflite); Hive; Firebase (auth/firestore); notificações; câmera e permissões; platform channels; internacionalização; testes (widget/unit/integration); build e deploy (Android/iOS); CI/CD; performance; arquitetura (Clean/MVVM); packages (pub.dev).
