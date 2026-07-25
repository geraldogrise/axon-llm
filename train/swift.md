# Swift — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Swift (iOS/macOS).
**Expert sugerido**: `swift_experts`. **Total est.**: ~125 lições.
**Convenção**: `treinamento_swift/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~45
### sintaxe (~12)
`let` vs `var`; tipos e inferência; opcionais (`?`/`!`); optional binding (`if let`/`guard`); nil coalescing; string interpolation; controle de fluxo; `switch` e pattern matching; loops e ranges; funções; argumentos e labels; closures.
### tipos (~16)
structs vs classes; propriedades (stored/computed); property observers (`willSet`/`didSet`); enums e associated values; protocolos; protocol extensions; generics; tuples; type casting (`as`/`is`); `Any`/`AnyObject`; value vs reference semantics; nested types; lazy properties; access control; extensions; error handling (`throws`/`do-catch`).
### funcional-avancado (~10)
closures avançadas; `map`/`filter`/`reduce`; higher-order functions; escaping closures; capture lists; opaque types (`some`); result builders; `Result` type; KeyPaths; functional idioms.
### concorrencia (~7)
async/await; `Task`; actors; `@MainActor`; structured concurrency; GCD (legado); Combine (visão geral).

## swiftui/ — ~40
introdução ao SwiftUI; Views e modificadores; layout (VStack/HStack/ZStack); state (`@State`/`@Binding`); `@StateObject`/`@ObservedObject`; `@EnvironmentObject`; listas e navegação; NavigationStack; formulários; gestos; animações; async images; sheets e alerts; grids; data flow; MVVM no SwiftUI; Core Data + SwiftUI; ciclo de vida do app; previews; acessibilidade; widgets.

## uikit-ecossistema/ — ~40
UIKit (visão geral e quando usar); ViewControllers; Auto Layout; Table/CollectionView; delegation e data source; storyboards vs code; ciclo de vida do app iOS; Core Data; URLSession (rede); Codable (JSON); persistência (UserDefaults); Swift Package Manager; testes (XCTest); Xcode e ferramentas; memória (ARC) e retain cycles; distribuição na App Store; push notifications; Combine (aprofundar); interop Objective-C; boas práticas.
