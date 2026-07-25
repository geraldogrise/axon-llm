# Objective-C — plano de treinamento (o que precisa constar)

**Objetivo**: cobrir tudo pra o expert responder qualquer pergunta sobre Objective-C (Apple/legado).
**Expert sugerido**: família em `swift_experts` ou `apple_experts`. **Total est.**: ~75 lições.
**Convenção**: `treinamento_objc/<família>/<subsetor>/*.md` → path = [família, subsetor].

## core/ — ~30
sintaxe e diferenças de C; tipos e `id`; a sintaxe de mensagens (`[obj metodo]`); NSObject; classes (`@interface`/`@implementation`); métodos de instância e classe; propriedades (`@property`); `@synthesize`; herança; protocolos (`@protocol`); categorias; extensions; `nil` e nil messaging; `BOOL`/`YES`/`NO`; selectors (`SEL`); `self`/`super`; inicializadores; blocks; enums e structs.

## foundation/ — ~25
NSString e NSMutableString; NSArray/NSMutableArray; NSDictionary/NSMutableDictionary; NSNumber; NSData; NSDate; coleções e enumeração; fast enumeration; formatação; NSError e tratamento de erros; NSNotification; KVC (Key-Value Coding); KVO (Key-Value Observing); NSCoding; NSFileManager; NSURLSession (rede); JSON (NSJSONSerialization); NSPredicate; boxing/unboxing.

## memoria-avancado/ — ~20
gerenciamento de memória; ARC (Automatic Reference Counting); retain/release (MRC legado); strong/weak/assign; retain cycles; autorelease pools; blocks e captura; `__weak`/`__strong`; delegation pattern; target-action; GCD (Grand Central Dispatch); threads; runtime dinâmico; method swizzling; associated objects; interoperabilidade com Swift; bridging; Cocoa/UIKit (visão geral); boas práticas.
