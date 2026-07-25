# Ruby on Rails (+ Ruby) — plano de treinamento (o que precisa constar)

> ⏳ **GERANDO** (fase 9) — `ruby_experts`, 2 famílias (ruby, rails), dados em `treinamento_ruby/`.


**Objetivo**: cobrir tudo pra o expert responder sobre a linguagem Ruby E o framework Rails.
**Expert sugerido**: `ruby_experts` (fase 9). **Total estimado**: ~225 lições (2 famílias, 15 subsetores).
**Convenção**: `treinamento_ruby/<família>/<subsetor>/*.md` → path = [família, subsetor].

## ruby/ (a linguagem) — ~81 lições
### sintaxe (~12)
variáveis e tipos; números e strings; símbolos; interpolação; operadores; condicionais (`if`/`unless`/`case`); loops (`while`/`until`/`times`); ranges; `puts`/`print`/`p`; comentários; convenções (Ruby style guide); truthiness (só `nil`/`false`).
### colecoes (~12)
arrays; hashes; blocks; `yield`; procs; lambdas; proc vs lambda; `map`/`select`/`reject`; `reduce`/`inject`; Enumerable; iteradores customizados; destructuring e splat.
### oop (~16)
classes e objetos; `initialize`/`self`; `attr_accessor`; herança; módulos e mixins (`include`/`extend`); visibilidade (private/protected); métodos de classe; constantes; duck typing; Comparable/Enumerable mixins; sobrecarga de operadores; `respond_to?`; singleton methods; refinements; `Struct`/`OpenStruct`; composição vs herança.
### strings (~10)
manipulação de strings; métodos essenciais; formatação (`format`/`%`); regex (`=~`/`match`); `gsub`/`sub`; `scan`; `split`/`join`; heredoc; encoding e UTF-8; strings imutáveis (frozen).
### excecoes (~9)
`begin`/`rescue`/`ensure`; hierarquia de exceções; `raise`; `retry`; exceções customizadas; `rescue` em métodos; `ensure` e recursos; `throw`/`catch`; boas práticas de tratamento de erros.
### metaprogramacao (~12)
`send`/`public_send`; `define_method`; `method_missing`; `respond_to_missing?`; `instance_variable_get/set`; `class_eval`/`instance_eval`; hooks (`included`/`inherited`); `const_get`; abrir classes (monkey patching); DSLs internas; reflection; `ObjectSpace`.
### modulos-gems (~10)
`require`/`require_relative`/`load`; namespacing com módulos; criar uma gem; `Gemfile` e Bundler; versionamento semântico; RubyGems e publicar; autoload/Zeitwerk; `Comparable` como módulo; pattern matching (`case/in`); IO e arquivos.

## rails/ (o framework) — ~144 lições
### fundamentos (~14)
filosofia (convention over configuration); `rails new` e estrutura; `rails server`/console; MVC no Rails; ciclo de uma requisição; ambientes (dev/test/prod); Rails 7/8 novidades; asset pipeline (Propshaft/Sprockets); importmaps; `rails generate`; `bin/rails` e rake tasks; credentials e secrets; configuração (`config/`); logging.
### routing (~10)
`routes.rb`; rotas RESTful e `resources`; rotas aninhadas; membros e coleções; rotas nomeadas e path helpers; constraints; redirects; namespaces e scopes; rotas para APIs; `direct`/`resolve`.
### controllers (~10)
controllers e actions; params e strong parameters; before/after actions (filters); sessões e cookies; flash; respond_to e formatos; renderização e redirect; tratamento de exceções (`rescue_from`); status codes; streaming.
### views (~12)
ERB; layouts; partials; helpers; form_with e form helpers; view helpers de link/URL; `content_for`/`yield`; collections e `render`; formatação (number/date helpers); componentes (ViewComponent); assets nas views; escaping e segurança (HTML).
### activerecord (~16)
models e a convenção; migrations; tipos de coluna e índices; CRUD; validações; validações customizadas; callbacks; `belongs_to`/`has_many`; `has_many :through`; `has_and_belongs_to_many`; polimorfismo; enums; transações; concerns; STI (single table inheritance); dirty tracking.
### queries (~12)
`where` e condições; `order`/`limit`/`offset`; scopes; `find`/`find_by`; eager loading (`includes`/`preload`/`eager_load`); o problema N+1; agregações (`count`/`sum`/`group`); joins; subqueries; `pluck`/`select`; SQL cru quando preciso; paginação (Kaminari/Pagy).
### recursos (~14)
Action Mailer; Active Storage (uploads); Active Job; caching (fragment/russian doll); I18n (internacionalização); Rails credentials; middleware Rack; Action Text (rich text); Action Mailbox; rate limiting; content security policy; feature flags; rake tasks customizadas; generators customizados.
### frontend (~10)
Hotwire (visão geral); Turbo Drive; Turbo Frames; Turbo Streams; Stimulus (controllers); importmaps vs bundlers (jsbundling); integração com React/Vue; Tailwind no Rails; broadcasts (Turbo + Action Cable); SPA vs Hotwire.
### background-tempo-real (~9)
Active Job (API); Sidekiq; filas e prioridades; jobs recorrentes; Action Cable (websockets); channels; broadcasting; Solid Queue/Solid Cable; retries e erros em jobs.
### auth (~9)
autenticação com `has_secure_password`; o generator de auth do Rails 8; Devise (setup); Devise (customização); autorização com Pundit; autorização com CanCanCan; roles e permissões; OmniAuth (login social); segurança (CSRF/sessões).
### api (~9)
API-only mode; serialização (Jbuilder/ActiveModel::Serializers/Alba); versionamento de API; autenticação por token/JWT; CORS; paginação em APIs; tratamento de erros JSON; documentação (OpenAPI); GraphQL no Rails (visão geral).
### testes (~12)
Minitest; RSpec (setup); testes de model; testes de request/controller; testes de sistema (Capybara); fixtures; factories (FactoryBot); mocking e stubbing; VCR (APIs externas); cobertura (SimpleCov); TDD/BDD no Rails; testes de jobs e mailers.
### deploy-performance (~7)
deploy com Kamal; Docker no Rails; Capistrano; variáveis de ambiente e credentials; monitoramento e APM; otimização de queries e caching; boas práticas de produção.

## Observação
Ruby on Rails = **duas coisas**: a linguagem Ruby (família `ruby/`) e o framework Rails
(família `rails/`). Separar em famílias distintas deixa o roteamento de subsetor mais preciso
(pergunta sobre `each`/blocks → ruby; pergunta sobre `has_many`/migrations → rails).
