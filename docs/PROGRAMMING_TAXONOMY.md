# axon-lang — Programming area (taxonomy plan)

Plan for adding a new top-level **area `programacao`** to axon-lang, covering the
**top 20 most popular languages** plus a rich **frontend** map. The area splits by
**language** (sector), then **framework family** (subsector), then the **specific
framework** (deeper leaves). A Portuguese question about, say, Spring Security routes
to `programacao → java → spring → security`, and the RAG stage answers from **only
that compartment**.

This mirrors the existing areas (`matematica`, `fisica`, …); it is added, not a rewrite.

> **Before you build this, read §8 — "Reality check: can it actually program?"** It
> is an honest assessment of what this architecture can and cannot do.

---

## 1. How it maps to axon-lang

| Level | Role | Example |
|-------|------|---------|
| 0 | **area** | `programacao` |
| 1 | **sector** = language | `python`, `javascript`, `java` |
| 2 | **subsector** = framework family / topic | `spring`, `frontend`, `orm`, `build` |
| 3+ | specific framework | `spring-boot`, `react`, `hibernate`, `jsf` |

The router is **compartmentalized**: at each node only that node's classifier runs, and
only the chosen branch is descended. Twenty languages and hundreds of frameworks still
cost just **one classifier per level** at query time.

---

## 2. Top 20 languages (popularity, ~2024–2025)

Order blends TIOBE / Stack Overflow / GitHub Octoverse. Each is a **sector**:

`python`, `javascript`, `java`, `c`, `cpp`, `csharp`, `typescript`, `php`, `go`,
`rust`, `kotlin`, `swift`, `ruby`, `sql`, `r`, `matlab`, `dart`, `scala`, `perl`,
`elixir`.

---

## 3. The tree (Java + JS/TS expanded; others follow the same shape)

```
programacao
├─ python
│  ├─ web    → django, flask, fastapi
│  ├─ data   → pandas, numpy, scikit-learn
│  ├─ ml     → pytorch, tensorflow
│  └─ async  → celery, asyncio
├─ javascript
│  ├─ frontend → react, vue, angular, svelte, solid
│  ├─ meta     → nextjs, nuxt, remix, astro
│  ├─ backend  → nodejs, express, nestjs
│  ├─ styling  → tailwind, bootstrap, sass
│  └─ build    → vite, webpack
├─ java
│  ├─ spring → spring-boot, spring-batch, spring-mvc, spring-security, spring-data, spring-cloud
│  ├─ jakarta-ee → jsf, servlet, ejb
│  ├─ web    → struts, primefaces, vaadin
│  ├─ quarkus · micronaut
│  ├─ orm    → hibernate, jpa
│  ├─ build  → maven, gradle
│  └─ testing → junit, mockito
├─ c            → glib, sdl · posix, win32
├─ cpp          → qt · boost, stl · unreal
├─ csharp       → aspnet-core, blazor · wpf, winforms, maui · unity · entity-framework
├─ typescript   → angular, nestjs, deno (+ shares JS frontend)
├─ php          → laravel, symfony, wordpress, codeigniter
├─ go           → gin, echo, fiber, beego
├─ rust         → actix, tokio, axum, rocket, bevy
├─ kotlin       → android, jetpack-compose, ktor, spring-kotlin
├─ swift        → swiftui, uikit, vapor
├─ ruby         → rails, sinatra, hanami
├─ sql          → postgresql, mysql, oracle, sqlserver, mongodb
├─ r            → tidyverse, ggplot2, shiny
├─ matlab       → simulink, toolboxes
├─ dart         → flutter
├─ scala        → akka, play, spark
├─ perl         → catalyst, mojolicious
└─ elixir       → phoenix, ecto
```

**Depth is free** — deepen any leaf when it needs sub-topics
(e.g. `spring-security → oauth2 / jwt / ldap`).

---

## 4. Frontend map (the "frontend também" you asked for)

Frontend is not one language — it spans several. axon-lang keeps each piece under its
language sector, so routing stays clean, but here is the cross-cutting view:

| Frontend tech | Lives under | Leaf |
|---|---|---|
| React, Vue, Angular, Svelte, Solid | `javascript → frontend` | `react`, `vue`, … |
| Next.js, Nuxt, Remix, Astro (meta) | `javascript → meta` | `nextjs`, … |
| Angular (TS-first), NestJS | `typescript → frontend/backend` | `angular`, `nestjs` |
| Tailwind, Bootstrap, Sass (CSS) | `javascript → styling` | `tailwind`, … |
| Flutter | `dart → flutter` | `flutter` |
| SwiftUI, UIKit | `swift` | `swiftui`, `uikit` |
| Jetpack Compose | `kotlin` | `jetpack-compose` |
| Blazor | `csharp → web` | `blazor` |
| Phoenix LiveView | `elixir → phoenix` | `phoenix` |

---

## 5. Ready-to-use taxonomy (Python)

Same nested-dict format the collector accepts (leaf = a Wikipedia search query).
Framework names are proper nouns, so they work on `pt.wikipedia.org`; for thin PT
coverage, add `en.wikipedia.org` (see §8). Save as `examples/taxonomy_programming.py`.

```python
PROGRAMMING = {
    "programacao": {
        "python": {
            "web":   {"django": "Django (framework)", "flask": "Flask (framework web)", "fastapi": "FastAPI"},
            "data":  {"pandas": "Pandas (software)", "numpy": "NumPy", "scikit-learn": "Scikit-learn"},
            "ml":    {"pytorch": "PyTorch", "tensorflow": "TensorFlow"},
            "async": {"celery": "Celery (software)", "asyncio": "Asyncio"},
        },
        "javascript": {
            "frontend": {"react": "React (biblioteca JavaScript)", "vue": "Vue.js",
                         "angular": "Angular (framework)", "svelte": "Svelte", "solid": "SolidJS"},
            "meta":     {"nextjs": "Next.js", "nuxt": "Nuxt.js", "remix": "Remix (framework)",
                         "astro": "Astro (framework web)"},
            "backend":  {"nodejs": "Node.js", "express": "Express.js", "nestjs": "NestJS"},
            "styling":  {"tailwind": "Tailwind CSS", "bootstrap": "Bootstrap (framework)", "sass": "Sass (linguagem)"},
            "build":    {"vite": "Vite", "webpack": "Webpack"},
        },
        "java": {
            "spring": {"spring-boot": "Spring Boot", "spring-batch": "Spring Batch",
                       "spring-mvc": "Spring Framework", "spring-security": "Spring Security",
                       "spring-data": "Spring Data", "spring-cloud": "Spring Cloud"},
            "jakarta-ee": {"jsf": "JavaServer Faces", "servlet": "Java Servlet", "ejb": "Enterprise JavaBeans"},
            "web":     {"struts": "Apache Struts", "primefaces": "PrimeFaces", "vaadin": "Vaadin"},
            "quarkus": "Quarkus",
            "micronaut": "Micronaut (framework)",
            "orm":     {"hibernate": "Hibernate (framework)", "jpa": "Jakarta Persistence"},
            "build":   {"maven": "Apache Maven", "gradle": "Gradle"},
            "testing": {"junit": "JUnit", "mockito": "Mockito"},
        },
        "c": {
            "libs":     {"glib": "GLib", "sdl": "Simple DirectMedia Layer"},
            "sistemas": {"posix": "POSIX", "win32": "API do Windows"},
        },
        "cpp": {
            "gui":  {"qt": "Qt (framework)"},
            "libs": {"boost": "Boost (biblioteca)", "stl": "Standard Template Library"},
            "game": {"unreal": "Unreal Engine"},
        },
        "csharp": {
            "web":     {"aspnet-core": "ASP.NET Core", "blazor": "Blazor"},
            "desktop": {"wpf": "Windows Presentation Foundation", "winforms": "Windows Forms", "maui": ".NET MAUI"},
            "game":    {"unity": "Unity (motor de jogo)"},
            "orm":     {"entity-framework": "Entity Framework"},
        },
        "typescript": {
            "frontend": {"angular": "Angular (framework)", "react-ts": "React (biblioteca JavaScript)"},
            "backend":  {"nestjs": "NestJS", "deno": "Deno"},
            "lang":     {"typescript": "TypeScript"},
        },
        "php":    {"laravel": "Laravel", "symfony": "Symfony", "wordpress": "WordPress", "codeigniter": "CodeIgniter"},
        "go":     {"gin": "Gin (framework web)", "echo": "Echo (framework)", "fiber": "Fiber (framework)", "beego": "Beego"},
        "rust":   {"actix": "Actix", "tokio": "Tokio (biblioteca)", "axum": "Axum (framework)",
                   "rocket": "Rocket (framework web)", "bevy": "Bevy (motor de jogo)"},
        "kotlin": {"android": "Android (sistema operacional)", "jetpack-compose": "Jetpack Compose",
                   "ktor": "Ktor", "spring-kotlin": "Spring Framework"},
        "swift":  {"swiftui": "SwiftUI", "uikit": "UIKit", "vapor": "Vapor (framework web)"},
        "ruby":   {"rails": "Ruby on Rails", "sinatra": "Sinatra (software)", "hanami": "Hanami (framework)"},
        "sql":    {"postgresql": "PostgreSQL", "mysql": "MySQL", "oracle": "Oracle Database",
                   "sqlserver": "Microsoft SQL Server", "mongodb": "MongoDB"},
        "r":      {"tidyverse": "Tidyverse", "ggplot2": "Ggplot2", "shiny": "Shiny (R)"},
        "matlab": {"simulink": "Simulink", "toolbox": "MATLAB"},
        "dart":   {"flutter": "Flutter (software)"},
        "scala":  {"akka": "Akka (toolkit)", "play": "Play Framework", "spark": "Apache Spark"},
        "perl":   {"catalyst": "Catalyst (software)", "mojolicious": "Mojolicious"},
        "elixir": {"phoenix": "Phoenix (framework web)", "ecto": "Ecto (biblioteca)"},
    }
}
```

---

## 6. Where to collect from (sources)

Wikipedia is only the starting point. For real programming depth, pull from docs,
tutorials, forums and git — the **multi-source collector already downloads HTML and
PDF** (`corpus.fetch_url_text` → download → read → discard), so most of these plug in as
`specs = [(path, url)]` today; git and sitemap crawling are two small helpers to add.

| Source | Gives | Ingest via | Honest caveat |
|--------|-------|-----------|---------------|
| **Official docs** (python.org, spring.io, react.dev, docs.oracle.com, go.dev) | authoritative reference | `fetch_url_text` + the site's sitemap.xml | usually open; respect robots.txt + rate limit (`delay`) |
| **MDN Web Docs** | web / frontend reference | `fetch_url_text` | content is CC-BY-SA → keep attribution |
| **Stack Overflow / Stack Exchange** | Q&A, **error → fix** | official **data dump** (archive.org) or API — *not* scraping | CC-BY-SA attribution; best source for §9.3 error lookup |
| **GitHub repos** | real code, READMEs, `docs/`, examples | `git clone --depth 1` → walk → `read_file` → `to_trash` | respect each repo's license; index selectively |
| **W3Schools** | beginner tutorials | `fetch_url_text` | ⚠️ its **ToS forbids automated scraping/republishing** — use for personal study only, and prefer MDN / official docs which are openly licensed |
| **DevDocs / Read the Docs** | aggregated framework docs | `fetch_url_text` | mostly open; check each project's license |
| **Wikipedia / Wikibooks** | concept overviews | existing `corpus.mw_*` | CC-BY-SA |

Rules that keep this clean and legal:
- **Prefer openly-licensed sources** (MDN, official docs, SO dump, permissive repos).
  Where a site's ToS forbids scraping (W3Schools), don't build a public dataset from it.
- **Be polite**: the collector's `delay` + persistent dedup already avoid hammering a
  host or re-pulling the same URL across runs.
- **Two helpers to add**: `collect_git(repo_url, path_labels)` (clone → index → discard)
  and a `sitemap` crawler that turns a docs site into `(path, url)` specs.

---

## 7. Official docs & forums per framework

Each framework has its **own canonical documentation site** — higher quality and more
current than Wikipedia, and the **primary RAG source** for programming. General forums
(Stack Overflow, Reddit) cover *all* frameworks at once — index them once and route by
tag/framework. Starter map (feed these to the collector; crawl each docs `sitemap.xml`
for full coverage):

| Framework | Official docs | Community / forum |
|-----------|---------------|-------------------|
| Angular | `angular.dev` | Stack Overflow `[angular]`, r/angular, Angular Discord |
| React | `react.dev` | Stack Overflow `[reactjs]`, r/reactjs |
| Vue | `vuejs.org` | `forum.vuejs.org`, Stack Overflow |
| Svelte | `svelte.dev` | Stack Overflow, Svelte Discord |
| Spring | `docs.spring.io`, `spring.io/guides` | Stack Overflow `[spring]` |
| Django | `docs.djangoproject.com` | `forum.djangoproject.com`, Stack Overflow |
| Flask | `flask.palletsprojects.com` | Stack Overflow |
| FastAPI | `fastapi.tiangolo.com` | GitHub Discussions, Stack Overflow |
| Node.js / Express | `nodejs.org/docs`, `expressjs.com` | Stack Overflow |
| .NET / ASP.NET | `learn.microsoft.com/dotnet` | Microsoft Q&A, Stack Overflow |
| Laravel | `laravel.com/docs` | Laracasts, Stack Overflow |
| Go | `go.dev/doc` | Go Forum, Stack Overflow |
| Rust | `doc.rust-lang.org` | `users.rust-lang.org`, Stack Overflow |
| Flutter | `docs.flutter.dev` | Stack Overflow, r/FlutterDev |
| HTML/CSS/JS | `developer.mozilla.org` (MDN) | — |

Wiring it up: keep a small map `{framework_path: docs_root_url}`, expand each via its
sitemap into many `(path, url)` specs, then `Collector.stream_train_specs(...)`
downloads → trains → discards, exactly like the other areas. Licensing still applies
(§6) — official docs and MDN are generally fine; forums via their open dumps/APIs.

---

## 8. Collect + train

Same pipeline as the other areas:

```bash
# 1) Router training (discard-after-train, dedup ledger, resumable)
python examples/train_linear.py           # after adding PROGRAMMING into its AREAS

# 2) Knowledge base for answering (gzip + int8 -> kb.json.gz)
AXON_BUDGET=10800 AXON_PER_LEAF=40 python examples/build_knowledge_base.py
```

Everything the other areas get, programming gets for free: discard-after-train,
persistent dedup, interleaved curriculum, compressed KB.

---

## 7. Answering (RAG)

1. `router.route("como configurar o Spring Security?")` → `programacao → java → spring → spring-security`
2. `kb.retrieve(query, path_prefix=that_path)` → top chunks **from that compartment only**
3. `kb.answer(query, router, multi=2)` for cross-framework questions.

---

## 8. Reality check: can axon-lang actually PROGRAM?

**Honest answer: no — not the way a coding assistant writes code. But it can become a
genuinely useful programming Q&A / documentation assistant.** Be clear-eyed about this
before investing collection time.

**What axon-lang is:** a **router (classifier) + RAG (retrieval)**. Given a question it
(1) identifies the language/framework and (2) retrieves the most relevant documentation
passages from that compartment and returns them. It is a smart, compartmentalized
**search + extractive answer** over the text it collected.

**What that does well for programming** ✅
- *"What is Spring Security and what is it for?"* → routes correctly, returns the explanation.
- *"Difference between Maven and Gradle?"* → retrieves and contrasts both.
- Points you to the right concept/snippet from indexed docs — like a focused, per-framework doc search.

**What it cannot do** ❌
- Write a new, correct, compilable program from a spec.
- Debug *your* code or reason about *your* specific logic.
- Produce novel code that isn't essentially a retrieved chunk.

**Why.** Generating working code needs a **large generative language model** (billions of
parameters) pretrained on **huge code corpora** with **GPU compute** — CodeLlama,
StarCoder, GPT-class. The `AxonLM` in pyaxon is a tiny transformer trained from scratch
on small CPU data: it can learn to babble text, not to emit correct programs. The
routing+RAG design **retrieves**; it does not reason or generate at that level.

**Two realistic paths:**

- **Path A — doc/knowledge assistant (fits this project, achievable).** RAG over
  official docs + curated code examples, per framework. You get a compartmentalized
  "answer machine" that quotes the right documentation. This is exactly what the
  architecture is good at. Recommended.
- **Path B — real code generation (different project).** Integrate an existing
  pretrained open code model. That gets you actual code, but it is a large model on a
  GPU — it breaks the "from scratch, educational, CPU" spirit of pyaxon. Honest, but a
  separate track.

**Recommendation:** build the programming area as a **routing + RAG documentation
assistant** (Path A). Treat "write my program for me" as out of scope for a from-scratch
CPU model. If you later want true code generation, plug in a pretrained code model
rather than trying to train one from zero.

---

## 9. Git repositories as a source — and can it fix bugs / suggest best practices?

You asked: *fix errors/bugs, help with best practices, by downloading from git repos,
and so on?* Splitting it honestly, because the parts have very different feasibility:

### 9.1 Git repos as a data source — ✅ yes, great idea
Cloning repos and indexing them is the **single best way** to enrich this system for
programming. A repo gives you real code, READMEs, `docs/`, `CONTRIBUTING.md`, examples,
and issue discussions — exactly the material a doc-assistant should retrieve.

- `corpus.read_file` already reads txt/md/py/pdf/docx/... — a git ingester is a thin
  new helper: `git clone --depth 1` → walk files → `read_file` each → index into the
  KnowledgeBase under `programacao → <lang> → <framework>` → optionally `to_trash` the
  clone (keep only the passages). Dedup by content hash is already built in.
- Index **selectively**: READMEs, `docs/`, top-level source, examples — not whole
  monorepos (noise + size). Respect licenses if you ever redistribute the text.

### 9.2 Best-practices help — ✅ mostly yes (retrieval + real tools)
Two complementary ways, both realistic:
- **Retrieval**: index style guides, "getting started", and framework best-practice
  docs; a question like *"how should I structure a Spring Boot project?"* returns real
  guidance and example layouts. ✅
- **Deterministic tools (recommended pairing)**: for *detecting* issues, real
  **linters/formatters** (ESLint, Checkstyle, ruff, clang-tidy) are far more reliable
  than any from-scratch model. Let the RAG **explain** a rule; let the linter **find**
  the violation. This hybrid is honest and actually works.

### 9.3 Fixing errors / bugs — ⚠️ split answer
- **Known error message → documented fix**: ✅ achievable. Paste a stack trace / error
  string → route to the framework → retrieve the closest known cause + fix from indexed
  Q&A / issues. This is a **compartmentalized Stack Overflow lookup** — retrieval, not
  generation — and it genuinely helps for common, well-documented errors.
- **Rewriting *your* specific buggy code into correct code**: ❌ not with this
  architecture. That needs a model that *understands your code and generates a patch* —
  a large pretrained code LLM (Path B in §8), GPU-scale. The routing+RAG here retrieves
  and explains; it does not reason over your program or synthesize a correct fix.

### 9.4 Bottom line
With git repos + linters you can build a strong **"programming knowledge + best-practice
assistant"**: it points you to real code, explains concepts, matches known errors to
known fixes, and defers detection to proper tools. What it will **not** do is
autonomously write or repair your programs — that remains a large-generative-model job.
Set the expectation there and the feature is both useful and deliverable.

---

## 10. Caveats specific to programming content

- **Wikipedia is thin on niche frameworks** (Spring Batch, PrimeFaces, Struts). For real
  depth, feed the **multi-source collector** official docs and PDF books:
  `Collector.stream_train_specs(...)` already reads HTML and PDF via
  `corpus.fetch_url_text` (download → read → discard).
- **Code-heavy text.** The RAG `clean_text` filter was tuned for prose and treats high
  symbol density as noise — it will drop code snippets. To index code, add a rule that
  keeps lines that look like code (indentation, `;{}()`), instead of discarding them.
- **Language gate.** axon-lang identifies Portuguese first. PT Wikipedia covers the big
  frameworks; for the long tail, add `en.wikipedia.org` as a source or relax the gate
  for the `programacao` area.
