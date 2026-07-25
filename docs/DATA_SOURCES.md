# axon-lang — curated Portuguese data sources

Reputable, quality, **Portuguese-language** sources to feed the collector beyond
Wikipedia. axon-lang gates Portuguese first (`langid`), so PT sources fit directly.
Grouped by **openness + reputation + educational quality**. Tier 1 is safe to collect
automatically; Tier 2 is reputable but has Terms of Service that restrict scraping.

---

## Tier 1 — open + high reputation (collect freely)

| Source | URL | Covers | Why it qualifies |
|--------|-----|--------|------------------|
| Wikilivros | `pt.wikibooks.org` | all subjects (didactic) | CC-BY-SA — **already supported** (`corpus.mw_*`) |
| Wikisource | `pt.wikisource.org` | literature, primary texts | CC-BY-SA — **already supported** |
| Wikcionário | `pt.wiktionary.org` | Portuguese, grammar, etymology | CC-BY-SA — already supported |
| SciELO | `scielo.br` | sciences (bio, chem, physics) | **peer-reviewed**, open access |
| Domínio Público (MEC) | `dominiopublico.gov.br` | literature, textbooks (PDF) | public domain (government) |
| **eduCAPES** | `educapes.capes.gov.br` | all subjects (OER) | government open educational resources repository |
| **BIOE** (Objetos Educacionais/MEC) | `objetoseducacionais2.mec.gov.br` | all subjects | curated educational objects, MEC |
| **Portal do Professor** (MEC) | `portaldoprofessor.mec.gov.br` | all subjects, lesson plans | government, pedagogical quality |
| **TV Escola / Canal Futura** | `tvescola.org.br`, `futura.org.br` | all subjects | established educational broadcasters |
| UNIVESP | `univesp.br` | all subjects | public university, open courseware |
| USP – eAulas / e-Disciplinas | `eaulas.usp.br` | all subjects | top Brazilian university |
| Khan Academy PT | `pt.khanacademy.org` | math, physics, chem, bio | world reference, CC-BY-NC-SA (educational use) |
| IMPA / OBMEP | `impa.br`, `obmep.org.br` | mathematics | leading math institute in Brazil |
| Química Nova na Escola | `qnesc.sbq.org.br` | chemistry | journal of the Brazilian Chemistry Society |
| Ciberdúvidas da Língua Portuguesa | `ciberduvidas.iscte-iul.pt` | Portuguese, grammar | classic language-quality reference |
| Academia Brasileira de Letras | `academia.org.br` | literature, language | the language's top institution |
| Instituto Butantan | `butantan.gov.br` | biology / health | reference scientific institute |

> Licensing on eduCAPES/BIOE varies per item (many CC/open) — check the resource before
> redistributing. Government and CC-licensed material is generally fine for study use.

---

## Tier 2 — reputable, but check ToS before automated collection

| Source | Covers | Note |
|--------|--------|------|
| **Nova Escola** (`novaescola.org.br`) | pedagogy, all subjects | high teaching quality; ToS restricts scraping |
| Brasil Escola (`brasilescola.uol.com.br`) | all subjects | school-level, commercial (UOL) |
| Mundo Educação (`mundoeducacao.uol.com.br`) | all subjects | school-level, commercial (UOL) |
| InfoEscola (`infoescola.com`) | all subjects | school-level, commercial |
| Toda Matéria (`todamateria.com.br`) | all subjects | school-level, commercial |

Good content, but these are commercial and typically forbid automated scraping /
redistribution. Use as reference; don't build a public dataset from them. Tier 1 covers
the same ground at university/institutional quality.

---

## Best pick per subject (reputation + open)

| Subject | Sources |
|---------|---------|
| Matemática | IMPA, OBMEP, SBM, Khan PT, Wikilivros |
| Física | SBF, USP Física, UNIVESP, Khan PT |
| Química | Química Nova na Escola (SBQ), SciELO |
| Biologia | SciELO, Instituto Butantan, USP Biociências |
| Português / Literatura | Ciberdúvidas, Academia Brasileira de Letras, Domínio Público, Wikisource |
| História | USP História, Biblioteca Nacional, Arquivo Nacional |

---

## How to plug in

The multi-source collector already reads **HTML and PDF** (`corpus.fetch_url_text`:
download → read → discard). `examples/collect_multisource.py` already combines
Wikipedia + Wikilivros + Wikisource.

1. **MediaWiki sources** (Wikilivros, Wikisource, Wikcionário) — already wired via
   `corpus.MW_SOURCES`; just include them in the taxonomy run.
2. **HTML/PDF sources** (SciELO, Domínio Público, eduCAPES, IMPA…) — build
   `specs = [(path, url)]` and call `Collector.stream_train_specs(...)`. For sites with a
   `sitemap.xml`, crawl it to expand one root URL into many article specs.
3. **Quality/etiquette** — respect `robots.txt`, keep the collector's `delay`, rely on
   the persistent dedup (never re-pulls a URL), and discard text after training.

---

## Honest notes

- **Reading level.** SciELO / IMPA are university-level — great for depth, but harder
  than school-level target text. Mix with Wikilivros / Khan for balance.
- **PDF-heavy sources** (Domínio Público) are handled by `corpus.read_pdf` (download →
  read → trash).
- **Tier 2 ToS.** Do not automate collection from commercial portals; keep them as
  human reference only.
