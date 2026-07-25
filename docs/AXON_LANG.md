# axon-lang — compartmentalized LLM (text → area → sector → subsector)

An architecture in the style of a **hierarchical mixture-of-experts**: the system first
understands that the conversation is in **Portuguese**, then routes the request through a
hierarchy of **area → sector → subsector** (arbitrary nesting), activating only the
minimal fraction of the system (resource savings). Each node in the hierarchy is a
lightweight classifier trained only among its children.

## Components

| Module | Role |
|--------|-------|
| `pyaxon.langid` | identifies the language (gate: "is it Portuguese?") |
| `pyaxon.corpus` | data pipeline: reads files/web, downloads Wikipedia, discards PDF after reading, dedup |
| `pyaxon.router` | compartmentalized hierarchical router (WordNB per node) |

## Flow

```
query → [langid] is it Portuguese? ──no──► ignore / other language
                   │ yes
                   ▼
            [router] area ──► sector ──► subsector   (activates only the path)
```

## Data pipeline (`pyaxon.corpus`)

- **Reads**: `.txt .md .csv .docx .xlsx .pdf` (`read_file`) and web pages (`fetch_url_text`).
- **PDF**: downloads → reads → **discards to the trash** (`to_trash`).
- **Wikipedia**: `wikipedia_extract`, `wikipedia_search`, `wikipedia_category`,
  `collect_articles` (with dedup by title and by content hash).
- **Auto-discovery**: `discover_taxonomy(root)` descends through Wikipedia's
  **subcategories** to *create the subsectors on its own*.
- **Dataset**: `build_wikipedia_dataset(taxonomy, manifest_path=...)` downloads and labels,
  recording the titles already pulled in the *manifest* (avoids duplication across runs).

## Example

```python
import pyaxon as ax

# 1) real dataset (Wikipedia PT), with persistent dedup
taxo = {"matematica": {"algebra": {"linear": "álgebra linear"}}, "fisica": {...}}
recs = ax.corpus.build_wikipedia_dataset(taxo, per_leaf=20, manifest_path="man.json")

# 2) train the hierarchical router
router = ax.router.HierarchicalRouter().fit([(r["text"], r["path"]) for r in recs])

# 3) language gate + routing (only the path is activated)
lid = ax.langid.LanguageIdentifier().fit()
q = "quero calcular o determinante e os autovalores de uma matriz"
if lid.is_portuguese(q):
    print(router.route(q))          # ['matematica', 'algebra', 'linear']
    print(router.last_activated_)   # 3 classifiers (one per level)
```

See `examples/axon_lang.py` (downloads Wikipedia, trains, evaluates per level, routes).

## Result (demonstration, 84 PT Wikipedia articles)

| Level | Accuracy |
|-------|:--------:|
| area  | ~88% |
| sector | ~69% |
| subsector | ~50% |

Clean queries route correctly; the fine-grained levels improve with **more data**
(the user's target is 1000+ pages — increase `PER_LEAF`).

## Streaming collection with a curriculum (`pyaxon.collect.Collector`)

The recommended way to scale: **download → train → delete the text immediately**,
keeping only the URL (CSV) and the knowledge (checkpoint). It learns by **interleaving
subjects** (like a child: Portuguese, math, physics... in rotation), starting
with the **basic subsectors**.

```python
router = ax.router.HierarchicalRouter()
col = ax.collect.Collector("corpus_dir")          # CSV = index + persistent dedup

col.stream_train(router, CURRICULO, target=1000,  # next 1000 (resumable)
                 chunk=4, checkpoint="router.json")
```

- **Resumable**: run it again for the next 1000/10000 — the CSV remembers what was already
  downloaded (title + hash), so it **never pulls the same source twice**.
- **No raw corpus**: the text is ingested (`partial_fit`) and **discarded** immediately;
  only the `sources.csv` (classified URLs) and the `router.json` (the training) remain.
- **Interleaved curriculum**: round-robin across the areas, prioritizing Portuguese.

`sources.csv` index: `id, source, url, title, area, sector, subsector, chars, hash, file`.

See `examples/collect_curriculum.py`.

## Honesty about the scope

This is the **foundation** of a compartmentalized LLM: language identification +
hierarchical routing via lightweight classifiers + a real data pipeline. It is **not**
a generative LLM per subsector (that would mean training one `AxonLM` per leaf, with much
more data/compute) — but the architecture already allows plugging a specialist model into
each router node when desired.
