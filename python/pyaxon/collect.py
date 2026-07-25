"""Large-scale corpus collector for axon-lang.

Downloads sources (Wikipedia and, in the future, PDFs/books/web) classifying each
into area/sector/subsector, writing **a CSV index** with the URL of each source.
It is **resumable** and does **persistent dedup**: it never pulls the same source
twice, even across runs (just run again to fetch the next 1000).
"""

import csv
import os
import re

from . import corpus

_CSV_FIELDS = ["id", "source", "url", "title", "area", "sector", "subsector", "chars", "hash", "file"]


def _safe(name):
    return re.sub(r"[^\w.-]+", "_", name).strip("_")[:120] or "untitled"


def specs_from_csv(csv_path):
    """Reuse an already-collected CSV index: returns [(path, url)] from the
    area/sector/subsector columns (labels already defined). Avoids manual relabeling."""
    specs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = [p for p in (row.get("area"), row.get("sector"), row.get("subsector")) if p]
            if path and row.get("url"):
                specs.append((path, row["url"]))
    return specs


def build_mw_specs(taxonomy, source="wikipedia", lang="pt", per_query=8):
    """Build specs (path, url) by searching a MediaWiki source (wikipedia,
    wikibooks, wikisource, wiktionary). Taxonomy leaf = query (or list)."""
    host = corpus.MW_SOURCES[source].format(lang=lang)
    specs = []

    def walk(node, prefix):
        for k, v in node.items():
            p = prefix + [k]
            if isinstance(v, dict):
                walk(v, p)
            else:
                for q in (v if isinstance(v, list) else [v]):
                    try:
                        titles = corpus.mw_search(host, q, limit=per_query * 2)
                    except Exception:  # noqa: BLE001
                        continue
                    for t in titles[:per_query]:
                        specs.append((p, corpus.mw_url(host, t)))
    walk(taxonomy, [])
    return specs


class Collector:
    """Accumulates a labeled corpus in `out_dir`, indexed by `sources.csv`.

    - `out_dir/sources.csv`: index (url, title, area/sector/subsector, file...).
    - `out_dir/<area>/<sector>/<subsector>/<title>.txt`: the text of each source.
    """

    def __init__(self, out_dir, csv_name="sources.csv"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, csv_name)
        self.seen_titles = set()
        self.seen_hashes = set()
        self.seen_urls = set()
        self.count = 0
        self._next_id = 1
        self._load()

    def _load(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(_CSV_FIELDS)
            return
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.seen_titles.add(row["title"])
                if row.get("url"):
                    self.seen_urls.add(row["url"])
                if row.get("hash"):
                    self.seen_hashes.add(row["hash"])
                self.count += 1
                self._next_id = max(self._next_id, int(row["id"]) + 1)

    def _append_row(self, row):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row[k] for k in _CSV_FIELDS])

    def _save_text(self, path_labels, title, text):
        folder = os.path.join(self.out_dir, *[_safe(p) for p in path_labels])
        os.makedirs(folder, exist_ok=True)
        fpath = os.path.join(folder, _safe(title) + ".txt")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(text)
        return os.path.relpath(fpath, self.out_dir)

    def collect_wikipedia(self, taxonomy, target=1000, per_query=15, lang="pt",
                          min_chars=400, log=print):
        """Walk the taxonomy (leaves = list of searches) and download from Wikipedia
        until reaching `target` NEW sources in this run. Writes everything to the CSV.
        Resumable.

        Returns the number of new sources added.
        """
        added = [0]

        def leaf_queries(val):
            return val if isinstance(val, list) else [val]

        def walk(node, prefix):
            if added[0] >= target:
                return
            for key, val in node.items():
                if added[0] >= target:
                    return
                path = prefix + [key]
                if isinstance(val, dict):
                    walk(val, path)
                    continue
                # leaf: one or more search queries
                area = path[0]
                sector = path[1] if len(path) > 1 else ""
                subsector = path[2] if len(path) > 2 else ""
                for query in leaf_queries(val):
                    if added[0] >= target:
                        return
                    try:
                        titles = corpus.wikipedia_search(query, lang, limit=per_query * 3)
                    except Exception as e:  # noqa: BLE001
                        if log:
                            log(f"  search failed ({query}): {e}")
                        continue
                    got = 0
                    for title in titles:
                        if added[0] >= target or got >= per_query:
                            break
                        if title in self.seen_titles:
                            continue
                        self.seen_titles.add(title)
                        try:
                            text = corpus.wikipedia_extract(title, lang)
                        except Exception:  # noqa: BLE001
                            continue
                        if len(text) < min_chars:
                            continue
                        h = corpus._text_hash(text)
                        if h in self.seen_hashes:
                            continue
                        self.seen_hashes.add(h)
                        rel = self._save_text(path, title, text)
                        url = f"https://{lang}.wikipedia.org/wiki/" + title.replace(" ", "_")
                        self._append_row({
                            "id": self._next_id, "source": "wikipedia", "url": url,
                            "title": title, "area": area, "sector": sector,
                            "subsector": subsector, "chars": len(text), "hash": h, "file": rel,
                        })
                        self._next_id += 1
                        self.count += 1
                        added[0] += 1
                        got += 1
                    if log and got:
                        log(f"  {' > '.join(path)} (q='{query}'): +{got} "
                            f"[total {self.count}]")

        walk(taxonomy, [])
        if log:
            log(f"collection: +{added[0]} new sources | accumulated total {self.count}")
        return added[0]

    def stream_train(self, router, taxonomy, target=1000, chunk=8, per_query=15, lang="pt",
                     min_chars=400, checkpoint=None, save_every=50, log=print):
        """"Download -> train -> delete" flow: for each new source, train the `router`
        (partial_fit) and **discard the text immediately** (only the URL goes to the CSV
        and the knowledge to the model). Interleaves the areas (curriculum, like a child
        learning) and saves the training to `checkpoint` periodically. Resumable.

        `taxonomy`: nested dict; leaf = search query (or list of queries).
        Returns the number of new sources ingested in this run."""
        # 1) group the leaves by AREA (level 0) so we can interleave
        areas = {}

        def gather(node, prefix):
            for k, v in node.items():
                p = prefix + [k]
                if isinstance(v, dict):
                    gather(v, p)
                else:
                    for q in (v if isinstance(v, list) else [v]):
                        areas.setdefault(p[0], []).append((p, q))
        gather(taxonomy, [])

        # 2) candidates (titles) per area -- search once, skip the already seen
        candidates = {}
        for area, leaves in areas.items():
            local, lst = set(), []
            for path, q in leaves:
                try:
                    titles = corpus.wikipedia_search(q, lang, limit=per_query * 3)
                except Exception:  # noqa: BLE001
                    continue
                for t in titles:
                    if t in local or t in self.seen_titles:
                        continue
                    local.add(t)
                    lst.append((path, t))
            candidates[area] = lst

        # 3) round-robin across the areas (interleaves subjects), training and discarding
        idx = {a: 0 for a in areas}
        added = 0
        while added < target and any(idx[a] < len(candidates[a]) for a in areas):
            for area in areas:
                pulled = 0
                while pulled < chunk and idx[area] < len(candidates[area]) and added < target:
                    path, title = candidates[area][idx[area]]
                    idx[area] += 1
                    if title in self.seen_titles:
                        continue
                    self.seen_titles.add(title)
                    try:
                        text = corpus.wikipedia_extract(title, lang)
                    except Exception:  # noqa: BLE001
                        continue
                    if len(text) < min_chars:
                        continue
                    h = corpus._text_hash(text)
                    if h in self.seen_hashes:
                        continue
                    self.seen_hashes.add(h)

                    router.partial_fit(text, path)     # TRAIN on this text
                    chars = len(text)
                    del text                           # DISCARD the text immediately

                    url = f"https://{lang}.wikipedia.org/wiki/" + title.replace(" ", "_")
                    self._append_row({
                        "id": self._next_id, "source": "wikipedia", "url": url,
                        "title": title, "area": path[0],
                        "sector": path[1] if len(path) > 1 else "",
                        "subsector": path[2] if len(path) > 2 else "",
                        "chars": chars, "hash": h, "file": "discarded",
                    })
                    self._next_id += 1
                    self.count += 1
                    added += 1
                    pulled += 1
                    if checkpoint and added % save_every == 0:
                        router.save(checkpoint)
                if log and pulled:
                    log(f"  {area}: +{pulled} (total ingested {self.count})")
        if checkpoint:
            router.save(checkpoint)
        if log:
            log(f"flow: +{added} sources ingested and discarded | CSV {self.csv_path}")
        return added

    def _fetch_text(self, url):
        """Download the text of a URL, choosing the method: MediaWiki, PDF or HTML."""
        import urllib.parse
        low = url.lower()
        for dom in ("wikipedia.org", "wikibooks.org", "wikisource.org", "wiktionary.org"):
            if dom in low and "/wiki/" in url:
                host = url.split("://", 1)[-1].split("/wiki/", 1)[0]
                title = urllib.parse.unquote(url.split("/wiki/", 1)[1]).replace("_", " ")
                return corpus.mw_extract(host, title)
        return corpus.fetch_url_text(url)  # PDF (download/read/discard) or HTML

    def stream_train_specs(self, router, specs, target=1000, chunk=4, checkpoint=None,
                           save_every=50, min_chars=300, delay=0.0, log=print):
        """Multi-source flow: `specs` = list of (path, url) from ANY source
        (Wikipedia, Wikibooks, Wikisource, websites, book PDFs). Interleaves by
        area, trains on each text and **discards it on the spot**; writes the URL to
        the CSV. Dedup by URL and by content hash. Resumable. `delay` = pause (s)
        between downloads (polite for long collections)."""
        import time
        by_area = {}
        for path, url in specs:
            by_area.setdefault(path[0], []).append((path, url))
        areas = list(by_area)
        idx = {a: 0 for a in areas}
        added = 0
        while added < target and any(idx[a] < len(by_area[a]) for a in areas):
            for area in areas:
                pulled = 0
                while pulled < chunk and idx[area] < len(by_area[area]) and added < target:
                    path, url = by_area[area][idx[area]]
                    idx[area] += 1
                    if url in self.seen_urls:
                        continue
                    self.seen_urls.add(url)
                    if delay:
                        time.sleep(delay)
                    try:
                        text = self._fetch_text(url)
                    except Exception:  # noqa: BLE001
                        continue
                    if not text or len(text) < min_chars:
                        continue
                    h = corpus._text_hash(text)
                    if h in self.seen_hashes:
                        continue
                    self.seen_hashes.add(h)

                    router.partial_fit(text, path)     # TRAIN
                    chars = len(text)
                    del text                           # DISCARD immediately

                    title = url.rstrip("/").split("/")[-1].replace("_", " ")[:120]
                    src = url.split("://", 1)[-1].split("/", 1)[0]
                    self.seen_titles.add(title)
                    self._append_row({
                        "id": self._next_id, "source": src, "url": url, "title": title,
                        "area": path[0], "sector": path[1] if len(path) > 1 else "",
                        "subsector": path[2] if len(path) > 2 else "",
                        "chars": chars, "hash": h, "file": "discarded",
                    })
                    self._next_id += 1
                    self.count += 1
                    added += 1
                    pulled += 1
                    if checkpoint and added % save_every == 0:
                        router.save(checkpoint)
                if log and pulled:
                    log(f"  {area}: +{pulled} (total {self.count})")
        if checkpoint:
            router.save(checkpoint)
        if log:
            log(f"multi-source flow: +{added} sources ingested and discarded")
        return added

    def dataset(self):
        """Read the collected corpus as {'text','path','title','url'} records for training."""
        recs = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fpath = os.path.join(self.out_dir, row["file"])
                if not os.path.exists(fpath):
                    continue
                with open(fpath, encoding="utf-8") as tf:
                    text = tf.read()
                path = [p for p in (row["area"], row["sector"], row["subsector"]) if p]
                recs.append({"text": text, "path": path, "title": row["title"],
                             "url": row["url"]})
        return recs
