"""axon-lang data pipeline: ingestion of files, web and Wikipedia.

Reads txt/md/csv/docx/xlsx/pdf and web pages; downloads Wikipedia articles to
build labeled datasets (area/sector/subsector). Downloaded PDFs are read and then
**discarded to the trash**. Heavy dependencies are imported on demand.
"""

import json
import os
import shutil
import tempfile

_UA = {"User-Agent": "pyaxon-axonlang/0.1 (educational; contact: example@example.com)"}


# ---------------------------------------------------------------------------
# Reading files by extension
# ---------------------------------------------------------------------------
def _read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        import chardet
        enc = chardet.detect(raw).get("encoding") or "utf-8"
    except Exception:
        enc = "utf-8"
    return raw.decode(enc, errors="replace")


def read_pdf(path):
    """Extract text from a PDF (pypdf; fallback PyMuPDF/fitz)."""
    try:
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(path).pages)
    except Exception:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)


def read_docx(path):
    import docx
    return "\n".join(p.text for p in docx.Document(path).paragraphs)


def read_xlsx(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            out.append(" ".join("" if c is None else str(c) for c in row))
    return "\n".join(out)


def read_csv(path):
    import csv
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return "\n".join(" ".join(row) for row in csv.reader(f))


def read_file(path):
    """Read a file and return the text, choosing the reader by extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md", ".rst", ".log", ".py", ".json"):
        return _read_text(path)
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    if ext in (".xlsx", ".xlsm"):
        return read_xlsx(path)
    if ext == ".csv":
        return read_csv(path)
    return _read_text(path)  # generic attempt


def to_trash(path):
    """Move a file to the trash (send2trash) or to a local .trash folder."""
    try:
        from send2trash import send2trash
        send2trash(path)
        return
    except Exception:
        trash = os.path.join(tempfile.gettempdir(), "pyaxon_trash")
        os.makedirs(trash, exist_ok=True)
        try:
            shutil.move(path, os.path.join(trash, os.path.basename(path)))
        except Exception:
            os.remove(path)


# ---------------------------------------------------------------------------
# Web
# ---------------------------------------------------------------------------
def fetch_url_text(url, timeout=20):
    """Download a URL and return the text. PDF: save to temp, read and trash it."""
    import requests
    r = requests.get(url, headers=_UA, timeout=timeout)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        fd, tmp = tempfile.mkstemp(suffix=".pdf")
        with os.fdopen(fd, "wb") as f:
            f.write(r.content)
        try:
            return read_pdf(tmp)
        finally:
            to_trash(tmp)  # discard the PDF after reading
    # HTML -> text
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return " ".join(soup.get_text(" ").split())


# ---------------------------------------------------------------------------
# Wikipedia (main source to "understand" a domain)
# ---------------------------------------------------------------------------
def _mw_api(host, params, timeout=20):
    """Generic call to the MediaWiki API (Wikipedia, Wikibooks, Wikisource...)."""
    import requests
    params = {"format": "json", "action": "query", **params}
    r = requests.get(f"https://{host}/w/api.php", params=params, headers=_UA, timeout=timeout)
    r.raise_for_status()
    return r.json()


def mw_extract(host, title):
    """Plain text of any MediaWiki page."""
    data = _mw_api(host, {"prop": "extracts", "explaintext": 1, "redirects": 1, "titles": title})
    for _, page in data.get("query", {}).get("pages", {}).items():
        if "extract" in page:
            return page["extract"]
    return ""


def mw_search(host, query, limit=20, offset=0):
    """Titles matching a search on a MediaWiki wiki. `offset` paginates deeper
    (to collect continuously without repeating the same articles)."""
    params = {"list": "search", "srsearch": query, "srlimit": limit}
    if offset:
        params["sroffset"] = offset
    data = _mw_api(host, params)
    return [item["title"] for item in data.get("query", {}).get("search", [])]


def mw_url(host, title):
    return f"https://{host}/wiki/" + title.replace(" ", "_")


# Per-project shortcuts (all MediaWiki) -- knowledge sources in Portuguese.
def wikipedia_extract(title, lang="pt"):
    return mw_extract(f"{lang}.wikipedia.org", title)


def wikipedia_search(query, lang="pt", limit=20):
    return mw_search(f"{lang}.wikipedia.org", query, limit)


def wikibooks_extract(title, lang="pt"):   # Wikibooks: educational material
    return mw_extract(f"{lang}.wikibooks.org", title)


def wikibooks_search(query, lang="pt", limit=20):
    return mw_search(f"{lang}.wikibooks.org", query, limit)


def wikisource_extract(title, lang="pt"):  # Wikisource: complete literary works
    return mw_extract(f"{lang}.wikisource.org", title)


def wikisource_search(query, lang="pt", limit=20):
    return mw_search(f"{lang}.wikisource.org", query, limit)


# Named providers (host, functions) for the multi-source collector.
MW_SOURCES = {
    "wikipedia": "{lang}.wikipedia.org",
    "wikibooks": "{lang}.wikibooks.org",
    "wikisource": "{lang}.wikisource.org",
    "wiktionary": "{lang}.wiktionary.org",
}


def wikipedia_category(category, lang="pt", limit=50):
    """Page titles of a category (e.g. 'Teoria dos números')."""
    cat = category if category.lower().startswith(("categoria:", "category:")) else \
        ("Categoria:" + category if lang == "pt" else "Category:" + category)
    data = _mw_api(f"{lang}.wikipedia.org", {"list": "categorymembers", "cmtitle": cat,
                            "cmlimit": limit, "cmtype": "page"})
    return [m["title"] for m in data.get("query", {}).get("categorymembers", [])]


def _text_hash(text):
    import hashlib
    return hashlib.md5(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def wikipedia_subcategories(category, lang="pt", limit=50):
    """Subcategories of a category (the AI uses this to *discover* subsectors)."""
    cat = category if category.lower().startswith(("categoria:", "category:")) else \
        ("Categoria:" + category if lang == "pt" else "Category:" + category)
    data = _mw_api(f"{lang}.wikipedia.org", {"list": "categorymembers", "cmtitle": cat,
                            "cmlimit": limit, "cmtype": "subcat"})
    out = []
    for m in data.get("query", {}).get("categorymembers", []):
        # strip the "Categoria:" prefix to make a clean label
        out.append(m["title"].split(":", 1)[-1])
    return out


def discover_taxonomy(root_category, depth=2, breadth=6, lang="pt", log=print):
    """Build a taxonomy automatically by descending Wikipedia's subcategories
    from `root_category` (the AI "creates the subsectors" by searching).

    Returns a nested dict; each leaf maps to the search query (the category name).
    `depth` = nesting levels; `breadth` = subcategories per level.
    """
    def build(category, level):
        if level >= depth:
            return category  # leaf: use its own name as the search query
        subs = wikipedia_subcategories(category, lang, limit=breadth * 3)[:breadth]
        if not subs:
            return category  # no subcategories -> leaf
        if log:
            log(f"{'  ' * level}discovered in '{category}': {subs}")
        node = {}
        for sub in subs:
            node[sub] = build(sub, level + 1)
        return node

    return {root_category: build(root_category, 0)}


def collect_articles(query, n=20, lang="pt", min_chars=200, seen_titles=None, seen_hashes=None):
    """Download up to `n` articles for `query`; skip titles/contents already seen (dedup).

    `seen_titles` and `seen_hashes` are sets updated in-place to avoid downloading
    the same article twice (across leaves and across runs)."""
    seen_titles = seen_titles if seen_titles is not None else set()
    seen_hashes = seen_hashes if seen_hashes is not None else set()
    titles = wikipedia_search(query, lang, limit=n * 3)
    out = []
    for t in titles:
        if t in seen_titles:
            continue
        seen_titles.add(t)
        txt = wikipedia_extract(t, lang)
        if len(txt) < min_chars:
            continue
        h = _text_hash(txt)
        if h in seen_hashes:
            continue  # same content already downloaded via another path
        seen_hashes.add(h)
        out.append((t, txt))
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Labeled dataset (area/sector/subsector)
# ---------------------------------------------------------------------------
def write_jsonl(records, path):
    """Save records {'text':..., 'path':[area,sector,subsector]} as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_wikipedia_dataset(taxonomy, per_leaf=20, lang="pt", manifest_path=None, log=print):
    """Download Wikipedia articles for each leaf of a taxonomy and build the dataset.

    `taxonomy` is a nested dict; each leaf maps to a search query.
    E.g. {"matematica": {"teoria dos numeros": "teoria dos números",
                         "algebra": "álgebra"}, "fisica": {"mecanica": "mecânica"}}

    `manifest_path`: JSON file with the already-downloaded titles -- avoids
    duplication across runs (records the pulled paths). Returns a list of
    {'text','path','title'}.
    """
    seen_titles, seen_hashes = set(), set()
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            man = json.load(f)
        seen_titles = set(man.get("titles", []))
        seen_hashes = set(man.get("hashes", []))

    records = []

    def walk(node, prefix):
        for key, val in node.items():
            path = prefix + [key]
            if isinstance(val, dict):
                walk(val, path)
            else:  # leaf: val is the search query
                if log:
                    log(f"downloading ~{per_leaf} articles for {' > '.join(path)} (q='{val}')")
                for title, txt in collect_articles(val, n=per_leaf, lang=lang,
                                                    seen_titles=seen_titles,
                                                    seen_hashes=seen_hashes):
                    records.append({"text": txt, "path": path, "title": title})

    walk(taxonomy, [])

    if manifest_path:  # record the pulled paths (persistent dedup)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"titles": sorted(seen_titles), "hashes": sorted(seen_hashes)}, f)
    return records
