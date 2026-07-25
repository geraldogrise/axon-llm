"""Offline axon-lang tests: langid, hierarchical router and file reading.

They do not use the network (the Wikipedia download is demonstrated in
examples/axon_lang.py).
"""

import pytest

pyaxon = pytest.importorskip("pyaxon")
ax = pyaxon


# ----- Language identification -----
def test_langid_detects_portuguese():
    lid = ax.langid.LanguageIdentifier().fit()
    assert lid.predict("a matemática estuda os números e as formas") == "pt"
    assert lid.is_portuguese("quero resolver um problema de álgebra linear")
    assert lid.predict("the cat sat on the mat and slept") == "en"
    assert not lid.is_portuguese("the weather is nice today outside")


# ----- Hierarchical router -----
def _synthetic():
    # vocabulary clearly separable by subsector
    return [
        ("matriz vetor determinante autovalor espaço linear", ["mat", "algebra", "linear"]),
        ("grupo anel corpo homomorfismo álgebra abstrata", ["mat", "algebra", "abstrata"]),
        ("primo fatoração divisibilidade congruência módulo", ["mat", "numeros", "primos"]),
        ("força massa aceleração newton movimento", ["fis", "mecanica", "classica"]),
        ("quantum onda partícula probabilidade spin", ["fis", "mecanica", "quantica"]),
    ] * 4  # repeat to give enough examples per class


def test_router_classifies_and_compartmentalizes():
    r = ax.router.HierarchicalRouter().fit(_synthetic())
    assert r.route("determinante de uma matriz e autovalores") == ["mat", "algebra", "linear"]
    assert r.route("número primo e fatoração") == ["mat", "numeros", "primos"]
    assert r.route("onda quântica e partícula") == ["fis", "mecanica", "quantica"]
    # compartmentalization: 3 levels -> at most 3 classifiers activated
    r.route("força e aceleração de newton")
    assert r.last_activated_ <= 3


def test_router_save_load(tmp_path):
    r = ax.router.HierarchicalRouter().fit(_synthetic())
    p = str(tmp_path / "router.json")
    r.save(p)
    r2 = ax.router.HierarchicalRouter().load(p)
    assert r2.route("matriz determinante autovalor") == ["mat", "algebra", "linear"]


# ----- File reading (data pipeline, offline) -----
def test_corpus_reads_files(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text("olá mundo em português", encoding="utf-8")
    assert "português" in ax.corpus.read_file(str(txt))

    md = tmp_path / "b.md"
    md.write_text("# Título\nconteúdo markdown", encoding="utf-8")
    assert "markdown" in ax.corpus.read_file(str(md))

    csv = tmp_path / "c.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    assert "1" in ax.corpus.read_file(str(csv))


def test_router_incremental_partial_fit():
    # online training: ingests one text at a time (the text can be discarded afterwards)
    r = ax.router.HierarchicalRouter()
    for text, path in _synthetic():
        r.partial_fit(text, path)
    assert r.route("matriz determinante autovalor espaço") == ["mat", "algebra", "linear"]
    assert r.route("primo fatoração módulo") == ["mat", "numeros", "primos"]


def test_collector_csv_dedup(tmp_path):
    # the Collector reloads the CSV index and knows what was already downloaded (no re-download)
    out = str(tmp_path / "corpus")
    col = ax.collect.Collector(out)
    col._append_row({"id": 1, "source": "wikipedia", "url": "http://x/A", "title": "A",
                     "area": "mat", "sector": "", "subsector": "", "chars": 100,
                     "hash": "h1", "file": "discarded"})
    col2 = ax.collect.Collector(out)  # new instance reads the existing CSV
    assert col2.count == 1
    assert "A" in col2.seen_titles and "h1" in col2.seen_hashes


def test_corpus_dedup_hash():
    h1 = ax.corpus._text_hash("Olá  Mundo")
    h2 = ax.corpus._text_hash("olá mundo")
    assert h1 == h2  # normalizes spaces/case -> same hash (avoids duplication)


# ----- Compression (int8 + gzip) -----
def test_int8_quantize_roundtrip():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    mat = rng.normal(0, 1, (20, 50)).astype("float32")
    q = ax.compress.quantize_int8(mat)
    back = ax.compress.dequantize_int8(q)
    assert back.shape == mat.shape
    # int8 with per-row scale: reconstruction is close (max error < 1 quantization step)
    assert np.abs(back - mat).max() < np.abs(mat).max() / 127 + 1e-6


def test_knowledge_base_compressed_save_load(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    docs = [
        ("a matemática estuda os números, a álgebra e a geometria das formas, e "
         "também as funções, as equações e a teoria dos conjuntos aplicada", ["mat"]),
        ("a física estuda a força, a energia, o movimento dos corpos e as ondas, "
         "além da termodinâmica, do eletromagnetismo e da relatividade", ["fis"]),
        ("o determinante de uma matriz e os seus autovalores aparecem na álgebra "
         "linear, junto de vetores, espaços vetoriais e transformações lineares", ["mat"]),
        ("a mecânica quântica descreve partículas, ondas e o spin, com estados de "
         "probabilidade, superposição e o princípio da incerteza de Heisenberg", ["fis"]),
    ] * 3
    kb = ax.rag.KnowledgeBase(dim=8, max_features=200, ngram=1)
    for text, path in docs:
        kb.add_document(text, path)
    kb.build()
    ref = kb.retrieve("autovalores de uma matriz", top_k=1)[0][0]

    p = str(tmp_path / "kb.json.gz")
    kb.save(p)                       # .gz -> compressed (gzip + int8) by default
    assert p and __import__("os").path.exists(p)

    kb2 = ax.rag.KnowledgeBase().load(p)     # no SVD recompute
    got = kb2.retrieve("autovalores de uma matriz", top_k=1)[0][0]
    assert got == ref                # quantized index still retrieves the same chunk

    # compressed file is smaller than the plain JSON of the same content
    plain = str(tmp_path / "kb.json")
    kb.save(plain, compressed=False)
    import os
    assert os.path.getsize(p) < os.path.getsize(plain)


# ----- Robustness features (modular router, sparse KB, confidence, versioning) -----
def _prog():
    return [
        ("matriz vetor determinante autovalor espaco algebra linear", ["mat", "algebra", "linear"]),
        ("primo fatoracao divisibilidade congruencia modulo numeros", ["mat", "numeros", "primos"]),
        ("forca massa aceleracao newton energia movimento mecanica", ["fis", "mecanica", "classica"]),
        ("onda particula quantica probabilidade spin atomo bohr", ["fis", "quantica", "atomo"]),
    ] * 5


def test_modular_router_incremental():
    r = ax.modular.ModularRouter(epochs=80)
    for t, p in _prog():
        r.add(t, p)
    r.fit()
    assert r.route("determinante de uma matriz") == ["mat", "algebra", "linear"]
    assert r.route("onda quantica de uma particula") == ["fis", "quantica", "atomo"]
    # adding a NEW area marks only it dirty -> mat/fis are not retrained
    for t, p in [("celula dna gene hereditariedade genetica", ["bio", "genetica", "dna"])] * 5:
        r.add(t, p)
    assert r._dirty == {"bio"}
    r.fit()
    assert r.route("o que e o dna e a genetica") == ["bio", "genetica", "dna"]
    assert r.route("determinante de uma matriz") == ["mat", "algebra", "linear"]  # still ok


def test_modular_confidence_gate():
    r = ax.modular.ModularRouter(epochs=80)
    for t, p in _prog():
        r.add(t, p)
    r.fit()
    assert 0.0 <= r.confidence("determinante de uma matriz") <= 1.0
    # an absurdly high threshold makes it abstain (empty path) instead of guessing
    assert r.route("determinante de uma matriz", threshold=1.01) == []


def test_modular_save_load_and_manifest(tmp_path):
    r = ax.modular.ModularRouter(epochs=80)
    for t, p in _prog():
        r.add(t, p)
    r.fit()
    prefix = str(tmp_path / "router")
    r.save(prefix)
    man = ax.manifest.read_manifest(prefix)
    assert man is not None and man["model"] == "ModularRouter"
    r2 = ax.modular.ModularRouter().load(prefix)
    assert r2.route("determinante de uma matriz") == ["mat", "algebra", "linear"]


def test_sparse_kb_retrieve(tmp_path):
    pytest.importorskip("scipy")
    kb = ax.vindex.SparseKB(ngram=1)
    docs = [
        ("a matematica estuda numeros, algebra e a geometria das formas, funcoes, "
         "equacoes, matrizes, determinantes e vetores em algebra linear", ["mat"]),
        ("a fisica estuda forca, energia, movimento, ondas, campo eletrico e "
         "magnetico, termodinamica e a mecanica quantica das particulas", ["fis"]),
    ] * 4
    for text, path in docs:
        kb.add_document(text, path)
    kb.build()
    got = kb.retrieve("determinante e matriz em algebra", path_prefix=["mat"], top_k=1)
    assert got and got[0][1] == ["mat"]
    p = str(tmp_path / "kb.sparse.json.gz")
    kb.save(p)
    kb2 = ax.vindex.SparseKB().load(p)
    assert kb2.retrieve("determinante e matriz", path_prefix=["mat"], top_k=1)[0][1] == ["mat"]


def test_logreg_minibatch():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (40, 4)), rng.normal(5, 1, (40, 4))])
    y = np.array([0] * 40 + [1] * 40)
    m = ax.ml.LogisticRegression(epochs=60, lr=0.1, batch_size=16).fit(X, y)
    assert m.score(X, y) > 0.95     # mini-batch training converges
