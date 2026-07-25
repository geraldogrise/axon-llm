"""Tests of the scikit-learn-style ML models (pyaxon.ml) and preprocessing."""

import pytest

np = pytest.importorskip("numpy")
pyaxon = pytest.importorskip("pyaxon")
ax = pyaxon


def _blobs(seed=0, n=60):
    """3 well-separated 2D Gaussian blobs (for classification/clustering)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [6.0, 6.0], [0.0, 6.0]])
    X = np.vstack([c + rng.normal(0, 0.5, (n, 2)) for c in centers])
    y = np.concatenate([np.full(n, i) for i in range(3)])
    return X, y


# ----- preprocessing -----
def test_standard_scaler():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    Z = ax.pre.StandardScaler().fit_transform(X)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Z.std(axis=0), 1, atol=1e-9)


def test_normalize_and_onehot():
    X = np.array([[3.0, 4.0]])
    assert np.allclose(np.linalg.norm(ax.pre.normalize(X, "l2"), axis=1), 1.0)
    assert ax.pre.one_hot([0, 2], 3).tolist() == [[1, 0, 0], [0, 0, 1]]


# ----- regression -----
def test_linear_regression():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (100, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.0  # exact linear relation
    model = ax.ml.LinearRegression().fit(X, y)
    assert model.score(X, y) > 0.999
    assert np.allclose(model.coef_, [3.0, -2.0], atol=1e-6)


def test_logistic_regression():
    X, y = _blobs()
    model = ax.ml.LogisticRegression(epochs=300, lr=0.1).fit(X, y)
    assert model.score(X, y) > 0.95


# ----- Naive Bayes -----
def test_gaussian_nb():
    X, y = _blobs()
    assert ax.ml.GaussianNB().fit(X, y).score(X, y) > 0.95


# ----- KMeans -----
def test_kmeans():
    X, y = _blobs()
    km = ax.ml.KMeans(n_clusters=3, seed=0).fit(X)
    # each cluster should be dominated by a single true class
    purities = 0
    for k in range(3):
        true_labels = y[km.labels_ == k]
        if len(true_labels):
            _, counts = np.unique(true_labels, return_counts=True)
            purities += counts.max() / counts.sum()
    assert purities / 3 > 0.9


# ----- KNN -----
def test_knn():
    X, y = _blobs()
    assert ax.ml.KNeighborsClassifier(k=5).fit(X, y).score(X, y) > 0.95


# ----- Decision tree -----
def test_decision_tree():
    X, y = _blobs()
    tree = ax.ml.DecisionTreeClassifier(max_depth=4).fit(X, y)
    assert tree.score(X, y) > 0.95


def test_tfidf_vectorizer():
    vec = ax.pre.TfidfVectorizer(max_features=50, ngram=1, min_df=1)
    X = vec.fit_transform(["gato preto corre", "gato branco dorme", "cão late alto"])
    assert X.shape[0] == 3
    assert abs(np.linalg.norm(X[0]) - 1.0) < 1e-9  # L2-normalized rows


def test_linear_router_hierarchical():
    data = [
        ("matriz vetor determinante autovalor espaco linear", ["mat", "algebra"]),
        ("primo fatoracao divisibilidade congruencia modulo", ["mat", "numeros"]),
        ("forca massa aceleracao newton movimento energia", ["fis", "mecanica"]),
        ("quantum onda particula probabilidade spin atomo", ["fis", "quantica"]),
    ] * 6
    r = ax.router.LinearRouter(epochs=150)
    for text, path in data:
        r.partial_fit(text, path)   # keeps only the bag; text discarded
    r.fit()
    assert r.route("determinante de uma matriz e autovalores") == ["mat", "algebra"]
    assert r.route("onda quantica de uma particula") == ["fis", "quantica"]
    assert r.last_activated_ == 2   # compartmentalized


# ----- New: PCA, SVM, Random Forest, metrics, cross-validation, MoE -----
def test_pca():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (100, 1))
    X = np.hstack([base, base * 2 + rng.normal(0, 0.01, (100, 1)), rng.normal(0, 0.01, (100, 1))])
    pca = ax.ml.PCA(n_components=1).fit(X)
    assert pca.transform(X).shape == (100, 1)
    assert pca.explained_variance_ratio_[0] > 0.95   # one direction dominates


def test_linear_svc():
    X, y = _blobs()
    assert ax.ml.LinearSVC(epochs=200, lr=0.05).fit(X, y).score(X, y) > 0.9


def test_random_forest():
    X, y = _blobs()
    assert ax.ml.RandomForestClassifier(n_estimators=15, seed=0).fit(X, y).score(X, y) > 0.95


def test_metrics():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 2, 2, 2]
    assert abs(ax.metrics.accuracy_score(y_true, y_pred) - 5 / 6) < 1e-9  # only index 3 wrong
    cm, labels = ax.metrics.confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3) and cm.sum() == 6
    f1 = ax.metrics.f1_score(y_true, y_pred, average="macro")
    assert 0.0 <= f1 <= 1.0


def test_cross_val_score():
    X, y = _blobs()
    scores = ax.model_selection.cross_val_score(lambda: ax.ml.GaussianNB(), X, y, cv=5)
    assert len(scores) == 5 and scores.mean() > 0.9


def test_moe_layer():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (12, 16))
    moe = ax.moe.MoELayer(d_model=16, d_hidden=32, num_experts=8, top_k=2, seed=1)
    out = moe(X)
    assert out.shape == (12, 16)                       # shape preserved
    assert moe.last_load_ is not None and abs(moe.last_load_.sum() - 1.0) < 1e-6
    assert moe.load_balance_loss(X) >= 1.0             # 1.0 = perfectly balanced


def test_soft_moe_trainable():
    # soft-gated MoE is differentiable end-to-end: one backward gives the gate a gradient
    import math
    rng = np.random.default_rng(0)
    X = ax.from_numpy(rng.normal(0, 1, (16, 8)).astype(np.float32))
    Y = ax.from_numpy(rng.normal(0, 1, (16, 8)).astype(np.float32))
    moe = ax.moe.SoftMoE(d_model=8, d_hidden=16, num_experts=4, seed=1)
    assert len(moe.parameters()) == 2 + 4 * 2 * 2       # gate (w+b) + 4 experts x (2 linears x [w+b])
    out = moe(X)
    assert out.shape() == [16, 8]                        # shape preserved
    loss = ax.mse_loss(out, Y)
    assert math.isfinite(loss.item())
    loss.backward()
    assert moe.gate.parameters()[0].grad() is not None   # gate is differentiable (trainable)
    # a low-lr step keeps it finite (it does train; sustained training can need grad clipping)
    opt = ax.optim.Adam(moe.parameters(), lr=0.001)
    opt.step()
    assert math.isfinite(ax.mse_loss(moe(X), Y).item())


def test_losses_bce_huber():
    # BCE: perfect prediction -> ~0 loss; wrong -> large
    assert ax.losses.bce_loss([1, 0, 1], [0.99, 0.01, 0.99]) < 0.05
    assert ax.losses.bce_loss([1, 0], [0.01, 0.99]) > 2.0
    # Huber: small error quadratic, large error linear (< MSE)
    small = ax.losses.huber_loss([0.0], [0.5], delta=1.0)
    assert abs(small - 0.125) < 1e-9                    # 0.5*0.5^2
    big = ax.losses.huber_loss([0.0], [10.0], delta=1.0)
    assert big < 0.5 * 10 ** 2                          # linear tail, below MSE


def test_conv2d():
    x = ax.from_numpy(np.ones((1, 1, 5, 5), dtype=np.float32))
    conv = ax.layers.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, seed=0)
    out = conv(x)
    assert out.shape() == [1, 3, 5, 5]                 # padding keeps size
    conv2 = ax.layers.Conv2d(1, 2, kernel_size=3, stride=2, padding=0, seed=0)
    assert conv2(x).shape() == [1, 2, 2, 2]            # stride halves
    # trainable via autograd: loss.backward() populates the weight gradient
    loss = ax.sum(out)
    loss.backward()
    assert conv.weight.grad().shape() == [3, 1, 3, 3]


# ----- Generation layer + unified system (roadmap #1: close the loop, offline) -----
def test_generate_extractive():
    # model=None -> extractive answer (top passage, verbatim). No LLM, no network.
    import pyaxon as ax
    passages = [("O determinante mede a area/volume de uma transformacao linear.",
                 ["mat", "algebra"], 0.9),
                ("Autovalores sao escalares lambda tais que Av = lambda v.",
                 ["mat", "algebra"], 0.7)]
    text, mode = ax.generate.grounded_answer("o que e determinante?", passages, model=None)
    assert mode == "extractive" and "determinante" in text.lower()
    assert ax.generate.extractive_answer([]) is None       # empty retrieval -> None


def test_axon_system_domain_gate():
    import pyaxon as ax
    # two tiny domains; router=None -> SparseKB retrieves globally within each expert
    # docs must clear split_chunks' 80-char minimum, so each is a full paragraph
    bio = ax.vindex.SparseKB(ngram=1)
    bio.add_document("A fotossintese ocorre nos cloroplastos das plantas, onde a clorofila "
                     "capta a luz do sol e a converte em energia quimica, produzindo glicose "
                     "e liberando oxigenio para a atmosfera durante o processo.",
                     ["biologia", "celula"])
    bio.add_document("O DNA carrega toda a informacao genetica dentro dos cromossomos das "
                     "celulas, e essa hereditariedade determina as caracteristicas que os "
                     "seres vivos herdam de seus pais ao longo das geracoes.",
                     ["biologia", "genetica"])
    bio.build()
    code = ax.vindex.SparseKB(ngram=1)
    code.add_document("Em Python voce usa a palavra def para declarar uma funcao e a palavra "
                      "return para devolver um valor ao final; os parametros ficam entre "
                      "parenteses e o corpo da funcao vem indentado logo abaixo da definicao.",
                      ["python", "sintaxe"])
    code.add_document("Uma lista em Python e criada com colchetes e aceita o metodo append "
                      "para adicionar elementos ao final; listas sao mutaveis e podem guardar "
                      "valores de tipos diferentes na mesma estrutura de dados sequencial.",
                      ["python", "estruturas"])
    code.build()
    system = ax.system.AxonSystem([ax.system.Expert("biologia", None, bio),
                                   ax.system.Expert("python", None, code)]).fit_domain_gate()
    # domain gate (WordNB) alone separates the two domains by vocabulary
    assert system.domain_gate.predict("fotossintese clorofila plantas") == "biologia"
    assert system.domain_gate.predict("funcao def return lista append") == "python"
    # model=None forces the real extractive path (no LLM, no network) -- tests the gate
    r1 = system.answer("como funciona a fotossintese nas plantas?", multi=0, model=None)
    assert r1["expert"] == "biologia" and r1["mode"] == "extractive"
    assert "fotossintese" in r1["answer"].lower()
    r2 = system.answer("como declarar uma funcao em Python?", multi=0, model=None)
    assert r2["expert"] == "python" and r2["mode"] == "extractive"
    # weighted average (retrieval + NB) -> same winner; alpha extremes agree here too
    assert system.route("append em lista", alpha=0.0)[0].name == "python"
    assert system.route("append em lista", alpha=1.0)[0].name == "python"
    # nothing relevant -> abstain
    r3 = system.answer("qual a cotacao do dolar hoje?", multi=0, min_score=0.5, model=None)
    assert r3["mode"] == "abstain"
