"""Tests for the pyaxon Python API (pytest).

Requires the _axon.pyd extension to be built (scripts\\build_python.ps1) and the
package to be on the PYTHONPATH (python/ folder) or installed.
"""

import math

import pytest

pyaxon = pytest.importorskip("pyaxon")
ax = pyaxon


def test_tensor_creation_and_read():
    t = ax.tensor([[1, 2], [3, 4]])
    assert t.shape() == [2, 2]
    assert t.numel() == 4
    assert t.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_matmul():
    a = ax.tensor([[1, 2], [3, 4]])
    b = ax.tensor([[5, 6], [7, 8]])
    c = a @ b
    assert c.tolist() == [[19.0, 22.0], [43.0, 50.0]]


def test_autograd_simple():
    # f = sum(x * x) => df/dx = 2x
    x = ax.tensor([1, 2, 3], requires_grad=True)
    loss = ax.sum(x * x)
    loss.backward()
    assert x.grad().tolist() == [2.0, 4.0, 6.0]


def test_broadcast_operators():
    a = ax.tensor([[1, 2], [3, 4]])
    b = ax.tensor([10, 20])  # broadcast over the row
    assert (a + b).tolist() == [[11.0, 22.0], [13.0, 24.0]]


def test_softmax_sums_one():
    y = ax.softmax(ax.tensor([[1, 2, 3], [1, 1, 1]]))
    rows = y.tolist()
    assert abs(sum(rows[0]) - 1.0) < 1e-6
    assert all(abs(v - 1 / 3) < 1e-6 for v in rows[1])


def test_bce_and_huber_autograd():
    # BCE with logits: near-perfect prediction -> tiny loss; and gradient flows
    logits = ax.tensor([[4.0], [-4.0]], requires_grad=True)
    targets = ax.tensor([[1.0], [0.0]])
    loss = ax.bce_loss(logits, targets)
    assert loss.item() < 0.05
    loss.backward()
    assert logits.grad().shape() == [2, 1]        # autograd reached the logits
    # Huber: quadratic within delta -> 0.5*0.5^2 = 0.125
    h = ax.huber_loss(ax.tensor([[0.5]]), ax.tensor([[0.0]]), 1.0)
    assert abs(h.item() - 0.125) < 1e-5


def test_cross_entropy_and_gelu():
    logits = ax.tensor([[10, 0, 0], [0, 10, 0]])
    targets = ax.tensor([0, 1])
    assert ax.cross_entropy(logits, targets).item() < 0.01
    # gelu(0) == 0
    assert abs(ax.gelu(ax.tensor([0.0])).tolist()[0]) < 1e-6


def test_layernorm_and_attention_modules():
    x = ax.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    ln = ax.nn.LayerNorm(4)
    y = ln(x)
    # each normalized row has mean ~0
    for row in y.tolist():
        assert abs(sum(row) / len(row)) < 1e-4

    attn = ax.nn.SelfAttention(4, causal=True, seed=1)
    out = attn(x)
    assert out.shape() == [2, 4]


def test_embedding():
    emb = ax.nn.Embedding(5, 3, seed=1)
    out = emb(ax.tensor([0.0, 2.0, 0.0]))
    assert out.shape() == [3, 3]


def test_tokenizer_bpe():
    tok = ax.BPETokenizer()
    tok.train("o rato roeu a roupa do rei de roma", vocab_size=50)
    assert tok.vocab_size > 10
    ids = tok.encode("o rato")
    assert tok.decode(ids) == "o rato"


def test_data_windows():
    ids = [0, 1, 2, 3, 4]
    w = ax.data.make_windows(ids, 2)
    # windows: ([0,1]->[1,2]), ([1,2]->[2,3]), ([2,3]->[3,4])
    assert len(w) == 3
    assert w[0] == ([0, 1], [1, 2])
    assert w[2] == ([2, 3], [3, 4])
    # batches cover all windows
    batches = list(ax.data.iter_batches(w, batch_size=2, shuffle=False))
    total = sum(len(b) for b in batches)
    assert total == 3


def test_rnn_lstm_dropout():
    x = ax.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert ax.nn.RNN(3, 8)(x).shape() == [3, 8]
    assert ax.nn.LSTM(3, 6)(x).shape() == [3, 6]
    # Dropout at inference (no_grad) is the identity.
    with ax.no_grad():
        assert ax.nn.Dropout(0.5)(x).tolist() == x.tolist()


def test_adamw_weight_decay():
    # weight_decay shrinks the weights even with grad ~0.
    w = ax.tensor([2.0, -3.0], requires_grad=True)
    loss = ax.sum(w) * ax.zeros([1])
    loss.backward()
    opt = ax.optim.Adam([w], lr=0.1, weight_decay=0.2)
    opt.step()
    assert abs(w.tolist()[0]) < 2.0


def test_top_p_sampling():
    # top_p deterministically picks the dominant token when it already covers >= p.
    logits = [10.0, 0.0, 0.0, 0.0]
    picks = {ax.lm._sample(logits, 0.5, 0, 0.9) for _ in range(20)}
    assert picks == {0}


def test_data_loaders(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text("abc def", encoding="utf-8")
    assert ax.data.load_text(str(p)) == "abc def"
    assert "".join(ax.data.iter_text_chunks(str(p), chunk_chars=2)) == "abc def"


def test_permute_slice_rand_reductions():
    t = ax.tensor([[1, 2, 3], [4, 5, 6]])
    assert t.permute([1, 0]).shape() == [3, 2]
    assert t.slice(0, 1, 1).tolist() == [[4.0, 5.0, 6.0]]
    assert ax.sum_axis(t, 0).tolist() == [5.0, 7.0, 9.0]
    assert ax.max_axis(t, 1).tolist() == [3.0, 6.0]
    r = ax.rand([50], seed=1, lo=0.0, hi=1.0)
    assert all(0.0 <= v < 1.0 for v in r.tolist())


def test_log_softmax_and_xavier_init():
    y = ax.log_softmax(ax.tensor([[1.0, 2.0, 3.0]])).tolist()[0]
    assert abs(sum(math.exp(v) for v in y) - 1.0) < 1e-5
    # Init.Xavier produces weights different from He (same seed).
    he = ax.nn.Linear(8, 4, seed=1, init=ax.nn.Init.He)
    xa = ax.nn.Linear(8, 4, seed=1, init=ax.nn.Init.Xavier)
    assert he.parameters()[0].tolist() != xa.parameters()[0].tolist()


def test_lr_scheduler():
    import pyaxon as _ax
    # cosine with warmup: rises during warmup, decays afterwards.
    assert _ax.lm.cosine_lr(0, 100, 0.1, warmup_steps=10) < 0.1
    assert abs(_ax.lm.cosine_lr(10, 100, 0.1, warmup_steps=10) - 0.1) < 1e-6
    assert _ax.lm.cosine_lr(99, 100, 0.1, warmup_steps=10) < 0.01


def test_optimizer_state_in_bundle(tmp_path):
    # Train, save the bundle WITH optimizer state, reload and check the state.
    m, tok, cfg = ax.lm.train_lm("ab ab cd cd ab cd", vocab_size=25, dim=16, num_heads=2,
                                 num_layers=1, context=3, epochs=5, seed=1, log=None)
    opt = ax.optim.Adam(m.parameters(), lr=0.01)
    # one step to generate non-trivial state
    logits = m.forward_batch(ax.tensor([0.0, 1, 2]), 1, 3)
    loss = ax.cross_entropy(logits, ax.tensor([1.0, 2, 0]))
    opt.zero_grad(); loss.backward(); opt.step()
    state_before = [t.tolist() for t in opt.state_dict()]

    prefix = str(tmp_path / "bundle")
    ax.lm.save_bundle(prefix, m, tok, cfg, optimizer=opt)

    m2, tok2, cfg2 = ax.lm.load_bundle(prefix)
    opt2 = ax.optim.Adam(m2.parameters(), lr=0.01)
    ax.lm.load_bundle(prefix, optimizer=opt2)
    state_after = [t.tolist() for t in opt2.state_dict()]
    assert state_after == state_before


def test_no_grad():
    x = ax.tensor([1, 2, 3], requires_grad=True)
    with ax.no_grad():
        y = ax.sum(x * x)
        assert not y.requires_grad
    assert ax.is_grad_enabled()


def test_checkpoint_save_load(tmp_path):
    m1 = ax.nn.Sequential([ax.nn.Linear(3, 4, seed=1), ax.nn.ReLU(), ax.nn.Linear(4, 2, seed=2)])
    x = ax.tensor([[0.5, -1.0, 2.0]])
    before = m1(x).tolist()

    path = str(tmp_path / "model.bin")
    ax.save(m1.parameters(), path)

    # New model (different weights) -> load -> must reproduce the output.
    m2 = ax.nn.Sequential([ax.nn.Linear(3, 4, seed=9), ax.nn.ReLU(), ax.nn.Linear(4, 2, seed=8)])
    ax.load(m2.parameters(), path)
    after = m2(x).tolist()
    assert after == before


def test_lm_trains_generates_and_bundles(tmp_path):
    # Tiny (fast) training: the AxonLM should learn to memorize a short text.
    text = "ab ab ab cd cd cd ab cd ab cd "
    model, tok, config = ax.lm.train_lm(
        text, vocab_size=30, dim=16, num_heads=2, num_layers=1, context=4,
        epochs=15, seed=1, log=None,
    )
    out = ax.lm.generate(model, tok, "ab", n_tokens=6, context=config["context"], top_k=1)
    assert isinstance(out, str) and len(out) > 2

    # Bundle: save and reload reproduces generation exactly (top_k=1 = deterministic).
    prefix = str(tmp_path / "lm")
    ax.lm.save_bundle(prefix, model, tok, config)
    model2, tok2, cfg2 = ax.lm.load_bundle(prefix)
    g1 = ax.lm.generate(model, tok, "ab", n_tokens=6, context=4, top_k=1)
    g2 = ax.lm.generate(model2, tok2, "ab", n_tokens=6, context=4, top_k=1)
    assert g1 == g2


def test_trains_xor():
    x = ax.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = ax.tensor([[0], [1], [1], [0]])

    model = ax.nn.Sequential([
        ax.nn.Linear(2, 8, seed=1),
        ax.nn.ReLU(),
        ax.nn.Linear(8, 1, seed=2),
    ])
    opt = ax.optim.Adam(model.parameters(), lr=0.05)

    first = last = None
    for epoch in range(2000):
        loss = ax.mse_loss(model(x), y)
        if epoch == 0:
            first = loss.item()
        last = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert last < first * 0.1
    assert last < 0.05

    pred = [row[0] for row in model(x).tolist()]
    assert pred[0] < 0.5 and pred[1] > 0.5 and pred[2] > 0.5 and pred[3] < 0.5


def test_cuda_backend_controls():
    """The CUDA control API exists on every build; on CPU-only builds toggling is a
    no-op but matmul must stay correct. On a CUDA build this exercises GPU dispatch."""
    import numpy as np

    assert isinstance(ax.cuda_available(), bool)
    assert isinstance(ax.cuda_device_name(), str)
    assert ax.cuda_available() == bool(ax.cuda_device_name())  # name iff compiled with CUDA

    prev = ax.is_cuda_enabled()
    prev_thr = ax.cuda_matmul_threshold()
    try:
        ax.set_cuda_matmul_threshold(0)          # force every matmul down the GPU path
        rng = np.random.default_rng(0)
        a = rng.standard_normal((96, 64)).astype(np.float32)
        b = rng.standard_normal((64, 48)).astype(np.float32)
        ta, tb = ax.from_numpy(a), ax.from_numpy(b)
        ax.set_cuda_enabled(False); off = ax.matmul(ta, tb).numpy()
        ax.set_cuda_enabled(True);  on = ax.matmul(ta, tb).numpy()
        assert np.abs(off - a @ b).max() < 1e-3
        assert np.abs(on - a @ b).max() < 1e-2   # GPU path (or same CPU path) still correct
        assert ax.is_cuda_enabled() is True
    finally:
        ax.set_cuda_enabled(prev)
        ax.set_cuda_matmul_threshold(prev_thr)
