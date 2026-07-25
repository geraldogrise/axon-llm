"""High-level helpers for the AxonLM: train, generate, save and load a complete
"bundle" (weights + tokenizer + config).

Used both by the CLI (`python -m pyaxon`) and by the example server.
"""

import json
import math
import random
import time
from contextlib import contextmanager

from . import data
from ._axon import BPETokenizer, cross_entropy, is_grad_enabled
from ._axon import load as _load_tensors
from ._axon import nn, optim
from ._axon import save as _save_tensors
from ._axon import set_grad_enabled, tensor


@contextmanager
def _no_grad():
    prev = is_grad_enabled()
    set_grad_enabled(False)
    try:
        yield
    finally:
        set_grad_enabled(prev)


def cosine_lr(step, total_steps, base_lr, warmup_steps=0):
    """Learning rate scheduler: linear warmup followed by cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_lm(text, *, vocab_size=100, dim=64, num_heads=4, num_layers=3, context=24,
             epochs=8, lr=0.002, batch_size=8, seed=1, schedule=False, warmup=0, log=print):
    """Train an AxonLM on `text`. Returns (model, tokenizer, config).

    schedule=True uses warmup + cosine decay on the learning rate.
    """
    random.seed(seed)
    tok = BPETokenizer()
    tok.train(text, vocab_size)
    ids = tok.encode(text)
    windows = data.make_windows(ids, context)
    if not windows:
        raise ValueError("text too short for the chosen context_len")

    model = nn.AxonLM(vocab_size=tok.vocab_size, dim=dim, num_heads=num_heads,
                      num_layers=num_layers, seed=seed)
    opt = optim.Adam(model.parameters(), lr=lr)

    steps_per_epoch = max(1, (len(windows) + batch_size - 1) // batch_size)
    total_steps = steps_per_epoch * epochs
    step = 0
    if log:
        log(f"vocab={tok.vocab_size} tokens={len(ids)} windows={len(windows)}")
    for epoch in range(1, epochs + 1):
        total = n = 0
        t0 = time.perf_counter()
        tok_count = 0
        for batch in data.iter_batches(windows, batch_size):
            if schedule:
                opt.set_lr(cosine_lr(step, total_steps, lr, warmup))
            # 3D batch: flatten B sequences of size `context` into a single
            # forward_batch call (one big matmul instead of B small ones).
            b = len(batch)
            x_flat, y_flat = [], []
            for x_ids, y_ids in batch:
                x_flat += [float(t) for t in x_ids]
                y_flat += [float(t) for t in y_ids]
            logits = model.forward_batch(tensor(x_flat), b, context)
            loss = cross_entropy(logits, tensor(y_flat))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
            step += 1
            tok_count += b * context
        if log:
            dt = time.perf_counter() - t0
            tps = tok_count / dt if dt > 0 else 0.0
            log(f"epoch {epoch}/{epochs} loss={total / n:.4f} lr={opt.lr:.5f} "
                f"tokens/s={tps:,.0f}")

    config = {"vocab_size": tok.vocab_size, "dim": dim, "num_heads": num_heads,
              "num_layers": num_layers, "context": context, "seed": seed,
              "trained_epochs": epochs}
    return model, tok, config


def _sample(logits_row, temperature, top_k, top_p):
    """Sample a token with temperature + top-k and/or top-p (nucleus).

    top_k > 0: keep only the k highest logits.
    top_p in (0,1): keep the smallest set whose cumulative prob. >= top_p.
    """
    temperature = max(0.05, temperature)
    order = sorted(range(len(logits_row)), key=lambda i: logits_row[i], reverse=True)
    if top_k and top_k > 0:
        order = order[:top_k]
    m = logits_row[order[0]]
    probs = [(i, math.exp((logits_row[i] - m) / temperature)) for i in order]
    s = sum(p for _, p in probs)
    probs = [(i, p / s) for i, p in probs]
    if top_p and 0.0 < top_p < 1.0:
        kept, acc = [], 0.0
        for i, p in probs:  # already sorted by decreasing prob.
            kept.append((i, p))
            acc += p
            if acc >= top_p:
                break
        probs = kept
    total = sum(p for _, p in probs)
    r = random.random() * total
    acc = 0.0
    for i, p in probs:
        acc += p
        if r <= acc:
            return i
    return probs[-1][0]


# Kept for backward compatibility.
def sample_topk(logits_row, k=5, temperature=0.7):
    return _sample(logits_row, temperature, k, 0.0)


def generate(model, tokenizer, prompt, *, n_tokens=40, context=24, top_k=5, top_p=0.0,
             temperature=0.7, use_cache=True):
    """Generate text autoregressively (uses no_grad: no graph is built).

    use_cache=True uses a KV-cache: processes one token at a time without recomputing
    the K/V projections of the past (much faster on long sequences).
    Stops if the <eos> token is sampled. Accepts top-k and/or top-p.
    """
    ids = tokenizer.encode(prompt) or [0]
    eos = tokenizer.eos_id
    with _no_grad():
        if use_cache:
            model.reset_cache()
            logits = None
            for pos, t in enumerate(ids):
                logits = model.forward_step(int(t), pos)  # (1, V)
            pos = len(ids)
            for _ in range(int(n_tokens)):
                nxt = _sample(logits.tolist()[-1], float(temperature), int(top_k), float(top_p))
                if nxt == eos:
                    break
                ids.append(nxt)
                logits = model.forward_step(nxt, pos)
                pos += 1
        else:
            for _ in range(int(n_tokens)):
                ctx = ids[-context:]
                logits = model(tensor([float(t) for t in ctx])).tolist()
                nxt = _sample(logits[-1], float(temperature), int(top_k), float(top_p))
                if nxt == eos:
                    break
                ids.append(nxt)
    return tokenizer.decode(ids)


def save_bundle(prefix, model, tokenizer, config, optimizer=None):
    """Save weights (.ckpt), tokenizer (.bpe), config (.json) and, if given, the
    optimizer state (.opt) -- allowing training to resume exactly."""
    with open(prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(config, f)
    _save_tensors(model.parameters(), prefix + ".ckpt")
    tokenizer.save(prefix + ".bpe")
    if optimizer is not None:
        _save_tensors(optimizer.state_dict(), prefix + ".opt")


def load_bundle(prefix, optimizer=None):
    """Carrega um bundle. Retorna (model, tokenizer, config). Se `optimizer` for
    dado e existir um .opt, restaura o estado do otimizador (retomar treino)."""
    import os

    with open(prefix + ".json", "r", encoding="utf-8") as f:
        config = json.load(f)
    tok = BPETokenizer()
    tok.load(prefix + ".bpe")
    model = nn.AxonLM(vocab_size=config["vocab_size"], dim=config["dim"],
                      num_heads=config["num_heads"], num_layers=config["num_layers"],
                      seed=config.get("seed", 1))
    _load_tensors(model.parameters(), prefix + ".ckpt")
    if optimizer is not None and os.path.exists(prefix + ".opt"):
        state = optimizer.state_dict()  # tensors with the right shapes
        _load_tensors(state, prefix + ".opt")
        optimizer.load_state_dict(state)
    return model, tok, config
