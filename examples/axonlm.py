"""AxonLM — mini generative language model (decoder-only Transformer).

Trains at the character level on a short text and then GENERATES text,
predicting one character at a time (autoregressive). It is the project's
"climax": a tiny generative AI, trained from scratch, running on pyaxon's C++ core.
"""

import math
import random

import pyaxon as ax

random.seed(0)

# --- Training corpus (short, to fit on the CPU) ---
TEXT = "o rato roeu a roupa do rei de roma. "

# --- Char-level tokenizer: maps each character to an id ---
chars = sorted(set(TEXT))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)
print(f"Corpus: {len(TEXT)} characters | vocabulary: {vocab_size} symbols")


def encode(s):
    return [float(stoi[c]) for c in s]


# Input and target sequences (predict the next character).
data = encode(TEXT)
x = ax.tensor(data[:-1])       # tokens 0..n-1
y = ax.tensor(data[1:])        # tokens 1..n  (shifted by 1)

# --- Model ---
model = ax.nn.AxonLM(vocab_size=vocab_size, dim=64, num_heads=4, num_layers=2, seed=1)
opt = ax.optim.Adam(model.parameters(), lr=0.003)

print("\nTraining AxonLM (char-level)...")
for epoch in range(1, 401):
    logits = model(x)
    loss = ax.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 50 == 0 or epoch == 1:
        print(f"  epoch {epoch:4d} | loss = {loss.item():.4f}")


def softmax_sample(logits_row, temperature):
    """Sample an index from a row of logits, with temperature."""
    m = max(logits_row)
    exps = [math.exp((v - m) / temperature) for v in logits_row]
    s = sum(exps)
    probs = [e / s for e in exps]
    r = random.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1


def generate(prompt, n_chars, temperature=0.5):
    """Generate text autoregressively from a prompt."""
    ids = encode(prompt)
    out = list(prompt)
    with ax.no_grad():  # inference: no need to build the autograd graph
        for _ in range(n_chars):
            logits = model(ax.tensor(ids)).tolist()
            nxt = softmax_sample(logits[-1], temperature)  # use the last position
            out.append(itos[nxt])
            ids.append(float(nxt))
    return "".join(out)


print("\n--- Generation (prompt = 'o rato') ---")
print(generate("o rato", 40, temperature=0.4))

# --- Checkpoint: save and reload the weights ---
import os

ckpt = os.path.join(os.path.dirname(__file__), "axonlm.ckpt")
ax.save(model.parameters(), ckpt)
print(f"\nCheckpoint saved to {ckpt} ({os.path.getsize(ckpt)} bytes)")

# New model with the same hyperparameters -> load the weights -> same generation.
model2 = ax.nn.AxonLM(vocab_size=vocab_size, dim=64, num_heads=4, num_layers=2, seed=999)
ax.load(model2.parameters(), ckpt)
model = model2
print("After reloading the checkpoint:")
print(generate("o rato", 40, temperature=0.4))
os.remove(ckpt)
