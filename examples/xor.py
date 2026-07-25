"""XOR in Python using pyaxon — the same example as examples/xor.cpp, now
with the Python API (PyTorch-style API, core running in C++)."""

import pyaxon as ax

# XOR data
x = ax.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = ax.tensor([[0], [1], [1], [0]])

# Model: 2 -> 8 -> ReLU -> 1
model = ax.nn.Sequential([
    ax.nn.Linear(2, 8, seed=1),
    ax.nn.ReLU(),
    ax.nn.Linear(8, 1, seed=2),
])

opt = ax.optim.Adam(model.parameters(), lr=0.05)

print("Training XOR (via pyaxon in Python)...")
for epoch in range(1, 2001):
    pred = model(x)
    loss = ax.mse_loss(pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 400 == 0 or epoch == 1:
        print(f"epoch {epoch:4d} | loss = {loss.item():.6f}")

print("\nResult (expected: 0, 1, 1, 0):")
pred = model(x)
for i, row in enumerate(pred.tolist()):
    inputs = x.tolist()[i]
    print(f"  {inputs} -> {row[0]:.3f}")
