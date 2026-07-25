"""Chatbot based on a Markov chain (intermediate project from the plan).

Does not use a neural network: it learns the transition probabilities between
words from a text and generates new sentences by sampling the chain. It is the
simplest text-generation baseline — useful to compare against AxonLM (Transformer).
"""

import random

random.seed(0)

TEXT = (
    "o rato roeu a roupa do rei de roma. "
    "o rei de roma mandou prender o rato. "
    "o rato fugiu para a casa da rainha. "
    "a rainha gostou do rato e o rato ficou. "
) * 3


def build_chain(text, order=1):
    """Build the chain: (n previous words) -> list of next words."""
    words = text.split()
    chain = {}
    for i in range(len(words) - order):
        key = tuple(words[i:i + order])
        chain.setdefault(key, []).append(words[i + order])
    return chain, words


def generate(chain, words, order=1, n=25):
    """Generate text by sampling the chain from a random starting point."""
    start = random.randrange(len(words) - order)
    state = tuple(words[start:start + order])
    out = list(state)
    for _ in range(n):
        nxts = chain.get(state)
        if not nxts:
            break
        nxt = random.choice(nxts)
        out.append(nxt)
        state = tuple(out[-order:])
    return " ".join(out)


if __name__ == "__main__":
    chain, words = build_chain(TEXT, order=2)
    print("Markov chatbot (order 2) — generated text:")
    print(" ", generate(chain, words, order=2, n=30))
