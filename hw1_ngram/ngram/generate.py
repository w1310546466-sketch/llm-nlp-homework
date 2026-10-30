"""Sampling from an n-gram model."""

import random

from .model import BOS


def alphabet(model):
    """Characters the model has seen, excluding boundary symbols."""
    chars = [g[0] for g in model.c.count[1] if g[0] not in (BOS, "</s>")]
    return sorted(chars)


def sample(model, seed: str, n_chars: int = 100, order: int = 3,
           rng: random.Random | None = None) -> str:
    rng = rng or random.Random(0)
    chars = alphabet(model)
    if not chars:
        raise ValueError("model has an empty vocabulary")

    hist = list(seed)
    out = list(seed)
    for _ in range(n_chars):
        ctx = tuple(hist[-(order - 1):]) if order > 1 else ()
        probs = [model.prob(ctx, c) for c in chars]
        total = sum(probs)
        if total <= 0:
            nxt = rng.choice(chars)
        else:
            r = rng.random() * total
            acc = 0.0
            nxt = chars[-1]
            for c, p in zip(chars, probs):
                acc += p
                if acc >= r:
                    nxt = c
                    break
        out.append(nxt)
        hist.append(nxt)
    return "".join(out)
