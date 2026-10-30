"""Bag-of-characters classifier on top of embeddings.

Same shape as fastText supervised: average the vectors of the text's features,
then a linear layer with softmax. The only trick here is that the feature table
can be initialised from skip-gram vectors instead of random numbers.
"""

import numpy as np


class BagRegressor:
    def __init__(self, n_classes, dim, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, size=(n_classes, dim))
        self.b = np.zeros(n_classes)

    @staticmethod
    def softmax(z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def fit(self, X, y, epochs=40, lr=0.5, l2=1e-4, batch=32, seed=0):
        rng = np.random.default_rng(seed)
        idx = np.arange(len(X))
        for _ in range(epochs):
            rng.shuffle(idx)
            for start in range(0, len(X), batch):
                part = idx[start:start + batch]
                gw = np.zeros_like(self.w)
                gb = np.zeros_like(self.b)
                for i in part:
                    p = self.softmax(self.w @ X[i] + self.b)
                    oh = np.zeros_like(p)
                    oh[y[i]] = 1.0
                    diff = p - oh
                    gw += np.outer(diff, X[i])
                    gb += diff
                self.w -= lr * (gw / len(part) + l2 * self.w)
                self.b -= lr * (gb / len(part))
        return self

    def predict(self, X):
        if not X:
            return np.array([], dtype=int)
        X = np.asarray(X)
        return np.argmax(X @ self.w.T + self.b, axis=1)

    def accuracy(self, X, y):
        if not X:
            return 0.0
        return float((self.predict(X) == np.asarray(y)).mean())


class FeatureTable:
    """Holds one vector per unigram character and per bigram."""

    def __init__(self, chars, bigrams, dim, seed=1):
        self.chars = list(chars)
        self.char_index = {c: i for i, c in enumerate(self.chars)}
        self.bigrams = list(bigrams)
        self.bigram_index = {b: len(self.chars) + i for i, b in enumerate(self.bigrams)}
        self.dim = dim
        self.table = np.random.default_rng(seed).normal(0, 0.1, size=(len(self.chars) + len(self.bigrams), dim))

    @classmethod
    def from_skipgram(cls, sg, bigrams=(), dim=None):
        dim = dim or sg.dim
        table = cls(sg.vocab, bigrams, dim)
        table.table[:len(sg.vocab)] = sg.w_in
        for b in table.bigrams:
            a, c = b[0], b[1]
            if a in sg.index and c in sg.index:
                table.table[table.bigram_index[b]] = (sg.w_in[sg.index[a]] + sg.w_in[sg.index[c]]) / 2
        return table

    def features(self, text, use_bigrams=True):
        ids = [self.char_index[c] for c in text if c in self.char_index]
        if use_bigrams:
            ids += [self.bigram_index[a + b] for a, b in zip(text, text[1:])
                    if a + b in self.bigram_index]
        if not ids:
            return np.zeros(self.dim)
        return self.table[ids].mean(axis=0)

    def build(self, texts, use_bigrams=True):
        return [self.features(t, use_bigrams) for t in texts]
