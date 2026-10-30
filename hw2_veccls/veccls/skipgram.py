"""Skip-gram with negative sampling, written with numpy only.

This mirrors what word2vec does internally: subsample frequent tokens, take
window pairs, push the true context up and a few sampled contexts down.
"""

import time

import numpy as np

from .corpus import CJK

CJK_CHAR = CJK


class SkipGram:
    def __init__(self, sentences, dim=64, window=3, min_count=3, negatives=5,
                 lr=0.05, epochs=3, subsample=1e-4, seed=42):
        self.dim = dim
        self.window = window
        self.negatives = negatives
        self.lr = lr
        self.epochs = epochs
        self.subsample = subsample
        self.rng = np.random.default_rng(seed)

        freq = {}
        for sent in sentences:
            for ch in sent:
                if CJK_CHAR.match(ch):
                    freq[ch] = freq.get(ch, 0) + 1

        ordered = sorted(freq.items(), key=lambda kv: -kv[1])
        self.vocab = [w for w, n in ordered if n >= min_count]
        self.index = {w: i for i, w in enumerate(self.vocab)}
        self.freq = np.array([freq[w] for w in self.vocab], dtype=np.float64)

        self.corpus = []
        for sent in sentences:
            ids = [self.index[c] for c in sent if c in self.index]
            if len(ids) > 1:
                self.corpus.append(ids)

        n = len(self.vocab)
        self.w_in = (self.rng.random((n, dim)) - 0.5) / dim
        self.w_out = np.zeros((n, dim))

        total = self.freq.sum()
        p = self.freq ** 0.75
        self.neg_dist = p / p.sum()

        rel = self.freq / total
        self.keep_prob = np.minimum(1.0, (np.sqrt(rel / subsample) + 1) * subsample / rel)

    # -- training -------------------------------------------------------

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -6.0, 6.0)))

    def fit(self, verbose=False):
        rng = self.rng
        start = time.perf_counter()
        n_vocab = len(self.vocab)
        for epoch in range(self.epochs):
            for sent in self.corpus:
                for i, center in enumerate(sent):
                    if rng.random() > self.keep_prob[center]:
                        continue
                    lo = max(0, i - self.window)
                    hi = min(len(sent), i + self.window + 1)
                    for j in range(lo, hi):
                        if j == i:
                            continue
                        ctx = sent[j]
                        v = self.w_in[center]
                        u = self.w_out[ctx]
                        grad = (1.0 - self._sigmoid(float(v @ u))) * self.lr
                        g = grad * u
                        self.w_out[ctx] += grad * v

                        for neg in rng.choice(n_vocab, size=self.negatives, p=self.neg_dist):
                            if neg == ctx:
                                continue
                            u = self.w_out[neg]
                            gn = -self._sigmoid(float(v @ u)) * self.lr
                            self.w_out[neg] += gn * v
                            g = g + gn * u
                        self.w_in[center] += g
            if verbose:
                print(f"  epoch {epoch + 1}/{self.epochs} "
                      f"{time.perf_counter() - start:.1f}s", flush=True)
        return time.perf_counter() - start

    # -- inspection -----------------------------------------------------

    def vector(self, word):
        return self.w_in[self.index[word]] if word in self.index else None

    def neighbors(self, word, topn=8):
        v = self.vector(word)
        if v is None:
            return []
        mat = self.w_in / (np.linalg.norm(self.w_in, axis=1, keepdims=True) + 1e-9)
        sims = mat @ (v / (np.linalg.norm(v) + 1e-9))
        order = np.argsort(-sims)
        out = []
        for i in order:
            if len(out) >= topn:
                break
            if self.vocab[i] != word:
                out.append((self.vocab[i], float(sims[i])))
        return out

    def analogy(self, a, b, c, topn=5):
        va, vb, vc = self.vector(a), self.vector(b), self.vector(c)
        if va is None or vb is None or vc is None:
            return []
        target = vb - va + vc
        mat = self.w_in / (np.linalg.norm(self.w_in, axis=1, keepdims=True) + 1e-9)
        sims = mat @ (target / (np.linalg.norm(target) + 1e-9))
        out = []
        for i in np.argsort(-sims):
            w = self.vocab[i]
            if w in (a, b, c) or len(out) >= topn:
                continue
            out.append((w, float(sims[i])))
        return out
