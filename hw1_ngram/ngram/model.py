"""n-gram 计数与各平滑估计器。

这里全部是字符级，因为字符级不需要分词器，对任何中文文本都能直接跑。
换成词级只需要改调用方一行：把字符列表换成词列表。
"""

import math
from collections import defaultdict

BOS = "<s>"
EOS = "</s>"


class Counts:
    """保存 1..N 阶的原始 n-gram 计数，以及每个上下文的总次数。"""

    def __init__(self, order: int):
        if order < 1:
            raise ValueError("阶数必须大于等于 1")
        self.order = order
        self.count = [None] + [defaultdict(int) for _ in range(order)]
        self.ctx_total = [None] + [defaultdict(int) for _ in range(order)]

    def add(self, sentence: str):
        pad_n = self.order - 1
        seq = [BOS] * pad_n + list(sentence) + [EOS]
        for k in range(1, self.order + 1):
            ck, ctot = self.count[k], self.ctx_total[k]
            for i in range(len(seq) - k + 1):
                gram = tuple(seq[i:i + k])
                ck[gram] += 1
                if k >= 2:
                    ctot[gram[:-1]] += 1
        # 一元的分母就是 token 总数，统一记在空上下文下。
        self.ctx_total[1][()] += len(seq)

    @property
    def n_entries(self) -> int:
        return sum(len(self.count[k]) for k in range(1, self.order + 1))

    def unigram_total(self) -> int:
        return self.ctx_total[1].get((), 0)


def build(sentences, order: int) -> Counts:
    counts = Counts(order)
    for s in sentences:
        counts.add(s)
    return counts


class MLE:
    """最大似然估计。未登录的 n-gram 概率为 0，困惑度会直接崩掉。"""

    name = "mle"

    def __init__(self, counts: Counts):
        self.c = counts

    def prob(self, ctx, word) -> float:
        k = len(ctx) + 1
        num = self.c.count[k].get(ctx + (word,), 0)
        den = self.c.ctx_total[k].get(ctx, 0)
        return num / den if den else 0.0


class AddK:
    """加 k 平滑，k = 1 即 Laplace。"""

    def __init__(self, counts: Counts, k: float = 1.0, vocab_size: int | None = None):
        self.c = counts
        self.k = k
        self.vocab = vocab_size or len(counts.count[1])
        self.name = "laplace" if k == 1.0 else f"add-{k}"

    def prob(self, ctx, word) -> float:
        n = len(ctx) + 1
        num = self.c.count[n].get(ctx + (word,), 0) + self.k
        den = self.c.ctx_total[n].get(ctx, 0) + self.k * self.vocab
        return num / den


class StupidBackoff:
    """计数不打折，回退时乘一个固定惩罚。不保证归一化。"""

    name = "stupid-backoff"

    def __init__(self, counts: Counts, alpha: float = 0.4):
        self.c = counts
        self.alpha = alpha
        self.order = counts.order

    def prob(self, ctx, word) -> float:
        for k in range(min(len(ctx) + 1, self.order), 0, -1):
            sub = ctx[len(ctx) - (k - 1):] if k > 1 else ()
            num = self.c.count[k].get(sub + (word,), 0)
            if num:
                den = self.c.ctx_total[k].get(sub, 0)
                if den:
                    return (num / den) * (self.alpha ** (len(ctx) + 1 - k))
        return 1e-10


class KneserNey:
    """插值 Kneser-Ney，使用单一折扣。

    低阶不用原始频次，而用续接计数——这个字前面出现过多少种不同的字。
    这样回退分布衡量的是"能组合进多少种上下文"，而不是"多常见"。
    """

    name = "kneser-ney"

    def __init__(self, counts: Counts, discount: float = 0.75):
        self.c = counts
        self.D = discount
        self.order = counts.order
        # 续接计数：只统计"作为后继出现过"，供低阶回退使用
        self._cont = [None] + [defaultdict(int) for _ in range(self.order + 1)]
        for k in range(2, self.order + 1):
            for gram in self.c.count[k]:
                self._cont[k - 1][gram[1:]] += 1

        # 每个上下文下出现过多少种不同的后继，用于算插值权重
        self._types = [None] + [defaultdict(int) for _ in range(self.order + 1)]
        for k in range(1, self.order + 1):
            for gram in self.c.count[k]:
                self._types[k][gram[:-1]] += 1

        self._cont_total = sum(self._cont[1].values()) if self.order > 1 else 0

    def _unigram(self, word) -> float:
        if self.order == 1:
            # 一元没有续接计数可用，退化为最大似然
            num = self.c.count[1].get((word,), 0)
            den = self.c.unigram_total()
            return num / den if den else 1e-10
        num = self._cont[1].get((word,), 0)
        if num == 0:
            return 1e-10
        return num / self._cont_total

    def prob(self, ctx, word) -> float:
        if not ctx:
            return self._unigram(word)
        k = len(ctx) + 1
        if k > self.order:
            ctx = ctx[-(self.order - 1):]
            k = self.order
        c_hw = self.c.count[k].get(ctx + (word,), 0)
        c_h = self.c.ctx_total[k].get(ctx, 0)
        if c_h == 0:
            # 上下文本身没见过，直接退到更短的上下文
            return self.prob(ctx[1:], word)
        lower = self.prob(ctx[1:], word)
        lam = self.D * self._types[k].get(ctx, 0) / c_h
        if c_hw == 0:
            return lam * lower
        return max(c_hw - self.D, 0) / c_h + lam * lower


def logprob(model, ctx, word) -> float:
    p = model.prob(ctx, word)
    return math.log(p) if p > 0 else float("-inf")
