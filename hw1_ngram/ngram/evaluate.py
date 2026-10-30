"""困惑度与覆盖率统计。"""

import math

from .model import BOS, EOS


def perplexity(model, sentences, order: int) -> float:
    pad_n = order - 1
    total, n = 0.0, 0
    for sent in sentences:
        seq = [BOS] * pad_n + list(sent) + [EOS]
        for i in range(pad_n, len(seq)):
            ctx = tuple(seq[max(0, i - pad_n):i])
            p = model.prob(ctx, seq[i])
            if p <= 0:
                return float("inf")
            total += math.log(p)
            n += 1
    return math.exp(-total / max(n, 1))


def unseen_ratio(counts, sentences, order: int) -> float:
    """测试集中未在训练计数里出现过的 n-gram 比例。"""
    pad_n = order - 1
    total = seen = 0
    for sent in sentences:
        seq = [BOS] * pad_n + list(sent) + [EOS]
        for i in range(len(seq) - order + 1):
            gram = tuple(seq[i:i + order])
            total += 1
            if counts.count[order].get(gram, 0) > 0:
                seen += 1
    return 1 - seen / max(total, 1)
