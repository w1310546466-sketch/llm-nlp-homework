"""同一套实验的 gensim 版本，给装了 gensim 的机器用。

`veccls/` 里的 numpy 实现是因为这台机器连不上 PyPI，装不了 gensim。在正常
机器上，这个文件是更短的路，也可以用来给 `veccls` 的结果做交叉验证。

    pip install gensim jieba scikit-learn
    python gensim_baseline.py --data data/labeled.tsv

顺带澄清一个常被混用的名字：`gensim.models.FastText` 是无监督的子词向量，
带 `__label__` 的有监督分类在 Facebook 的 `fasttext` 包里，gensim 并不提供。
"""

import argparse
import random
from pathlib import Path

import jieba
import numpy as np
from gensim.models import FastText, Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        label, _, text = line.partition("\t")
        rows.append((label, text))
    return rows


def train_embeddings(texts, dim=100, sg=1):
    # 字符级切分，不需要分词器
    tokenized = [list(t) for t in texts]
    model = Word2Vec(tokenized, vector_size=dim, window=5, min_count=3,
                     sg=sg, negative=5, sample=1e-3, epochs=5, workers=4, seed=42)
    return model


def train_fasttext(texts, dim=100):
    tokenized = [list(t) for t in texts]
    return FastText(tokenized, vector_size=dim, window=5, min_count=3, sg=1,
                    negative=5, min_n=2, max_n=4, bucket=200_000, epochs=5,
                    workers=4, seed=42)


def sentence_vectors(model, texts):
    out = []
    for text in texts:
        vecs = [model.wv[c] for c in text if c in model.wv]
        out.append(np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/labeled.tsv")
    ap.add_argument("--classes", nargs="*", type=int, default=[2, 3])
    args = ap.parse_args()

    rows = load(args.data)
    idx = list(range(len(rows)))
    random.Random(0).shuffle(idx)
    n_test = int(len(rows) * 0.2)
    test_ids, train_ids = set(idx[:n_test]), set(idx[n_test:])

    train = [rows[i] for i in sorted(train_ids)]
    test = [rows[i] for i in sorted(test_ids)]

    w2v = train_embeddings([t for _, t in train])
    print("译 的近邻：", w2v.wv.most_similar("译", topn=6))

    for n_class in args.classes:
        keep = sorted({lab for lab, _ in rows})[:n_class]
        tr = [(lab, t) for lab, t in train if lab in keep]
        te = [(lab, t) for lab, t in test if lab in keep]
        Xtr = sentence_vectors(w2v, [t for _, t in tr])
        Xte = sentence_vectors(w2v, [t for _, t in te])
        clf = LogisticRegression(max_iter=2000).fit(Xtr, [lab for lab, _ in tr])
        acc = accuracy_score([lab for lab, _ in te], clf.predict(Xte))
        print(f"{n_class} 类：{acc:.3f}")


if __name__ == "__main__":
    main()
