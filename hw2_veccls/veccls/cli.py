"""命令行入口。

    python -m veccls run --data data/labeled.tsv --out results
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .classifier import BagRegressor, FeatureTable
from .corpus import label_distribution, load_tsv, split
from .skipgram import SkipGram

# 固定类别顺序，保证"2 类"永远是同一对，不受文件里行序影响。
CLASS_ORDER = ["体育", "金融", "教育", "文化", "科技", "医疗", "农业", "法律"]


def pick_bigrams(rows, allowed, min_count=5):
    counts = Counter()
    for _, text in rows:
        for a, b in zip(text, text[1:]):
            if a in allowed and b in allowed:
                counts[a + b] += 1
    return [bg for bg, n in counts.most_common() if n >= min_count]


def run(args):
    rows = load_tsv(Path(args.data))
    print(f"{len(rows)} 条，类别分布 {label_distribution(rows)}", flush=True)
    train, test = split(rows, test_ratio=args.test_ratio, seed=args.seed)

    # 1. 训练 skip-gram。给了正文语料就用正文，标题太短，字向量铺不开。
    if args.embed_corpus:
        embed_sentences = [ln.strip() for ln in
                           Path(args.embed_corpus).read_text(encoding="utf-8").splitlines()
                           if ln.strip()]
        print(f"用 {args.embed_corpus} 的 {len(embed_sentences)} 段正文训练词向量",
              flush=True)
    else:
        embed_sentences = [text for _, text in train]

    t0 = time.perf_counter()
    sg = SkipGram(embed_sentences, dim=args.dim, window=args.window,
                  min_count=args.min_count, negatives=args.negatives,
                  lr=args.lr, epochs=args.epochs, seed=args.seed)
    print(f"字符表 {len(sg.vocab)}，{len(sg.corpus)} 句", flush=True)
    train_sec = sg.fit(verbose=True)

    probes = [w for w in args.probe if w in sg.index] or sg.vocab[:5]
    neighbors = {w: [(k, round(v, 3)) for k, v in sg.neighbors(w, 8)] for w in probes}

    # 2. 分类：先少类，再多类
    bigrams = pick_bigrams(train, sg.index, min_count=args.bigram_min_count)
    tables = {
        "random": FeatureTable(sg.vocab, bigrams, sg.dim, seed=1),
        "skipgram": FeatureTable.from_skipgram(sg, bigrams),
    }
    labels_all = [c for c in CLASS_ORDER if c in {lab for lab, _ in rows}]
    labels_all += sorted({lab for lab, _ in rows} - set(labels_all))
    results = {}

    for n_class in args.classes:
        keep = set(labels_all[:n_class])
        tr = [(lab, txt) for lab, txt in train if lab in keep]
        te = [(lab, txt) for lab, txt in test if lab in keep]
        lab2id = {lab: i for i, lab in enumerate(sorted(keep))}

        for table_name, table in tables.items():
            for use_bg in (False, True):
                Xtr = table.build([t for _, t in tr], use_bg)
                Xte = table.build([t for _, t in te], use_bg)
                ytr = [lab2id[lab] for lab, _ in tr]
                yte = [lab2id[lab] for lab, _ in te]

                clf = BagRegressor(len(lab2id), sg.dim, seed=0)
                clf.fit(Xtr, ytr, epochs=args.classify_epochs, lr=args.classify_lr)
                acc = clf.accuracy(Xte, yte)
                key = f"{table_name}/{'uni+bi' if use_bg else 'uni'}"
                results.setdefault(f"{n_class}类", {})[key] = round(acc, 4)
                print(f"  {n_class} 类 | {key}：训练 {len(Xtr)}，"
                      f"测试 {len(Xte)}，准确率 {acc:.3f}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": str(args.data),
        "rows": len(rows),
        "label_dist": label_distribution(rows),
        "skipgram": {
            "dim": sg.dim, "window": sg.window, "negatives": sg.negatives,
            "epochs": sg.epochs, "lr": sg.lr, "train_seconds": round(train_sec, 1),
            "vocab": len(sg.vocab),
        },
        "neighbors": neighbors,
        "classification": results,
        "total_seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "veccls_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(out / "char_vectors.npy", sg.w_in)
    (out / "vocab.txt").write_text("\n".join(sg.vocab), encoding="utf-8")
    print(json.dumps(payload["classification"], ensure_ascii=False, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="veccls")
    sub = p.add_subparsers(required=True)

    r = sub.add_parser("run", help="训练词向量并评测分类器")
    r.add_argument("--data", default="data/labeled.tsv")
    r.add_argument("--embed-corpus", default="", help="可选：用于训练词向量的正文")
    r.add_argument("--out", default="results")
    r.add_argument("--dim", type=int, default=64)
    r.add_argument("--window", type=int, default=3)
    r.add_argument("--min-count", type=int, default=3)
    r.add_argument("--negatives", type=int, default=5)
    r.add_argument("--epochs", type=int, default=3)
    r.add_argument("--lr", type=float, default=0.05)
    r.add_argument("--classes", nargs="*", type=int, default=[2, 3])
    r.add_argument("--bigram-min-count", type=int, default=5)
    r.add_argument("--classify-epochs", type=int, default=40)
    r.add_argument("--classify-lr", type=float, default=0.5)
    r.add_argument("--test-ratio", type=float, default=0.2)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--probe", nargs="*", default=["译", "奖", "检", "模"])
    r.set_defaults(func=run)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
