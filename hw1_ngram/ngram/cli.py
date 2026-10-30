"""Command line entry point.

    python -m ngram build --raw data/raw/*.txt --out data/corpus.txt
    python -m ngram train --corpus data/corpus.txt --orders 1 2 3 4 5
"""

import argparse
import glob
import json
import random
import time
from pathlib import Path

from . import corpus as corpus_mod
from .evaluate import perplexity, unseen_ratio
from .generate import sample
from .model import AddK, Counts, KneserNey, MLE, StupidBackoff, build


def split(sentences, test_ratio=0.1, seed=42):
    idx = list(range(len(sentences)))
    random.Random(seed).shuffle(idx)
    n_test = int(len(sentences) * test_ratio)
    return ([sentences[i] for i in idx[n_test:]],
            [sentences[i] for i in idx[:n_test]])


def cmd_build(args):
    raws = args.raw or sorted(glob.glob("data/raw/*.txt"))
    if not raws:
        raise SystemExit("no raw files; pass --raw or drop files into data/raw/")
    stats = corpus_mod.build_corpus(raws, Path(args.out))
    stats["sources"] = [Path(p).name for p in raws]
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_train(args):
    sentences = corpus_mod.load(Path(args.corpus))
    train, test = split(sentences)
    vocab = len({ch for s in train for ch in s})
    print(f"train {len(train)} sents / test {len(test)} sents, vocab {vocab}",
          flush=True)

    rows = []
    for order in args.orders:
        t0 = time.perf_counter()
        counts = build(train, order)
        build_sec = time.perf_counter() - t0
        pad_n = order - 1

        row = {
            "order": order,
            "entries": counts.n_entries,
            "build_seconds": round(build_sec, 3),
            "unseen_ratio": round(unseen_ratio(counts, test, order), 4),
            "perplexity": {},
        }

        # Kneser-Ney needs continuation counts, which only exist from order 2 up.
        models = [MLE(counts), AddK(counts, 1.0, vocab), AddK(counts, 0.01, vocab),
                  StupidBackoff(counts)]
        if order >= 2:
            models.append(KneserNey(counts))

        for m in models:
            t1 = time.perf_counter()
            ppl = perplexity(m, test, order)
            row["perplexity"][m.name] = (
                None if ppl == float("inf") else round(ppl, 2))
            row["perplexity"][m.name + "_seconds"] = round(time.perf_counter() - t1, 3)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    # Samples from the best-performing configuration on this corpus.
    counts3 = build(train, 3)
    counts5 = build(train, 5)
    gens = {
        "seed": args.seed_text,
        "kn3": [sample(KneserNey(counts3), args.seed_text, 100, 3, random.Random(i))
                for i in range(3)],
        "sb3": [sample(StupidBackoff(counts3), args.seed_text, 100, 3, random.Random(i))
                for i in range(3)],
        "kn5": [sample(KneserNey(counts5), args.seed_text, 100, 5, random.Random(i))
                for i in range(3)],
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus": {
            "file": str(args.corpus),
            "train_sentences": len(train),
            "test_sentences": len(test),
            "train_chars": sum(len(s) for s in train),
            "test_chars": sum(len(s) for s in test),
            "vocab": vocab,
        },
        "results": rows,
        "generations": gens,
    }
    (out / "ngram_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nsamples")
    for key, values in gens.items():
        if key == "seed":
            continue
        print(f"[{key}]")
        for s in values:
            print("  " + s)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ngram")
    sub = p.add_subparsers(required=True)

    b = sub.add_parser("build", help="clean raw text into a corpus")
    b.add_argument("--raw", nargs="*")
    b.add_argument("--out", default="data/corpus.txt")
    b.set_defaults(func=cmd_build)

    t = sub.add_parser("train", help="fit models and report perplexity")
    t.add_argument("--corpus", default="data/corpus.txt")
    t.add_argument("--orders", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    t.add_argument("--out", default="results")
    t.add_argument("--seed-text", default="在阳光明媚的五月")
    t.set_defaults(func=cmd_train)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
