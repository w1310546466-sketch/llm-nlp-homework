"""从课件文本重建一份可用的分类数据集。

课程分发的是人民日报 9801，这里不方便再分发，所以早期版本用课件正文替代：
每句按所属讲次打标签（L1 / L2 / L3）。现在仓库里提交的 `data/labeled.tsv`
已经换成人民日报标题，这个脚本保留下来，是为了替换语料时仍能跑通全流程。

    python scripts/make_dataset.py --src ../_text_*.txt --out data/labeled.tsv
"""

import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path

CJK = re.compile(r"[\u4e00-\u9fff]")
KEEP = re.compile(r"[^\u4e00-\u9fff，。！？；：、“”‘’（）《》]")

LABELS = {
    "L1.pptx": "L1",
    "L2.part1.pptx": "L2",
    "L2-part2.pptx": "L2",
    "L3-part1-updated.pptx": "L3",
    "L3-part2-updated.pptx": "L3",
}


def sentences(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("--- slide") or re.search(r"\d+\s*/\s*\d+$", line):
                continue
            line = KEEP.sub("", re.sub(r"\s+", " ", line))
            if len(line) < 12 or len(CJK.findall(line)) / len(line) < 0.6:
                continue
            for seg in re.split(r"(?<=[。！？])", line):
                seg = seg.strip()
                if len(seg) >= 12 and re.search(r"[，。；：？！]", seg):
                    yield seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="*", default=None)
    ap.add_argument("--out", default="data/labeled.tsv")
    ap.add_argument("--per-label", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    srcs = []
    for pattern in (args.src or ["../_text_*.txt"]):
        matched = glob.glob(pattern)
        srcs.extend(matched or [pattern])
    buckets = defaultdict(list)
    seen = set()
    for path in srcs:
        name = Path(path).name.replace("_text_2026-秋季-大语言模型与应用实践-", "").replace(".txt", "")
        label = LABELS.get(name)
        if not label:
            continue
        for s in sentences(path):
            if s in seen:
                continue
            seen.add(s)
            buckets[label].append(s)

    import random
    rng = random.Random(args.seed)
    lines = []
    for label, items in sorted(buckets.items()):
        rng.shuffle(items)
        for s in items[:args.per_label]:
            lines.append(f"{label}\t{s}")
    rng.shuffle(lines)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"{out}：{len(lines)} 条，"
          f"{ {k: min(len(v), args.per_label) for k, v in buckets.items()} }")


if __name__ == "__main__":
    main()
