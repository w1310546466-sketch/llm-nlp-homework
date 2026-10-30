"""词向量与分类实验的数据集工具。"""

import random
import re
from pathlib import Path

CJK = re.compile(r"[\u4e00-\u9fff]")
KEEP = re.compile(r"[^\u4e00-\u9fff，。！？；：、“”‘’（）《》]")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = KEEP.sub("", text)
    return text


def load_tsv(path: Path):
    """读取 `标签<TAB>文本` 格式，空行和格式不对的行直接跳过。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            label, _, text = line.partition("\t")
            text = clean_text(text)
            if len(CJK.findall(text)) >= 6:
                rows.append((label.strip(), text))
    return rows


def split(rows, test_ratio=0.2, seed=0):
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n_test = int(len(rows) * test_ratio)
    test = [rows[i] for i in idx[:n_test]]
    train = [rows[i] for i in idx[n_test:]]
    return train, test


def label_distribution(rows):
    out = {}
    for label, _ in rows:
        out[label] = out.get(label, 0) + 1
    return dict(sorted(out.items()))
