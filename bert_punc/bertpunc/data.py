"""数据集构造：把带标点的句子拆成"无标点输入 + 逐字标签"。

中文 BERT 基本一字一 token，所以字符位置和 token 位置几乎一一对应；遇到需要
切子词的 tokenizer 时，只保留每个词第一个子词的标签，其余置 -100 不计分。
"""

import random
from pathlib import Path

PUNCT = ["，", "。", "？", "！"]
LABELS = ["O"] + PUNCT
LABEL2ID = {c: i for i, c in enumerate(LABELS)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


def load_sentences(path, min_len=10, max_len=120):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if min_len <= len(s) <= max_len:
                out.append(s)
    return out


def split_chars(sentence):
    """去掉标点得到模型输入；标点作为它前面那个字的标签。"""
    chars, labels = [], []
    for ch in sentence:
        if ch in LABEL2ID:
            if chars:
                labels[-1] = LABEL2ID[ch]
        else:
            chars.append(ch)
            labels.append(0)
    return chars, labels


class PuncDataset:
    """按需分词的数据集；只需要 tokenizer，不依赖 torch 的 DataLoader。"""

    def __init__(self, sentences, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.items = []
        for s in sentences:
            chars, char_labels = split_chars(s)
            if len(chars) >= 8:
                self.items.append((chars, char_labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch

        chars, char_labels = self.items[i]
        enc = self.tokenizer(chars, is_split_into_words=True, truncation=True,
                             max_length=self.max_len, padding="max_length")
        word_ids = enc.word_ids()
        labels, prev = [], None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev:
                labels.append(char_labels[wid])
            else:
                labels.append(-100)
            prev = wid
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(labels),
        }


def split_train_valid(sentences, n_train, n_valid, seed=0):
    rng = random.Random(seed)
    rng.shuffle(sentences)
    return sentences[:n_train], sentences[n_train:n_train + n_valid]


def read_text(path: Path) -> str:
    return "".join(path.read_text(encoding="utf-8").split())
