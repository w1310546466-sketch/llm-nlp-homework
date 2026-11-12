# -*- coding: utf-8 -*-
"""不依赖模型的实验：错误记忆的检索覆盖率（第二版）。

第一版两个问题：
  1. "否定"用单字「不」「未」匹配，290 条全被误判；
  2. 实词覆盖用 2-4 字 n-gram，几乎不可能重合，得 0.021 是度量本身的问题。

这一版改成短语级线索，重叠改用字符 4-gram 的 Jaccard。
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"D:\DSH\大语言模型\work\l4_protocols")
from run_protocols import BM25  # noqa: E402

BASE = Path(r"D:\DSH\大语言模型")
OUT = Path(r"D:\DSH\大语言模型\work\l4_protocols\results")

# 短语级线索，避免单字误触发
CATS = {
    "术语": ["术语", "译名", "定名", "词条", "名词"],
    "否定": ["否定", "双重否定", "否定的", "未表达否定", "否定词"],
    "情态": ["情态", "shall", "must", "may", "义务", "允许"],
    "条件": ["条件", "前提", "除非", "unless", "如果"],
    "范围": ["范围", "适用", "管辖", "限定", "覆盖"],
    "指代": ["指代", "指称", "前者", "后者", "latter", "主语"],
    "漏译": ["漏译", "漏掉", "缺译", "未译", "省略", "遗漏"],
    "语序": ["语序", "句式", "顺序", "结构", "语序"],
    "一致性": ["一致", "前后", "统一", "对应"],
}


def classify(reason):
    hits = [k for k, kws in CATS.items() if any(kw in reason for kw in kws)]
    return hits or ["其他"]


def grams(text, n=4):
    t = re.sub(r"[^\u4e00-\u9fff]", "", text)
    return {t[i:i + n] for i in range(max(len(t) - n + 1, 0))}


def main():
    err = json.loads((BASE / "translation_errors.json").read_text(encoding="utf-8"))
    srcs = [r["原始英文句子"] for r in err]
    bm = BM25(srcs)
    n = len(err)

    hit1 = hit3 = classified = clf_hit1 = clf_hit3 = 0
    jac = []
    dist = Counter()

    for i, r in enumerate(err):
        scores = bm.score(r["原始英文句子"])
        scores[i] = -1
        top = list(scores.argsort()[::-1][:3])

        t = classify(r["错误原因"])
        dist.update(t)
        nb = [classify(err[j]["错误原因"]) for j in top]
        same1 = bool(set(t) & set(nb[0]))
        same3 = any(set(t) & set(x) for x in nb)
        hit1 += same1
        hit3 += same3
        if t != ["其他"]:
            classified += 1
            clf_hit1 += same1
            clf_hit3 += same3

        g = grams(r["错误原因"])
        u = set().union(*[grams(err[j]["错误原因"]) for j in top])
        if g:
            jac.append(len(g & u) / len(g | u))

    print(f"错例 {n} 条，能归类 {classified} 条（{classified/n:.1%}）")
    print(f"全部：top1 一致 {hit1/n:.3f}，top3 至少一条一致 {hit3/n:.3f}")
    if classified:
        print(f"仅可归类：top1 {clf_hit1/classified:.3f}，top3 {clf_hit3/classified:.3f}")
    print(f"原因文本 4-gram Jaccard 平均 {sum(jac)/len(jac):.3f}")
    print("类型分布:", dict(dist.most_common()))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "retrieval_only.json").write_text(json.dumps({
        "n": n, "classified": classified,
        "type_hit_top1": round(hit1 / n, 3), "type_hit_top3": round(hit3 / n, 3),
        "clf_hit_top1": round(clf_hit1 / classified, 3) if classified else None,
        "clf_hit_top3": round(clf_hit3 / classified, 3) if classified else None,
        "jaccard4": round(sum(jac) / len(jac), 3),
        "type_dist": dict(dist.most_common()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
