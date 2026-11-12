# -*- coding: utf-8 -*-
"""把四种协议的轨迹整理成报告用的表。

失败分三类，对应课件那三种：
  1. 没有正确候选   —— 生成端问题
  2. 有正确候选却选错 —— 选择端问题
  3. 选中但改坏      —— 修订端问题
"""

import json
from collections import Counter
from pathlib import Path

OUT = Path(r"D:\DSH\大语言模型\work\l4_protocols\results")
THRESH = 0.55


def load():
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    t = json.loads((OUT / "traces.json").read_text(encoding="utf-8"))
    return s, t


def main():
    s, t = load()
    rows = t["translate"]
    by_proto = {}
    for r in rows:
        by_proto.setdefault(r["protocol"], []).append(r)

    print("协议 | 覆盖 | 选择损失 | 平均F1 | 平均调用")
    for p in "ABCD":
        rs = by_proto.get(p, [])
        if not rs:
            continue
        cov = sum(r["coverage"] for r in rs) / len(rs)
        loss = sum(r["selection_loss"] for r in rs) / len(rs)
        f1 = sum(r["chosen_f1"] for r in rs) / len(rs)
        calls = sum(r["calls"] for r in rs) / len(rs)
        print(f"{p} | {cov:.3f} | {loss:.3f} | {f1:.3f} | {calls:.2f}")

    print("\n失败分类")
    for p in "ABCD":
        rs = by_proto.get(p, [])
        if not rs:
            continue
        no_cand = sum(1 for r in rs if not r["coverage"])
        sel_err = sum(1 for r in rs if r["coverage"] and r["chosen_f1"] < THRESH)
        ok = sum(1 for r in rs if r["chosen_f1"] >= THRESH)
        print(f"{p}: 无正确候选 {no_cand} / 有选错 {sel_err} / 通过 {ok}")

    print("\n错例检出率:", s.get("detect_rate"))
    det = t["detect"]
    good = [d for d in det if d["hit"] >= 2]
    print("检出样例:")
    for d in good[:3]:
        print(" 答案:", d["answer"][:80])
        print(" 原因:", d["reason"][:80])


if __name__ == "__main__":
    main()
