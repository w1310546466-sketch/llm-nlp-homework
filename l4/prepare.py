# -*- coding: utf-8 -*-
"""L4 第 130 页实验：准备固定材料。

从课程给的三份数据里取：
  - 翻译题：正确译文集里长度适中的句子（原文 + 参考译文）
  - 错例题：错译集里的句子（原文 + 错误翻译 + 错误原因 + 正确翻译）

检索用的历史错误库：全部 504 条错例。
"""

import json
import random
import re
from pathlib import Path

BASE = Path(r"D:\DSH\大语言模型")
OUT = Path(r"D:\DSH\大语言模型\work\l4_protocols")
OUT.mkdir(parents=True, exist_ok=True)

random.seed(0)


def load(name):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def main():
    ok = load("correct_translations.json")
    err = load("translation_errors.json")
    pol = load("contextual_polish.json")

    # 翻译题：原文 30-80 词，译文 20-120 字，避免太长跑不动
    pool = [r for r in ok
            if 30 <= len(words(r["原文"])) <= 80 and 20 <= len(r["译文"]) <= 120]
    random.shuffle(pool)
    items = pool[:24]

    # 错例题：错误原因清楚的，长度适中
    epool = [r for r in err
             if 20 <= len(words(r["原始英文句子"])) <= 70 and len(r["错误原因"]) >= 25]
    random.shuffle(epool)
    eitems = epool[:24]

    # 检索库：全部错例，只留字段
    memory = [{"src": r["原始英文句子"], "bad": r["错误翻译"],
               "reason": r["错误原因"]} for r in err]

    (OUT / "data.json").write_text(json.dumps({
        "translate": items,
        "errors": eitems,
        "memory": memory,
        "polish_count": len(pol),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"翻译题 {len(items)} 条，错例题 {len(eitems)} 条，检索库 {len(memory)} 条")
    print("翻译题示例:", items[0]["原文"][:100])
    print("参考译文:", items[0]["译文"][:60])


if __name__ == "__main__":
    main()
