# -*- coding: utf-8 -*-
"""下一项机制：把一次九分类拆成九次二分判断。

基线的问题不是找不到依据，而是判别崩塌——24 条全判成"漏译"。这里逐类问
"是否存在 X 类问题"，看类型分布会不会恢复，以及与参考原因的类型是否更接近。
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, r"D:\DSH\大语言模型\work\l4_protocols")
from retrieval_only import classify  # noqa: E402

MODEL = r"D:\DSH\大语言模型\models\qwen2.5-1.5b-instruct"
BASE = Path(r"D:\DSH\大语言模型")
OUT = Path(r"D:\DSH\大语言模型\work\l4_system\results")

TYPES = ["否定", "情态", "指代", "条件", "范围", "术语", "漏译", "语序", "一致性"]
HINT = {
    "否定": "no / not / unless / except 一类否定或例外被译错、译丢",
    "情态": "shall / must / may / should 的语气变了",
    "指代": "latter / such / it 指代的对象错了或不清",
    "条件": "if / when 的限定挂错了成分",
    "范围": "适用对象或范围被扩大、缩小",
    "术语": "术语译名不对或前后不统一",
    "漏译": "原文有的成分译文里没有",
    "语序": "句子结构重排后逻辑连接断了",
    "一致性": "同一个词前后译法不一致",
}

PROMPT = """原文：
{src}

译文：
{trans}

这段译文是否存在【{t}】这一类问题？判断标准：{hint}。
只回答"是"或"否"，然后给一句依据。"""


def main():
    err = json.loads((BASE / "translation_errors.json").read_text(encoding="utf-8"))
    items = err[:24]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def ask(prompt):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=1700).to("cuda")
        out = model.generate(**ids, max_new_tokens=40, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    rows, t0 = [], time.time()
    for i, it in enumerate(items):
        positives = []
        for t in TYPES:
            ans = ask(PROMPT.format(src=it["原始英文句子"][:1200],
                                    trans=it["错误翻译"][:600], t=t, hint=HINT[t]))
            head = ans[:6]
            if "是" in head and "否" not in head:
                positives.append(t)
        gold = classify(it["错误原因"])
        rows.append({"id": i, "gold": gold, "pred": positives,
                     "hit": bool(set(gold) & set(positives))})
        if (i + 1) % 6 == 0:
            print(f"  {i+1}/{len(items)}", flush=True)

    dist = Counter()
    for r in rows:
        for t in r["pred"]:
            dist[t] += 1
    empty = sum(1 for r in rows if not r["pred"])
    acc = sum(r["hit"] for r in rows) / len(rows)
    print("预测类型分布:", dict(dist.most_common()))
    print(f"一条都没判出问题: {empty}/{len(rows)}")
    print(f"与参考类型有交集: {acc:.3f}")

    (OUT / "binary_check.json").write_text(json.dumps({
        "rows": rows, "dist": dict(dist.most_common()),
        "empty": empty, "type_hit": round(acc, 3),
        "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
