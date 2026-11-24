# -*- coding: utf-8 -*-
"""L4 第 135 页：一个能验收的最小系统——标准文档译文审校助手。

用户输入：英文原文 + 待审译文
允许材料：1656 条参考译文（检索相似条款作为术语与句式参照）
产物：问题清单（类型 / 依据 / 建议）
失败出口：定位不到时说"疑似"并指出缺什么信息，不硬编

基线：直接问模型"这段译文有什么问题"
加一项机制：检索 3 条同类条款的原文-译文对，作为参照再问
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, r"D:\DSH\大语言模型\work\l4_protocols")
from run_protocols import BM25  # noqa: E402

MODEL = r"D:\DSH\大语言模型\models\qwen2.5-1.5b-instruct"
BASE = Path(r"D:\DSH\大语言模型")
WORK = Path(r"D:\DSH\大语言模型\work\l4_system")
OUT = WORK / "results"

SYS = "你是标准文档的译文审校。只指出译文相对原文的具体问题，不要评价文风。"

BASE_PROMPT = """原文：
{src}

译文：
{trans}

指出译文中的问题。按格式回答：
问题类型：<否定/情态/指代/条件/范围/术语/漏译/语序/一致性 之一>
依据：<原文的哪个词或结构>
建议：<怎么改>"""

EVIDENCE_PROMPT = """参考下面几条同一标准的条款，它们的原文与标准译法如下：

{evidence}

原文：
{src}

译文：
{trans}

对照参考条款的术语与句式，指出译文中的问题。按格式回答：
问题类型：<否定/情态/指代/条件/范围/术语/漏译/语序/一致性 之一>
依据：<原文的哪个词或结构>
建议：<怎么改>"""


def build_data():
    ok = json.loads((BASE / "correct_translations.json").read_text(encoding="utf-8"))
    err = json.loads((BASE / "translation_errors.json").read_text(encoding="utf-8"))
    WORK.mkdir(parents=True, exist_ok=True)
    return ok, err


def evidence_for(src, ok, bm, k=3):
    scores = bm.score(src)
    order = np.argsort(-scores)
    picked = []
    for i in order:
        item = ok[int(i)]
        if item["原文"].strip() == src.strip():
            continue          # 排除同一句，避免直接给答案
        picked.append(item)
        if len(picked) >= k:
            break
    return "\n\n".join(f"原文：{p['原文'][:220]}\n译文：{p['译文'][:160]}" for p in picked)


def parse_answer(text):
    """把回答拆成三段，用来算结构完整率。"""
    parts = {}
    for key in ("问题类型", "依据", "建议"):
        m = re.search(key + r"[：:]\s*(.+)", text)
        parts[key] = m.group(1).strip()[:120] if m else None
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    args = ap.parse_args()

    ok, err = build_data()
    bm = BM25([r["原文"] for r in ok])
    items = err[:args.limit]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def ask(prompt, max_new=160):
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=1700).to("cuda")
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()

    for i, it in enumerate(items):
        src, bad, reason = it["原始英文句子"], it["错误翻译"], it["错误原因"]

        t1 = time.time()
        a_base = ask(BASE_PROMPT.format(src=src[:1500], trans=bad[:800]))
        t_base = time.time() - t1

        ev = evidence_for(src, ok, bm, 3)
        t2 = time.time()
        a_ev = ask(EVIDENCE_PROMPT.format(evidence=ev, src=src[:1200], trans=bad[:700]))
        t_ev = time.time() - t2

        rows.append({
            "id": i, "reason": reason[:200],
            "baseline": a_base, "baseline_sec": round(t_base, 2),
            "evidence": a_ev, "evidence_sec": round(t_ev, 2),
            "evidence_text": ev[:400],
            "base_parts": parse_answer(a_base),
            "ev_parts": parse_answer(a_ev),
        })
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{len(items)}", flush=True)

    def struct_rate(key):
        return sum(1 for r in rows if all(r[key].get(k) for k in ("问题类型", "依据", "建议"))) / len(rows)

    summary = {
        "items": len(rows),
        "baseline_struct": round(struct_rate("base_parts"), 3),
        "evidence_struct": round(struct_rate("ev_parts"), 3),
        "baseline_sec": round(sum(r["baseline_sec"] for r in rows) / len(rows), 2),
        "evidence_sec": round(sum(r["evidence_sec"] for r in rows) / len(rows), 2),
        "total_seconds": round(time.time() - t0, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "traces.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
