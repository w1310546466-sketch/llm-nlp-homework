# -*- coding: utf-8 -*-
"""L4 第 130 页：四种调用协议的对照实验。

协议：
  A 直接调用     一次翻译
  B 单条分解     先列术语与结构，再译
  C 多候选聚合   3 个候选 + 自选
  D 验证选择     3 个候选 + 从历史错误检索出的检查清单 + 自选

评价不追 BLEU：记录覆盖率（候选里有没有可用的）、选择损失（有却没选中）、
成本（调用次数与输出量）、以及错例题上的问题检出率。
"""

import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = r"D:\DSH\大语言模型\models\qwen2.5-1.5b-instruct"
WORK = Path(r"D:\DSH\大语言模型\work\l4_protocols")
OUT = WORK / "results"
MAX_NEW = 220
K = 3


# ------------------------------------------------------------------ 检索

class BM25:
    """词级 BM25，用来找相似的历史错例。不引第三方库。"""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [self.tok(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(len(d) for d in self.docs) / len(self.docs)
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {w: np.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    @staticmethod
    def tok(s):
        return re.findall(r"[a-z0-9]+", s.lower())

    def score(self, query):
        q = self.tok(query)
        scores = []
        for d in self.docs:
            tf = Counter(d)
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                idf = self.idf.get(w, 0.0)
                num = tf[w] * (self.k1 + 1)
                den = tf[w] + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
                s += idf * num / den
            scores.append(s)
        return np.array(scores)

    def top(self, query, k=3):
        s = self.score(query)
        return np.argsort(-s)[:k]


# ------------------------------------------------------------------ 生成

def make_chat(tok, model):
    @torch.no_grad()
    def chat(system, user, max_new=MAX_NEW, sample=False, seed=0):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=1800).to("cuda")
        torch.manual_seed(seed)
        kw = dict(max_new_tokens=max_new, do_sample=sample, pad_token_id=tok.eos_token_id)
        if sample:
            kw.update(temperature=0.8, top_p=0.9)
        out = model.generate(**ids, **kw)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return chat


def char_f1(pred, ref):
    """字符级 F1，只做诊断用，不是优化目标。"""
    p, r = Counter(pred), Counter(ref)
    common = sum((p & r).values())
    if common == 0:
        return 0.0
    prec = common / max(sum(p.values()), 1)
    rec = common / max(sum(r.values()), 1)
    return 2 * prec * rec / (prec + rec)


# ------------------------------------------------------------------ 协议

SYS_TRANS = "你是标准文档翻译，译文用规范中文，保留条款编号与术语，只输出译文。"


def proto_a(chat, src):
    out = chat(SYS_TRANS, f"把下面的英文译成中文：\n\n{src}", sample=False)
    return {"candidates": [out], "chosen": out, "calls": 1}


def proto_b(chat, src):
    plan = chat("你是术语与结构分析助手。",
                f"阅读下面的英文，列出关键术语、条款结构、以及翻译时容易出错的点。"
                f"只列要点，不要翻译。\n\n{src}", max_new=160, sample=False)
    out = chat(SYS_TRANS,
               f"参考下面的分析，翻译这段英文，只输出译文。\n\n分析：\n{plan}\n\n原文：\n{src}",
               sample=False)
    return {"candidates": [out], "chosen": out, "calls": 2, "plan": plan}


def _pick(chat, src, cands, extra=""):
    listing = "\n\n".join(f"候选{i+1}：\n{c}" for i, c in enumerate(cands))
    prompt = (f"原文：\n{src}\n\n{listing}\n\n{extra}"
              f"选出最符合原文、术语准确、中文通顺的一条，只回答候选编号的数字。")
    ans = chat("你是严格的译文评审。", prompt, max_new=10, sample=False)
    m = re.search(r"[1-9]", ans)
    idx = int(m.group()) - 1 if m else 0
    return idx if 0 <= idx < len(cands) else 0


def proto_c(chat, src):
    cands = [chat(SYS_TRANS, f"把下面的英文译成中文：\n\n{src}", sample=True, seed=i)
             for i in range(K)]
    idx = _pick(chat, src, cands)
    return {"candidates": cands, "chosen": cands[idx], "calls": K + 1, "picked": idx}


GLOBAL_CHECKLIST = [
    "否定与例外：no / not / unless / except 是否全部译出，有没有把否定译成肯定",
    "情态：shall / must / may / should 的语气是否对应（必须 / 可以 / 应当）",
    "指代：latter / former / such / it 指的是哪个成分，主语是否明确",
    "条件与范围：if / when / where / applies to 的限定是否挂在正确的成分上",
    "术语：术语是否与原文一致，同一个词前后是否统一",
    "括注与编号：括号里的条款号、图号、缩写是否漏译或错位",
    "结构：英文长句拆分后，中文有没有丢掉逻辑连接",
    "增译：有没有加入原文没有的信息",
]


def proto_d(chat, src):
    """三候选 + 全局检查清单核对后选择。

    单条检索在 retrieval_only.py 里测过：按源句检索出来的历史错误，其原因与
    当前错误的类型只有 0.217 一致率，清单不相关，所以这里改用全局清单。
    """
    cands = [chat(SYS_TRANS, f"把下面的英文译成中文：\n\n{src}", sample=True, seed=100 + i)
             for i in range(K)]
    checklist = "\n".join(f"- {c}" for c in GLOBAL_CHECKLIST)
    idx = _pick(chat, src, cands,
                extra=f"逐条核对下面的检查项，再决定哪条最合格：\n{checklist}\n\n")
    return {"candidates": cands, "chosen": cands[idx], "calls": K + 1,
            "picked": idx, "checklist": GLOBAL_CHECKLIST}


def detect(chat, src, bad):
    ans = chat("你是标准文档审校。",
               f"原文：\n{src}\n\n译文：\n{bad}\n\n这段译文有什么问题？指出具体错误，一句话。",
               max_new=90, sample=False)
    return ans


# ------------------------------------------------------------------ 主流程

def main():
    data = json.loads((WORK / "data.json").read_text(encoding="utf-8"))
    memory = data["memory"]
    bm25 = BM25([m["src"] for m in memory])

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda()
    model.eval()
    chat = make_chat(tok, model)
    print("模型就绪", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary = {}
    traces = []

    protos = {"A": lambda s: proto_a(chat, s),
              "B": lambda s: proto_b(chat, s),
              "C": lambda s: proto_c(chat, s),
              "D": lambda s: proto_d(chat, s)}

    for name, fn in protos.items():
        rows = []
        for i, item in enumerate(data["translate"]):
            r = fn(item["原文"])
            f1s = [char_f1(c, item["译文"]) for c in r["candidates"]]
            chosen_f1 = char_f1(r["chosen"], item["译文"])
            best = max(f1s)
            rows.append({
                "id": i, "chars": len(item["原文"]),
                "coverage": best >= 0.55,
                "selection_loss": round(best - chosen_f1, 3),
                "chosen_f1": round(chosen_f1, 3),
                "calls": r["calls"],
                "picked": r.get("picked"),
                "chosen": r["chosen"][:300],
                "candidates": [c[:300] for c in r["candidates"]],
                "plan": r.get("plan", "")[:300],
                "checklist": r.get("checklist"),
            })
            traces.append({"protocol": name, "id": i, **rows[-1]})
            if (i + 1) % 8 == 0:
                print(f"  {name} {i+1}/{len(data['translate'])}", flush=True)
        n = len(rows)
        summary[name] = {
            "coverage": round(sum(r["coverage"] for r in rows) / n, 3),
            "avg_selection_loss": round(sum(r["selection_loss"] for r in rows) / n, 3),
            "avg_f1": round(sum(r["chosen_f1"] for r in rows) / n, 3),
            "avg_calls": round(sum(r["calls"] for r in rows) / n, 2),
        }
        print(name, summary[name], flush=True)

    # 错例题：能否指出问题
    det = []
    for i, item in enumerate(data["errors"]):
        ans = detect(chat, item["原始英文句子"], item["错误翻译"])
        reason = item["错误原因"]
        key = [w for w in re.findall(r"[\u4e00-\u9fff]{2,4}", reason)
               if w not in ("原文", "译文", "现译", "这里", "需要", "应当")]
        hit = sum(1 for w in set(key) if w in ans)
        det.append({"id": i, "hit": hit, "target": min(len(set(key)), 5),
                    "answer": ans[:200], "reason": reason[:200]})
        if (i + 1) % 8 == 0:
            print(f"  检出 {i+1}/{len(data['errors'])}", flush=True)
    detect_rate = sum(1 for d in det if d["hit"] >= 2) / len(det)
    print("检出率", round(detect_rate, 3), flush=True)

    (OUT / "summary.json").write_text(json.dumps(
        {"protocols": summary, "detect_rate": round(detect_rate, 3),
         "seconds": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "traces.json").write_text(json.dumps(
        {"translate": traces, "detect": det}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
