# -*- coding: utf-8 -*-
"""在长篇古文上做断句，并比较批处理与串行解码的代价。

    python eval_guji.py --model results/model --text data/古文.txt

两件事：零样本迁移的标点质量，以及 batch 1 / 8 / 32 的真实耗时。
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from bertpunc.data import ID2LABEL, read_text, split_chars
from bertpunc.decode import make_windows, merge, render


def decode(model, tokenizer, ids, device, batch_size, win, overlap):
    windows = make_windows(len(ids), win, overlap)
    rows, calls = [], 0
    t0 = time.time()
    pad = tokenizer.pad_token_id
    model.eval()
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            part = windows[i:i + batch_size]
            seqs = [tokenizer.build_inputs_with_special_tokens(ids[s:e]) for s, e in part]
            maxlen = max(len(x) for x in seqs)
            input_ids = torch.tensor([x + [pad] * (maxlen - len(x)) for x in seqs], device=device)
            attn = torch.tensor([[1] * len(x) + [0] * (maxlen - len(x)) for x in seqs], device=device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits.argmax(-1).cpu().numpy()
            calls += 1
            for row in logits:
                rows.append(row[1:])      # 去掉 [CLS]
            # 这里只取每个窗口的有效长度，交给 merge 校验
    rows = [row[:e - s] for (s, e), row in zip(windows, rows)]
    labels = merge(rows, windows, len(ids))
    return labels, {"windows": len(windows), "forward_calls": calls,
                    "seconds": round(time.time() - t0, 3), "batch_size": batch_size}


def naive_decode(model, tokenizer, ids, device, chunk=400, keep=300):
    """课件批评的做法：每次 400 字，只取前 300 字，退回句号作为下一起点。"""
    calls = 0
    t0 = time.time()
    model.eval()
    pos = 0
    with torch.no_grad():
        while pos < len(ids):
            seg = ids[pos:pos + chunk]
            input_ids = torch.tensor([tokenizer.build_inputs_with_special_tokens(seg)], device=device)
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
            calls += 1
            pos += keep
    return {"forward_calls": calls, "seconds": round(time.time() - t0, 3)}


def prf(gold, pred):
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        if g != 0 and p == g:
            tp += 1
        elif g != 0:
            fn += 1
            if p != 0:
                fp += 1
        elif p != 0:
            fp += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            "support": tp + fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/model")
    ap.add_argument("--text", default="data/古文.txt")
    ap.add_argument("--chars", type=int, default=1500)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--out", default="results/guji_results.json")
    args = ap.parse_args()

    # [CLS] 和 [SEP] 也占位置，窗口最多 510 个真实 token
    args.win = min(args.win, 510)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device)

    text = read_text(Path(args.text))[:args.chars]
    chars, gold = split_chars(text)
    # 逐字转 id，避免整段文本触发 tokenizer 的长度告警
    ids = [tokenizer.convert_tokens_to_ids(c) for c in chars]
    if len(ids) != len(chars):
        raise SystemExit(f"token 数 {len(ids)} 与字数 {len(chars)} 不一致，需要子词对齐")

    result = {"device": str(device), "chars": len(chars), "win": args.win, "overlap": args.overlap,
              "gold_punct": sum(1 for g in gold if g != 0)}

    for bs in (1, 8, 32):
        labels, stats = decode(model, tokenizer, ids, device, bs, args.win, args.overlap)
        if bs == 1:
            result["zero_shot"] = prf(gold, labels)
            result["sample_gold"] = render(chars[:200], gold[:200], ID2LABEL)
            result["sample_pred"] = render(chars[:200], labels[:200], ID2LABEL)
        result[f"batch_{bs}"] = stats
        print(f"batch={bs:>2} windows={stats['windows']} forward={stats['forward_calls']} "
              f"{stats['seconds']}s", flush=True)

    result["naive"] = naive_decode(model, tokenizer, ids, device)
    print("串行做法：", result["naive"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["zero_shot"], ensure_ascii=False))


if __name__ == "__main__":
    main()
