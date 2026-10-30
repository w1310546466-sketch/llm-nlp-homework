# -*- coding: utf-8 -*-
"""训练：BERT 逐字预测标点。

    python train.py --corpus data/corpus.txt --epochs 2 --batch 32

默认超参对应报告里的那次运行（40k 句、2 轮、batch 32、lr 5e-5）。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from bertpunc.data import (ID2LABEL, LABEL2ID, LABELS, PuncDataset,
                           load_sentences, split_train_valid)

MODEL_NAME = os.environ.get("BERT_MODEL", "bert-base-chinese")


def evaluate(model, loader, device):
    """只统计标点位置的 P/R/F1：'O' 占九成以上，逐字准确率没有意义。"""
    model.eval()
    tp = fp = fn = 0
    per_class = {c: [0, 0, 0] for c in LABELS[1:]}
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
            preds = logits.argmax(-1).cpu().numpy()
            gold = batch["labels"].cpu().numpy()
            for p_row, g_row in zip(preds, gold):
                for p, g in zip(p_row, g_row):
                    if g == -100:
                        continue
                    if g != 0:
                        if p == g:
                            tp += 1
                            per_class[ID2LABEL[g]][0] += 1
                        else:
                            fn += 1
                            per_class[ID2LABEL[g]][2] += 1
                    if p != 0 and p != g:
                        fp += 1
                        if ID2LABEL[p] in per_class:
                            per_class[ID2LABEL[p]][1] += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    detail = {}
    for c, (t, f, n) in per_class.items():
        p = t / (t + f) if t + f else 0.0
        r = t / (t + n) if t + n else 0.0
        detail[c] = {"precision": round(p, 4), "recall": round(r, 4),
                     "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
                     "support": t + n}
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "per_class": detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--train-limit", type=int, default=40000)
    ap.add_argument("--valid-limit", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    sentences = load_sentences(args.corpus)
    train_s, valid_s = split_train_valid(sentences, args.train_limit, args.valid_limit)
    print(f"训练 {len(train_s)} 句，验证 {len(valid_s)} 句")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID).to(device)

    train_loader = DataLoader(PuncDataset(train_s, tokenizer, args.max_len),
                              batch_size=args.batch, shuffle=True)
    valid_loader = DataLoader(PuncDataset(valid_s, tokenizer, args.max_len),
                              batch_size=args.batch)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(train_loader) * args.epochs
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 1.0, total_iters=total)

    t0 = time.time()
    step, history = 0, []
    for ep in range(args.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += out.loss.item()
            step += 1
            if step % 200 == 0:
                print(f"  第 {ep+1} 轮 step {step}/{total} loss {running/200:.4f} "
                      f"{time.time()-t0:.0f}s", flush=True)
                running = 0.0
        metrics = evaluate(model, valid_loader, device)
        print(f"第 {ep+1} 轮验证：P {metrics['precision']} R {metrics['recall']} "
              f"F1 {metrics['f1']}", flush=True)
        history.append(metrics)

    train_sec = time.time() - t0
    out_dir = Path(args.out)
    (out_dir / "model").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "model")

    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "train_sentences": len(train_loader.dataset),
        "valid_sentences": len(valid_loader.dataset),
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "train_seconds": round(train_sec, 1),
        "history": history,
        "final": history[-1] if history else None,
    }
    (out_dir / "train_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("train_seconds", "final")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
