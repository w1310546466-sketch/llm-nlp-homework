"""从人民日报 1998 语料构造带主题标签的标题集。

语料本身没有主题标注，而作业要求"自选标签"，所以标签必须自己定义。这里取
每篇文章的第一行当标题，按关键词命中数归到得分最高的主题；一条都没命中、
或者两类打平的，直接丢掉。关键词表刻意写得很短，方便质疑，也方便改。

文章按编号前缀 `YYYYMMDD-版-文章` 分组，同一篇的句子不会被打散。

    python scripts/parse_peopledaily.py --src data/raw/*.txt --out data/labeled.tsv
"""

import argparse
import glob
import random
import re
from collections import Counter
from pathlib import Path

LINE_ID = re.compile(r"^(\d{8}-\d{2}-\d{3})-(\d{3})\s+")
PINYIN = re.compile(r"\{[^}]*\}")
BRACKETS = re.compile(r"[\[\]{}]")
ENCODINGS = ("utf-8", "gb18030", "gbk")

TOPICS = {
    "体育": ["球", "队", "赛", "冠军", "奥运", "教练", "运动员", "联赛", "足球", "篮球", "棋", "射击"],
    "金融": ["股", "银行", "金融", "证券", "保险", "利率", "汇率", "投资", "货币", "债", "税", "指数", "央行"],
    "教育": ["学校", "大学", "教育", "学生", "教师", "高考", "招生", "考试", "校园", "中学", "小学"],
    "文化": ["文化", "艺术", "电影", "音乐", "演出", "展览", "文物", "文学", "图书", "戏剧", "美术"],
    "科技": ["科技", "计算机", "网络", "信息", "卫星", "科研", "技术", "科学", "发明", "电子"],
    "医疗": ["医院", "病人", "医疗", "药品", "疾病", "卫生", "医生", "治疗", "健康", "防疫"],
    "农业": ["农业", "农民", "农村", "粮食", "种植", "水稻", "小麦", "牲畜", "丰收", "耕地"],
    "法律": ["法院", "法律", "案件", "判决", "律师", "犯罪", "起诉", "检察", "立法", "条例"],
}


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    # 某个文件的 4.2M 偏移处有几处坏字节，两百多万字里替换掉几个不影响结论。
    text = data.decode("gb18030", errors="replace")
    bad = text.count("\ufffd")
    print(f"  注意：{path.name} 有 {bad} 个替换字符")
    return text


def clean(line: str) -> str:
    body = PINYIN.sub("", LINE_ID.sub("", line)).strip()
    return BRACKETS.sub("", "".join(body.split()))


def iter_articles(paths):
    """逐篇产出 (标题, 正文)，同一篇文章的句子聚在一起。"""
    for path in paths:
        current_key, headline, body = None, "", []
        for raw in read_text(Path(path)).splitlines():
            m = LINE_ID.match(raw)
            if not m:
                continue
            key = m.group(1)
            text = clean(raw)
            if key != current_key:
                if current_key is not None and headline:
                    yield headline, "".join(body)
                current_key, headline, body = key, text, []
            elif not headline:
                headline = text
            else:
                body.append(text)
        if current_key is not None and headline:
            yield headline, "".join(body)


def assign(text: str):
    scores = {t: sum(1 for kw in kws if kw in text) for t, kws in TOPICS.items()}
    best = max(scores.values())
    if best == 0:
        return None
    winners = [t for t, s in scores.items() if s == best]
    return winners[0] if len(winners) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="*", default=None)
    ap.add_argument("--out", default="data/labeled.tsv")
    ap.add_argument("--body-out", default="", help="可选：另存一份正文供训练词向量")
    ap.add_argument("--per-label", type=int, default=700)
    ap.add_argument("--max-body-chars", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = []
    for pattern in (args.src or ["data/raw/*.txt"]):
        paths.extend(sorted(glob.glob(pattern)))

    buckets = {t: [] for t in TOPICS}
    body_chars, body_lines = 0, []
    for headline, body in iter_articles(paths):
        if len(headline) < 6 or len(headline) > 40:
            continue
        label = assign(headline)
        if label is not None:
            buckets[label].append(headline)
        if args.body_out and body_chars < args.max_body_chars and len(body) > 20:
            body_lines.append(body)
            body_chars += len(body)

    rng = random.Random(args.seed)
    rows = []
    for label, items in buckets.items():
        rng.shuffle(items)
        for text in items[:args.per_label]:
            rows.append(f"{label}\t{text}")
    rng.shuffle(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows), encoding="utf-8")

    print(f"保留标题 {len(rows)} 条")
    print("每类实际条数：", {k: min(len(v), args.per_label) for k, v in buckets.items()})
    print("各类可用条数：", {k: len(v) for k, v in buckets.items()})

    if args.body_out:
        Path(args.body_out).write_text("\n".join(body_lines), encoding="utf-8")
        print(f"正文 {len(body_lines)} 段 / {body_chars} 字 -> {args.body_out}")


if __name__ == "__main__":
    main()
