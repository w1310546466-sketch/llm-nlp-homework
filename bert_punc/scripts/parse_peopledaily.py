"""把人民日报 1998 版语料解析成干净的字符级训练语料。

这批文件长这样：

    19980101-01-001-006 在 １９９８年 来临 之际 ， 我 十分 高兴 地 通过 ...
    19980101-01-001-007 １９９７年 ， 是 中国 发展 历史 上{shang5} 非常 重要 的 ...

所以：GB18030 编码，一行一句，句子编号在行首，词之间用空格分开，标点单独
成项，部分词后面带花括号注音。这里没有词性标注——带词性的是另一个版本——
所以清洗工作主要是去掉行首编号和花括号。

    python scripts/parse_peopledaily.py --src data/raw/*.txt --out data/corpus.txt
"""

import argparse
import glob
import re
from pathlib import Path

LINE_ID = re.compile(r"^(\d{8}-\d{2}-\d{3}-\d{3})\s+")
PINYIN = re.compile(r"\{[^}]*\}")
BRACKETS = re.compile(r"[\[\]{}]")
# 只保留汉字和中文标点。全角数字、拉丁字母片段，以及语料里表示"该字无法编码"
# 的 〓 占位符，都在这里被丢掉。
KEEP = re.compile(r"[^\u4e00-\u9fff，。！？；：、“”‘’（）《》…—]")
ENCODINGS = ("utf-8", "gb18030", "gbk")


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    # 三月份那个文件有几处坏字节，严格解码会失败，退回容错解码。
    text = data.decode("gb18030", errors="replace")
    print(f"  注意：{path.name} 有 {text.count(chr(0xFFFD))} 个替换字符")
    return text


def to_sentences(line: str):
    body = PINYIN.sub("", LINE_ID.sub("", line)).strip()
    if not body:
        return []
    text = BRACKETS.sub("", "".join(body.split()))
    text = KEEP.sub("", text)
    # 通常一行就是一句；偶尔一行里有两句，这里再切一次。
    return [s for s in re.split(r"(?<=[。！？])", text) if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="*", default=None)
    ap.add_argument("--out", default="data/corpus.txt")
    ap.add_argument("--max-chars", type=int, default=0, help="0 表示不限")
    ap.add_argument("--min-length", type=int, default=8)
    args = ap.parse_args()

    paths = []
    for pattern in (args.src or ["data/raw/*.txt"]):
        paths.extend(sorted(glob.glob(pattern)))

    seen, out_lines, chars = set(), [], 0
    for path in paths:
        for raw in read_text(Path(path)).splitlines():
            for sent in to_sentences(raw):
                if len(sent) < args.min_length or sent in seen:
                    continue
                seen.add(sent)
                out_lines.append(sent)
                chars += len(sent)
            if args.max_chars and chars >= args.max_chars:
                break
        if args.max_chars and chars >= args.max_chars:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"文件 {len(paths)} 个 | 句子 {len(out_lines)} 句 | 字数 {chars} -> {out}")


if __name__ == "__main__":
    main()
