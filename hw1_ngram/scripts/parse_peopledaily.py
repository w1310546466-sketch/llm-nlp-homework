"""Parse the People's Daily 1998 release into a clean character corpus.

The files in this release look like:

    19980101-01-001-006 在 １９９８年 来临 之际 ， 我 十分 高兴 地 通过 ...
    19980101-01-001-007 １９９７年 ， 是 中国 发展 历史 上{shang5} 非常 重要 的 ...

So: GB18030 text, one sentence per line, the sentence id at the start of the
line, words separated by spaces, punctuation as its own token, and pinyin in
braces attached to some words. Nothing here is a POS tag -- the tags live in the
other release -- so cleaning is mostly dropping the ids and the braces.

    python scripts/parse_peopledaily.py --src data/raw/*.txt --out data/corpus.txt
"""

import argparse
import glob
import re
from pathlib import Path

LINE_ID = re.compile(r"^(\d{8}-\d{2}-\d{3}-\d{3})\s+")
PINYIN = re.compile(r"\{[^}]*\}")
BRACKETS = re.compile(r"[\[\]{}]")
# Keep hanzi and Chinese punctuation. Full-width digits, latin fragments and the
# corpus' `〓` placeholder for unencodable characters all land in the bin here.
KEEP = re.compile(r"[^\u4e00-\u9fff，。！？；：、“”‘’（）《》…—]")
ENCODINGS = ("utf-8", "gb18030", "gbk")


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    # One of the monthly files has a few undecodable bytes; see hw2's parser.
    text = data.decode("gb18030", errors="replace")
    print(f"  note: {path.name} decoded with {text.count(chr(0xFFFD))} replacement chars")
    return text


def to_sentences(line: str):
    body = PINYIN.sub("", LINE_ID.sub("", line)).strip()
    if not body:
        return []
    text = BRACKETS.sub("", "".join(body.split()))
    text = KEEP.sub("", text)
    # One line is normally one sentence; split anyway because a few lines carry two.
    return [s for s in re.split(r"(?<=[。！？])", text) if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="*", default=None)
    ap.add_argument("--out", default="data/corpus.txt")
    ap.add_argument("--max-chars", type=int, default=0, help="0 means everything")
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
    print(f"files {len(paths)} | sentences {len(out_lines)} | chars {chars} -> {out}")


if __name__ == "__main__":
    main()
