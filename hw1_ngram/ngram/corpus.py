"""语料读取与清洗。

课程给的语料是分词并标注了词性的文本。计数之前必须把标注去掉，否则模型
会把概率质量花在标注规范上，而不是语言本身。
"""

import re
from pathlib import Path

CJK = re.compile(r"[\u4e00-\u9fff]")
# 只保留汉字和中文标点。拉丁字母、竖线，以及导出幻灯片时带的不可见连接符
# 在这里都算噪声。
KEEP = re.compile(r"[^\u4e00-\u9fff，。！？；：、“”‘’（）《》〈〉…—·]")

MIN_LEN = 8
MIN_CJK_RATIO = 0.45


def clean_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if line.startswith("--- slide"):
        return ""
    if re.fullmatch(r"\d+\s*/\s*\d+", line):  # 页脚形如 “123 / 150”
        return ""
    if "http://" in line or "https://" in line:
        return ""
    line = re.sub(r"\s+", " ", line)
    return KEEP.sub("", line)


def is_sentence(line: str, min_len: int = MIN_LEN) -> bool:
    if len(line) < min_len:
        return False
    if len(CJK.findall(line)) / len(line) < MIN_CJK_RATIO:
        return False
    return bool(re.search(r"[，。；：？！]", line))


def sentences_from_file(path: Path):
    seen = set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = clean_line(raw)
            if not is_sentence(line):
                continue
            for seg in re.split(r"(?<=[。！？])", line):
                seg = seg.strip()
                if not is_sentence(seg) or seg in seen:
                    continue
                seen.add(seg)
                yield seg


def build_corpus(raw_files, out_path: Path) -> dict:
    """写出清洗并去重后的语料，返回基本统计。"""
    seen = set()
    kept = []
    for path in raw_files:
        for sent in sentences_from_file(Path(path)):
            if sent in seen:
                continue
            seen.add(sent)
            kept.append(sent)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(kept), encoding="utf-8")
    text = "\n".join(kept)
    return {
        "sentences": len(kept),
        "chars": len(text),
        "cjk_chars": len(CJK.findall(text)),
    }


def load(path: Path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
