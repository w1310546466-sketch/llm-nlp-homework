"""Corpus loading and cleaning.

The course corpus ships as word-segmented text with POS tags. Before counting
anything those tags have to go, otherwise the model spends its probability
mass on annotation conventions instead of language.
"""

import re
from pathlib import Path

CJK = re.compile(r"[\u4e00-\u9fff]")
# Keep hanzi and Chinese punctuation only. Latin fragments, pipes and the
# invisible word-joiner that shows up in exported slides are noise here.
KEEP = re.compile(r"[^\u4e00-\u9fff，。！？；：、“”‘’（）《》〈〉…—·]")

MIN_LEN = 8
MIN_CJK_RATIO = 0.45


def clean_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if line.startswith("--- slide"):
        return ""
    if re.fullmatch(r"\d+\s*/\s*\d+", line):  # page footer like "123 / 150"
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
    """Write a cleaned, deduplicated corpus and return basic stats."""
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
