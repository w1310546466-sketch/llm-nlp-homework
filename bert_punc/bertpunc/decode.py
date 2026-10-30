"""长文本的窗口切分与重叠裁决。

这两件事是纯逻辑，不依赖 torch，所以单独拿出来测。训练脚本只负责前向。
"""


def make_windows(n_tokens, win=128, overlap=32):
    """把 [0, n_tokens) 切成若干窗口，相邻窗口重叠 overlap。

    最后一个窗口一定会覆盖到末尾，所以不会漏字。
    """
    if n_tokens <= 0:
        raise ValueError("长度必须为正")
    if overlap >= win:
        raise ValueError("重叠必须小于窗口")
    stride = win - overlap
    windows = []
    start = 0
    while True:
        end = min(start + win, n_tokens)
        windows.append((start, end))
        if end == n_tokens:
            break
        start += stride
    return windows


def merge(labels_by_window, windows, n_tokens):
    """重叠区按"离窗口边缘最远"裁决，保证每个位置只取一次预测。

    labels_by_window[i] 只包含窗口 i 内真实 token 的标签（不含 [CLS]）。
    """
    best = [(-1, 0)] * n_tokens
    for (start, end), row in zip(windows, labels_by_window):
        if len(row) != end - start:
            raise ValueError(f"窗口 {start}-{end} 的标签数 {len(row)} 与窗口长度不符")
        for k, lab in enumerate(row):
            pos = start + k
            margin = min(pos - start, end - 1 - pos)
            if margin > best[pos][0]:
                best[pos] = (margin, int(lab))
    missing = [i for i, (m, _) in enumerate(best) if m < 0]
    if missing:
        raise ValueError(f"有 {len(missing)} 个位置未被覆盖，首个是 {missing[0]}")
    return [lab for _, lab in best]


def render(chars, labels, id2label):
    """把逐字标签还原成带标点的文本。"""
    out = []
    for ch, lab in zip(chars, labels):
        out.append(ch)
        if lab != 0:
            out.append(id2label[lab])
    return "".join(out)
