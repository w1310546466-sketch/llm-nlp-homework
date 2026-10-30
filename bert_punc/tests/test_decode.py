"""重叠区的裁决：位置越靠窗口中间，预测越可信。

左窗口 (0,6) 与右窗口 (4,10) 重叠在 [4,6)：
- 位置 4 在左窗口里离右边界 1，在右窗口里贴着左边界 0 → 取左窗口的值
- 位置 5 在左窗口里贴边界，在右窗口里离边界 1 → 取右窗口的值
"""

import unittest

from bertpunc.decode import make_windows, merge, render


class TestWindows(unittest.TestCase):
    def test_covers_every_position(self):
        n = 1000
        ws = make_windows(n, win=128, overlap=32)
        covered = set()
        for s, e in ws:
            covered.update(range(s, e))
        self.assertEqual(covered, set(range(n)))

    def test_no_window_exceeds_limit(self):
        for s, e in make_windows(1000, 128, 32):
            self.assertLessEqual(e - s, 128)

    def test_stride_is_win_minus_overlap(self):
        ws = make_windows(1000, 128, 32)
        for (s1, _), (s2, _) in zip(ws, ws[1:]):
            self.assertEqual(s2 - s1, 96)

    def test_short_text_single_window(self):
        self.assertEqual(make_windows(10, 128, 32), [(0, 10)])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            make_windows(0)

    def test_overlap_too_large_raises(self):
        with self.assertRaises(ValueError):
            make_windows(100, win=10, overlap=10)


class TestMerge(unittest.TestCase):
    def test_center_wins_in_overlap(self):
        windows = [(0, 6), (4, 10)]
        left = [1, 1, 1, 1, 9, 9]
        right = [2, 2, 2, 2, 2, 2]
        merged = merge([left, right], windows, 10)
        self.assertEqual(merged[4], 9)   # 左窗口里更靠中间
        self.assertEqual(merged[5], 2)   # 右窗口里更靠中间
        self.assertEqual(len(merged), 10)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            merge([[1, 2, 3]], [(0, 5)], 5)

    def test_uncovered_raises(self):
        with self.assertRaises(ValueError):
            merge([[1, 2]], [(0, 2)], 5)


class TestRender(unittest.TestCase):
    def test_inserts_labels_after_chars(self):
        out = render(["甲", "乙", "丙"], [0, 2, 0], {0: "O", 2: "。"})
        self.assertEqual(out, "甲乙。丙")

    def test_all_zero_returns_plain_text(self):
        self.assertEqual(render(["甲", "乙"], [0, 0], {0: "O"}), "甲乙")


if __name__ == "__main__":
    unittest.main()
