"""单元测试。运行：python -m unittest discover -s tests -v"""

import math
import random
import unittest

from ngram.evaluate import perplexity, unseen_ratio
from ngram.generate import alphabet, sample
from ngram.model import BOS, EOS, AddK, Counts, KneserNey, MLE, StupidBackoff, build


TOY = ["甲乙丙丁", "甲乙戊己", "丙丁庚辛"]


class TestCounts(unittest.TestCase):
    def test_bigram_counts_include_boundaries(self):
        c = Counts(2)
        c.add("甲乙")
        seq_len = 1 + 2 + 1  # 一个 BOS 填充 + 两个字 + 一个 EOS
        self.assertEqual(c.ctx_total[1][()], seq_len)
        self.assertEqual(c.count[2][(BOS, "甲")], 1)
        self.assertEqual(c.count[2][("甲", "乙")], 1)
        self.assertEqual(c.count[2][("乙", EOS)], 1)

    def test_entries_count(self):
        c = Counts(3)
        for s in TOY:
            c.add(s)
        self.assertGreater(c.n_entries, 0)

    def test_bad_order(self):
        with self.assertRaises(ValueError):
            Counts(0)


class TestEstimators(unittest.TestCase):
    def setUp(self):
        self.counts = build(TOY, 2)

    def test_mle_hits_unseen_with_zero(self):
        m = MLE(self.counts)
        self.assertEqual(m.prob(("甲",), "戊"), 0.0)
        self.assertAlmostEqual(m.prob(("甲",), "乙"), 1.0)

    def test_addk_is_never_zero(self):
        m = AddK(self.counts, 0.1, vocab_size=8)
        self.assertGreater(m.prob(("甲",), "辛"), 0.0)

    def test_laplace_sums_to_one_over_observed_ctx(self):
        m = AddK(self.counts, 1.0, vocab_size=8)
        total = sum(m.prob(("甲",), w) for w in ["乙", "戊", "丙", "丁", "己", "庚", "辛", EOS])
        self.assertLess(abs(total - 1.0), 1e-9)

    def test_stupid_backoff_monotone_in_alpha(self):
        a = StupidBackoff(self.counts, 0.4).prob(("甲",), "戊")
        b = StupidBackoff(self.counts, 0.1).prob(("甲",), "戊")
        self.assertGreater(a, b)

    def test_kneser_ney_unigram_normalised(self):
        counts = build(TOY, 2)
        m = KneserNey(counts)
        # 归一化是对整个词表成立的，包含 EOS。
        words = [g[0] for g in counts.count[1] if g[0] != BOS]
        total = sum(m.prob((), w) for w in words)
        self.assertLess(abs(total - 1.0), 1e-6)

    def test_kneser_ney_prefers_combinable_word(self):
        # "乙" 前面出现过两种不同的字，"丁" 只跟在 "甲" 后面。
        counts = build(["甲乙", "丙乙", "甲丁"], 2)
        m = KneserNey(counts)
        # 给一个没见过的上下文，会回退到一元分布，
        # 这时组合自由度更高的字应当胜出。
        self.assertGreater(m.prob(("辛",), "乙"), m.prob(("辛",), "丁"))


class TestEvaluation(unittest.TestCase):
    def test_mle_perplexity_is_infinite(self):
        counts = build(TOY, 2)
        self.assertTrue(math.isinf(perplexity(MLE(counts), ["甲乙癸"], 2)))

    def test_perfect_model_gives_one(self):
        class Always:
            def prob(self, ctx, w):
                return 1.0
        self.assertAlmostEqual(perplexity(Always(), ["甲乙"], 1), 1.0)

    def test_unseen_ratio_bounds(self):
        counts = build(TOY, 2)
        r = unseen_ratio(counts, TOY, 2)
        self.assertGreaterEqual(r, 0.0)
        self.assertLessEqual(r, 1.0)


class TestGenerate(unittest.TestCase):
    def test_alphabet_excludes_boundaries(self):
        counts = build(TOY, 3)
        chars = alphabet(KneserNey(counts))
        self.assertNotIn(BOS, chars)
        self.assertNotIn(EOS, chars)
        self.assertIn("甲", chars)

    def test_length_is_seed_plus_n(self):
        counts = build(TOY, 3)
        out = sample(KneserNey(counts), "甲乙", 20, 3, random.Random(0))
        self.assertEqual(len(out), 22)

    def test_deterministic_with_same_seed(self):
        counts = build(TOY, 3)
        a = sample(KneserNey(counts), "甲", 15, 3, random.Random(7))
        b = sample(KneserNey(counts), "甲", 15, 3, random.Random(7))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
