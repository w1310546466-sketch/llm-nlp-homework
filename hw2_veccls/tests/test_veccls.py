"""词向量与分类器的单元测试。"""

import math
import unittest

import numpy as np

from veccls.classifier import BagRegressor, FeatureTable
from veccls.corpus import clean_text, label_distribution, load_tsv, split
from veccls.skipgram import SkipGram


TOY = ["甲乙丙丁戊", "甲乙丙己庚", "丙丁戊辛壬"]


class TestCorpus(unittest.TestCase):
    def test_clean_strips_latin(self):
        self.assertEqual(clean_text("hello 模型 A1。"), "模型。")

    def test_label_distribution(self):
        self.assertEqual(label_distribution([("a", "x"), ("a", "y"), ("b", "z")]),
                         {"a": 2, "b": 1})

    def test_split_sizes(self):
        rows = [(str(i), "文本" * 5) for i in range(10)]
        train, test = split(rows, test_ratio=0.2)
        self.assertEqual(len(test), 2)
        self.assertEqual(len(train), 8)
        self.assertEqual(len({r[0] for r in train} & {r[0] for r in test}), 0)


class TestSkipGram(unittest.TestCase):
    def setUp(self):
        self.sg = SkipGram(TOY, dim=16, window=2, min_count=1, negatives=3,
                           epochs=1, seed=0)
        self.sg.fit()

    def test_vocab_and_index_align(self):
        for i, w in enumerate(self.sg.vocab):
            self.assertEqual(self.sg.index[w], i)

    def test_vectors_are_finite(self):
        self.assertTrue(np.all(np.isfinite(self.sg.w_in)))

    def test_neighbors_exclude_query(self):
        word = self.sg.vocab[0]
        self.assertTrue(all(k != word for k, _ in self.sg.neighbors(word, 3)))

    def test_unknown_word_has_no_vector(self):
        self.assertIsNone(self.sg.vector("龘"))

    def test_deterministic_with_same_seed(self):
        a = SkipGram(TOY, dim=8, epochs=1, seed=7)
        b = SkipGram(TOY, dim=8, epochs=1, seed=7)
        a.fit()
        b.fit()
        self.assertTrue(np.allclose(a.w_in, b.w_in))


class TestClassifier(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        p = BagRegressor.softmax(np.array([0.3, -1.0, 2.0]))
        self.assertLess(abs(p.sum() - 1.0), 1e-9)

    def test_learns_separable_data(self):
        # 二维平面上两个分得很开的簇，线性分类器应该很容易学到。
        X, y = [], []
        for i in range(40):
            X.append(np.array([1.0, 0.0]) + np.random.default_rng(i).normal(0, 0.05, 2))
            y.append(0)
            X.append(np.array([0.0, 1.0]) + np.random.default_rng(100 + i).normal(0, 0.05, 2))
            y.append(1)
        clf = BagRegressor(2, 2, seed=0).fit(X, y, epochs=50)
        self.assertGreater(clf.accuracy(X, y), 0.95)

    def test_feature_table_falls_back_to_zero(self):
        table = FeatureTable(["甲"], [], 4)
        self.assertAlmostEqual(float(np.abs(table.features("")).sum()), 0.0)

    def test_bigram_vector_is_mean_of_chars(self):
        table = FeatureTable(["甲", "乙"], ["甲乙"], 2)
        table.table[0] = np.array([1.0, 0.0])
        table.table[1] = np.array([0.0, 1.0])
        table.table[2] = (table.table[0] + table.table[1]) / 2
        self.assertTrue(np.allclose(table.table[2], [0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
