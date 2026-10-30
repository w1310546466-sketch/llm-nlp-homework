"""词向量与文本分类的 numpy 实现。"""

from .classifier import BagRegressor, FeatureTable
from .skipgram import SkipGram

__all__ = ["SkipGram", "BagRegressor", "FeatureTable"]
__version__ = "0.1.0"
