"""用 BERT 做中文断句标点的实验代码。"""

from .decode import make_windows, merge, render

__all__ = ["make_windows", "merge", "render"]
__version__ = "0.1.0"
