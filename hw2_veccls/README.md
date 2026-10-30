# hw2：词向量与文本分类

纯 numpy 实现，不依赖 gensim。

## 目录

```
veccls/
  corpus.py      TSV 读取、清洗、切分
  skipgram.py    skip-gram + 负采样 + 高频词下采样
  classifier.py  FastText 式词袋特征 + softmax
  cli.py         `python -m veccls run`
gensim_baseline.py    有 gensim 时跑的对照版本
scripts/
  parse_peopledaily.py   标题 → 主题标签；正文 → 训练语料
tests/
  test_veccls.py         12 个单元测试
data/
  labeled.tsv    1,117 条标题，8 个主题，每类约 140 条
results/
  veccls_results.json、vocab.txt
```

## 数据

标题取自人民日报 1998 年上半年那六个文件的每篇文章首行，清洗方式与 `hw1` 一致。语料本身没有主题标注，所以标签是 `scripts/parse_peopledaily.py` 按关键词规则打出来的：八组关键词（体育、金融、教育、文化、科技、医疗、农业、法律），每条标题归到命中最多的一类，一条都没命中或者两类打平的直接丢掉。

```bash
python scripts/parse_peopledaily.py --src data/raw/*.txt \
    --out data/labeled.tsv --body-out data/body.txt --per-label 140
```

`--per-label 140` 不是随手定的：六个月里"医疗"只能凑出 141 条可用标题，140 是能保持八类均衡的最大值。类别大小不一致的话，2 类/4 类/8 类的对比就没有意义了。

词向量在正文上训练，不在标题上。标题平均 15 个字，2600 多个字的词表根本铺不开，`--embed-corpus` 就是为这件事留的参数。

## 用法

```bash
python -m veccls run --data data/labeled.tsv --embed-corpus data/body.txt \
    --classes 2 4 8 --out results
```

## 结果

词向量：40 万字正文，维度 64，窗口 3，负采样 5，3 个 epoch，单核 187 秒。分类器：均值池化特征，40 个 epoch。

| 类别数 | 随机/单字 | 随机/单字+二元组 | skip-gram/单字 | skip-gram/单字+二元组 |
|---|---|---|---|---|
| 2 类（体育、金融） | 0.7937 | 0.7937 | 0.9841 | **1.0000** |
| 4 类（加教育、文化） | 0.6694 | 0.7258 | 0.8065 | **0.8226** |
| 8 类 | 0.5426 | 0.5785 | 0.7489 | **0.8027** |

测试集分别是 63、124、223 条，8 类那一行有一两个点的波动，别当精确值看。

近邻抽查：

```
模 → 规(0.64) 颇(0.62) 范(0.61) 拟(0.61) 矩(0.59)      规模、模范、模拟
检 → 测(0.67) 查(0.66) 嫌(0.61) 督(0.60) 逮(0.56) 监(0.54)
奖 → 铜(0.65) 励(0.60) 获(0.52) 颁(0.51)               铜牌、奖励、获奖
译 → 岱(0.83) 琦(0.83) 雯(0.82) 娟(0.81)               人名用字，失败
```

前三个说明向量确实抓到了搭配。第四个是失败案例：这批新闻里"译"大量出现在音译人名中，它的上下文被名字占满了。这个反例留在仓库里。

## 消融说明了什么

- 预训练初始化收益很大：2 类 +19 个点，4 类 +13.7，8 类 +20.6（都相对随机初始化、单字特征）。训练集只有九百条标题时，分类器没法从零学出一个语义空间。
- 二元组只有在类别多的时候才帮上忙（4 类 +1.6，8 类 +5.4）。这和在更小的课件语料上得到的结论相反——那次均值池化把二元组稀释掉了；这次数据换成了标题，"股价""球队""法院""水稻"这类两字线索本身就很接近类别标记，池化之后仍有区分度。

## gensim 基线

```bash
pip install gensim jieba scikit-learn
python gensim_baseline.py --data data/labeled.tsv
```

一个容易混的点：`gensim.models.FastText` 是无监督的子词向量，带 `__label__` 的有监督分类在 Facebook 的 `fasttext` 包里，两者不是一回事。
