# 大语言模型与应用实践 · 课程作业

2026 年秋季《大语言模型与应用实践》的课程作业。三个编程项目各自独立、能单独运行，另有第三讲六篇文字作业。

| 项目 | 内容 | 状态 |
|---|---|---|
| [`hw1_ngram`](hw1_ngram) | 字符级 n-gram 语言模型，五种平滑方法，困惑度对比与文本续写 | 测试通过，结果已提交 |
| [`hw2_veccls`](hw2_veccls) | 从零实现 skip-gram 与负采样，FastText 式文本分类，附 gensim 基线 | 测试通过，结果已提交 |
| [`bert_punc`](bert_punc) | BERT 逐字预测标点，长文用重叠窗口 + 批处理解码 | 已训练，F1 0.914 |
| [`l3`](l3) | 第三讲六道思考题：后训练演进、提示词泄漏、提示经验检验、宪法规则、唐诗英译、DPO 对比 | 文档，无代码 |

`l3` 是纯文本作业，没有可执行的代码。

## 为什么是手写实现

写这套代码的机器装不上 gensim：没有 PyPI 访问，`jieba`、`scikit-learn`、`torch` 也都缺，只有 numpy。

与其交一份"没跑过"的作业，不如把算法自己写一遍。写的过程中反而看清了几件库会替你藏起来的事：负采样里词频的 0.75 次方到底怎么采样、Kneser-Ney 的续接计数为什么能救回退、高频词下采样对常见字的伤害有多大。

`hw2_veccls/gensim_baseline.py` 保留了标准 gensim 版本，在有环境的机器上可以直接跑，用来对照。

## 快速开始

```bash
# hw1：训练 1–5 阶模型并输出困惑度
cd hw1_ngram
python -m ngram train --corpus data/sample.txt --out results

# hw2：词向量 + 文本分类
cd ../hw2_veccls
python -m veccls run --data data/labeled.tsv --embed-corpus data/body.txt --classes 2 4 8 --out results

# bert_punc：训练断句模型（需要 GPU 或耐心）
cd ../bert_punc
python train.py --corpus data/corpus.txt --epochs 2 --batch 32 --out results
```

需要 Python 3.10 以上。`hw1` 只用标准库，`hw2` 需要 numpy，`bert_punc` 需要 torch 与 transformers。

## 测试

```bash
cd hw1_ngram && python -m unittest discover -s tests -v   # 15 个
cd hw2_veccls && python -m unittest discover -s tests -v  # 12 个
cd bert_punc && python -m unittest discover -s tests -v   # 11 个
```

## 数据

三个项目都跑在人民日报 1998 年上半年那六个文件上（GB18030 编码）。原始语料不方便随仓库分发，所以提交的是派生材料：`hw1` 里 900 句的切片，`hw2` 里 1117 条按关键词标注的标题。`bert_punc` 另外用《三国演义》做零日文迁移测试，文本可重新下载。解析脚本按原格式处理，重跑全量数据从这几条命令开始：

```bash
# hw1：200 万字的干净句子
python hw1_ngram/scripts/parse_peopledaily.py --src data/raw/*.txt \
    --out hw1_ngram/data/corpus.txt --max-chars 2000000

# hw2：带标签的标题 + 用于训练词向量的正文
python hw2_veccls/scripts/parse_peopledaily.py --src data/raw/*.txt \
    --out hw2_veccls/data/labeled.tsv --body-out hw2_veccls/data/body.txt --per-label 140

# bert_punc：断句训练语料
python bert_punc/scripts/parse_peopledaily.py --src data/raw/*.txt \
    --out bert_punc/data/corpus.txt --max-chars 2000000
```

这批语料有几个要处理的地方：每行行首是句子编号，部分词后面跟着花括号注音（`历史上{shang5}`），`〓` 表示原编码表里没有的字符，另外三月那个文件在 4.25MB 附近有 3 个坏字节，严格解码会直接失败。

## 关于结果

数字都是跑出来的，没有修饰：

- Kneser-Ney 在 4 阶触底、5 阶回升，是语料规模造成的，不是实现问题。
- `hw2` 里 `译` 的近邻全是人名用字，因为这些字在这批新闻里就是跟着人名出现的。
- 8 类分类准确率 0.80，测试集只有 223 条，别当成精确值。
- `bert_punc` 的标点模型在现代文上 F1 0.914，迁到《三国演义》只有 0.50，掉得很厉害，报告里写了错在哪。
- `bert_punc` 的批处理解码只比串行快一倍左右，不是数量级的差距，原因在报告里拆开了。
