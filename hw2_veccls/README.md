# hw2_veccls

Word embeddings and text classification, numpy only.

## What is here

```
veccls/
  corpus.py      TSV loading, cleaning, split
  skipgram.py    skip-gram + negative sampling + frequent-word subsampling
  classifier.py  fastText-style bag-of-features softmax
  cli.py         `python -m veccls run`
gensim_baseline.py   same experiment with gensim + sklearn, for machines that have them
scripts/
  parse_peopledaily.py   headlines -> topic labels; body text for embeddings
tests/
  test_veccls.py     12 tests
data/
  labeled.tsv   1,117 headlines, 8 topics, ~140 each
  body.txt      not committed; regenerated from the raw release
results/
  veccls_results.json, vocab.txt
```

## Data

Headlines are taken from the People's Daily 1998 release: the first line of
every article, cleaned the same way as in `hw1_ngram`. The corpus has no topic
labels, so `scripts/parse_peopledaily.py` assigns each headline to the topic
with the most keyword hits among eight hand-written lists (体育 金融 教育 文化
科技 医疗 农业 法律), and drops headlines that match nothing or tie.

```bash
python scripts/parse_peopledaily.py --src data/raw/*.txt \
    --out data/labeled.tsv --body-out data/body.txt --per-label 140
```

`--per-label 140` is not arbitrary: 医疗 only yields 141 usable headlines in the
six-month release, so 140 is the largest balanced set. Uneven classes would make
the 2/4/8 comparison meaningless.

Embeddings are trained on body text, not headlines. Headlines average 15
characters, which is far too little to place 2,600 characters; `--embed-corpus`
takes a body-text file for that.

## Usage

```bash
python -m veccls run --data data/labeled.tsv --embed-corpus data/body.txt \
    --classes 2 4 8 --out results
```

## Results

Embeddings: 400k characters of body text, dim 64, window 3, 5 negatives,
3 epochs, 187 s on one core. Classifier: mean-pooled features, 40 epochs.

| classes | random / chars | random / chars+bigrams | skip-gram / chars | skip-gram / chars+bigrams |
|---|---|---|---|---|
| 2 (体育, 金融) | 0.7937 | 0.7937 | 0.9841 | **1.0000** |
| 4 (+教育, 文化) | 0.6694 | 0.7258 | 0.8065 | **0.8226** |
| 8 (all) | 0.5426 | 0.5785 | 0.7489 | **0.8027** |

Test sets are 63 / 124 / 223 headlines, so the 8-class numbers carry a couple of
points of noise.

Nearest neighbours from the skip-gram run:

```
模 → 规(0.64) 颇(0.62) 范(0.61) 拟(0.61) 矩(0.59)     规模, 模范, 模拟
检 → 测(0.67) 查(0.66) 嫌(0.61) 督(0.60) 逮(0.56) 监(0.54)
奖 → 铜(0.65) 励(0.60) 获(0.52) 颁(0.51)              铜牌, 奖励, 获奖
译 → 岱(0.83) 琦(0.83) 雯(0.82) 娟(0.81)              names, i.e. junk
```

`模`, `检`, `奖` all land next to characters that form real compounds with them.
`译` fails: in this corpus it shows up inside transliterated personal names, so
its neighbours are name characters. That is a fair finding and it is left in.

## What the ablation says

- Pretrained initialisation is worth a lot: +19 pts at 2 classes, +13.7 at 4,
  +20.6 at 8 (against random init, chars only). With ~900 training headlines the
  classifier cannot learn a semantic space from scratch.
- Bigrams help once there are enough classes (+1.6 at 4, +5.4 at 8). This is the
  opposite of what the same experiment showed on the smaller lecture corpus,
  where mean pooling diluted them — with 900 headlines that encode eight topics,
  two-character cues like 股价/球队 carry the signal.

## gensim baseline

```bash
pip install gensim jieba scikit-learn
python gensim_baseline.py --data data/labeled.tsv
```

`gensim.models.FastText` is unsupervised subword embedding; the supervised
classifier is `fasttext.train_supervised` from Facebook's package. They are
different things and are often conflated.
