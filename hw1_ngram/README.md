# hw1_ngram

Character-level n-gram language models, written from scratch. No dependencies.

## What is here

```
ngram/
  corpus.py     cleaning and sentence splitting
  model.py      counts + MLE / AddK / Stupid Backoff / Kneser-Ney
  evaluate.py   perplexity, unseen n-gram ratio
  generate.py   sampling
  cli.py        `python -m ngram build|train`
scripts/
  parse_peopledaily.py   tagged 1998 release -> clean sentences
tests/
  test_ngram.py          15 tests
data/
  sample.txt   900 sentences from the real corpus, enough to run everything
results/
  ngram_results.json   committed output of the full run
```

## Usage

```bash
# fit orders 1..5 on the sample and print a table
python -m ngram train --corpus data/sample.txt --out results

# build a corpus from the tagged People's Daily release
python scripts/parse_peopledaily.py --src data/raw/*.txt --out data/corpus.txt --max-chars 2000000

# or from any plain text
python -m ngram build --raw data/raw/*.txt --out data/corpus.txt
```

Samples are written to stdout and into `results/ngram_results.json` together
with timings, entry counts and all perplexities.

## Results

People's Daily 1998, first two million characters of the six-month release:
47,066 sentences, 4,614 distinct characters, 9:1 train/test split.

| order | entries | build (s) | unseen n-grams | MLE | Laplace | Add-0.01 | Stupid Backoff | Kneser-Ney |
|---|---|---|---|---|---|---|---|---|
| 1 | 4,615 | 0.39 | 0.03% | ∞ | 661.66 | 662.55 | 663.39 | — |
| 2 | 282,774 | 1.30 | 8.2% | ∞ | 221.52 | 107.62 | 83.37 | 92.70 |
| 3 | 1,146,154 | 3.00 | 35.6% | ∞ | 1004.35 | 173.64 | 48.05 | 52.61 |
| 4 | 2,439,933 | 4.69 | 61.0% | ∞ | 2193.57 | 586.14 | 67.30 | **49.58** |
| 5 | 3,945,268 | 6.74 | 75.2% | ∞ | 2964.98 | 1282.44 | 128.25 | 50.75 |

Whole run: 57 seconds, single core.

Three things fall out:

1. MLE is unusable at every order — one unseen n-gram zeroes the whole sentence probability. At order 5, 75% of test n-grams are unseen.
2. Laplace gets steadily worse with order, 662 to 2965. A fixed pseudo-count spread over a 4,614-character vocabulary over-smooths exactly where the distribution is sharpest.
3. Kneser-Ney is best everywhere and bottoms out at order 4 (49.58), then rises slightly at order 5. That is the data-size wall: 3.9M five-gram entries from 1.8M training characters means most contexts are singletons, so backoff beats the estimate.

The last point is the interesting one, and it is the empirical form of "increasing *n* does not fix long distance dependencies".

## Sampling

Fixed prompt `在阳光明媚的五月`, 100 characters, temperature 1:

```
[kneser-ney, order 3]
在阳光明媚的五月的社会主义市场去年月，大厅里马科夫，等等。第十八日在工作...

[kneser-ney, order 5]
在阳光明媚的五月的风”在山东青岛市委市政府统一公布，自年月日起，北京市开展...

[stupid backoff, order 3]
在阳光明媚的五月端的建立外交部发言人阿地力雄厚爽的问题，又精神经济份合作...
```

Now that the corpus is news, the samples contain real collocations (`外交部发言人`,
`市委市政府`, `开展送温暖活动`), but none of them produce a grammatical clause.
That is the ceiling of a character-level Markov model, and order 5 is not
visibly better than order 3 even though its perplexity is slightly lower.

## Notes

- The cleaning step is not cosmetic. This release carries pinyin in braces
  (`历史上{shang5}`), sentence ids at the head of every line, and `〓` where the
  original encoder had no glyph. All three are removed before counting;
  leaving them in inflates the vocabulary with annotation instead of language.
- Kneser-Ney at order 1 is not computed: continuation counts require bigrams.
  The CLI skips it rather than printing a degenerate number.
- `data/corpus.txt` is not committed (4 MB of derived text). `sample.txt` is a
  900-line slice so the tests and the quick-start command work without the raw
  release.
