# llm-nlp-homework

Coursework for *大语言模型与应用实践* (Fall 2026). Two projects, each self-contained.

| Project | What it does | Status |
|---|---|---|
| [`hw1_ngram`](hw1_ngram) | Character-level n-gram language models with five smoothing schemes, perplexity comparison, sampling | tests pass, results committed |
| [`hw2_veccls`](hw2_veccls) | Skip-gram with negative sampling from scratch, plus a fastText-style text classifier, and a gensim baseline | tests pass, results committed |

## Why hand-rolled implementations

The machine this was developed on has no PyPI access, so `gensim`, `jieba`,
`scikit-learn` and `torch` could not be installed. Rather than submit something
that was never run, both projects implement the core algorithms directly on
`numpy` and the standard library, and `hw2_veccls/gensim_baseline.py` keeps the
short gensim version for machines that have it.

That turned out to be useful for its own sake: writing the negative sampling
loop and the Kneser-Ney recursion by hand makes the parts that libraries hide
visible.

## Quick start

```bash
# hw1: fit 1..5-gram models and print perplexity
cd hw1_ngram
python -m ngram train --corpus data/sample.txt --out results

# hw2: embeddings + classification
cd ../hw2_veccls
python -m veccls run --data data/labeled.tsv --out results
```

Both need Python 3.10+; `hw2` needs `numpy`. There are no other dependencies.

## Tests

```bash
cd hw1_ngram && python -m unittest discover -s tests -v
cd hw2_veccls && python -m unittest discover -s tests -v
```

## Data

Both projects run on the People's Daily 1998 release (GB18030, six monthly
files). The raw files are not redistributable, so what is committed is the
derived material: a 900-sentence slice for `hw1`, and 1,117 keyword-labelled
headlines for `hw2`. Both parsers handle the release as shipped and are the
entry point for rebuilding everything:

```bash
# hw1: 2M characters of clean sentences
python hw1_ngram/scripts/parse_peopledaily.py --src data/raw/*.txt \
    --out hw1_ngram/data/corpus.txt --max-chars 2000000

# hw2: labelled headlines plus body text for the embeddings
python hw2_veccls/scripts/parse_peopledaily.py --src data/raw/*.txt \
    --out hw2_veccls/data/labeled.tsv --body-out hw2_veccls/data/body.txt --per-label 140
```

The release needs a little care: sentence ids sit at the head of every line,
pinyin appears in braces (`历史上{shang5}`), `〓` marks characters the original
encoder could not represent, and one monthly file has three undecodable bytes
around offset 4.25M.

## Notes on honesty

The results are reported as they came out. Kneser-Ney bottoming out at order 4
rather than 5 is a data-size wall, not a bug. The `译` neighbours in `hw2` are
name characters because that is what the corpus contains. The 8-class accuracy
in `hw2` is 0.80 and the test set is only 223 headlines, so it is not a precise
number.
