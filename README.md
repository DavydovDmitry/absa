# ABSA
This is pipeline for Aspect-Based Sentiment Analysis of texts.
Primary tasks:
- extract aspect terms;
- classify aspect terms on aspect categories;
- classify aspect terms on polarity.

### Essential processing stages:

![](notebooks/images/pipeline.svg)

# Setup

- [Python](https://www.python.org/downloads/) >= 3.7
- [Poetry](https://python-poetry.org/docs/) >= 0.12
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.5

----------
Execution
----------
```shell script
python train.py    # to train classifiers
python example.py  # to process text in example directory
```
