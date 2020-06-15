# Essential stages
![](analysis/images/pipeline.svg)

# Setup

**1) Requirements**

- [Python](https://www.python.org/downloads/) >= 3.7
- [Poetry](https://python-poetry.org/docs/) >= 0.12
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.5

**2) Requisites**

To install all requisites you can execute script:

```bash
./setup.sh
```

Or run by yourself:

* Install dependencies with poetry

```bash
poetry install --no-dev
printf 'y\n\n' | python -c 'import stanfordnlp; stanfordnlp.download("ru")'
```

* Download pretrained embeddings

```bash
wget http://vectors.nlpl.eu/repository/20/185.zip -P ./embeddings/
unzip embeddings/185.zip -d embeddings/tayga_upos_skipgram_300_2_2019 &&
rm embeddings/185.zip embeddings/README
```


Take a look at [RusVectōrēs](https://rusvectores.org/ru/models/) for
another pretrained embeddings. In this case don't forget to replace filename
in [absa/\_\_init__.py](https://gitlab.com/davydovdmitry/absa/-/blob/master/absa/__init__.py)


* Download dataset

```bash
mkdir -p ./datasets/SemEval2016/ &&
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1RZUyBrWQ0OwlIsmN0axewKg21koYmgQf' -O ./datasets/SemEval2016/train.xml &&
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1JR3gblfNXQHApmDzY4FCCjv_0wVug7dO' -O ./datasets/SemEval2016/test.xml
```

----------
Execution
----------

Execute `train.py` to train classifiers.<br>
Put your text file to input directory and run `process.py`.

```bash
python train.py
python process.py
```
