*****
ABSA
*****
This is pipeline for aspect-based sentiment analysis. Essential stages:

1. Preprocess::

    +------------------------------------- Preprocess pipeline ----------+
    |             |                                                      |
    |             V                                                      |
    |       Upload reviews                                               |
    |             |                                                      |
    |             |     List[Reviews]                                    |
    |             V                                                      |
    |         Spell check                                                |
    |             |                                                      |
    |             |     List[Reviews]                                    |
    |             V                                                      |
    |      Dependency parsing                                            |
    |             |                                                      |
    |             V                                                      |
    +--------------------------------------------------------------------+

2. Aspect term extraction

3. Polarity classification

----------
Setup
----------

**Requirements**

- `Python <https://www.python.org/downloads/>`_>=3.7
- `Poetry <https://python-poetry.org/docs/>`_>=0.12 # or another dependency manager
- `Torch <https://pytorch.org/get-started/locally/>`_>=1.5

To install all requisites you can execute script:

.. code-block:: bash

    ./setup.sh

Or run by yourself:

* Install dependencies with poetry

.. code-block:: bash

    poetry install --no-dev
    printf 'y\n\n' | python -c 'import stanfordnlp; stanfordnlp.download("ru")'


* Download pretrained embeddings

.. code-block:: bash

    wget http://vectors.nlpl.eu/repository/20/185.zip -P ./RusVectores/
    unzip RusVectores/185.zip -d RusVectores/tayga_upos_skipgram_300_2_2019
    rm RusVectores/185.zip

Take a look at `RusVectōrēs <https://rusvectores.org/ru/models/>`_ for
another pretrained embeddings. In this case don't forget to replace filename
in `src/__init__.py <https://gitlab.com/davydovdmitry/diploma-research/-/blob/master/src/__init__.py>`_


* Download dataset

.. code-block:: bash

    mkdir ./datasets/SemEval2016/
    wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1RZUyBrWQ0OwlIsmN0axewKg21koYmgQf' -O ./datasets/SemEval2016/train.xml
    wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1JR3gblfNXQHApmDzY4FCCjv_0wVug7dO' -O ./datasets/SemEval2016/test.xml
