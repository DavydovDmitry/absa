*****
ABSA
*****

This is pipeline for aspect-based sentiment analysis.

----------
Execution
----------

Just execute `run_pipeline.py` to run full pipeline:

.. code-block:: bash

    python run_pipeline.py

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

    wget http://vectors.nlpl.eu/repository/20/185.zip -P ./embeddings/
    unzip embeddings/185.zip -d embeddings/tayga_upos_skipgram_300_2_2019 &&
    rm embeddings/185.zip embeddings/README

Take a look at `RusVectōrēs <https://rusvectores.org/ru/models/>`_ for
another pretrained embeddings. In this case don't forget to replace filename
in `absa/__init__.py <https://gitlab.com/davydovdmitry/absa/-/blob/master/absa/__init__.py>`_


* Download dataset

.. code-block:: bash

    mkdir -p ./datasets/SemEval2016/ &&
    wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1RZUyBrWQ0OwlIsmN0axewKg21koYmgQf' -O ./datasets/SemEval2016/train.xml &&
    wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1JR3gblfNXQHApmDzY4FCCjv_0wVug7dO' -O ./datasets/SemEval2016/test.xml


------
Essential stages
------

::


                  |
    +-------------------------------------- Preprocess pipeline ---------------+
    |             |                                                            |
    |             V                                                            |
    |       Upload reviews                                                     |
    |             |                                                            |
    |             |     List[Reviews]                                          |
    |             V                                                            |
    |         Spell check                                                      |
    |             |                                                            |
    |             |     List[Reviews]                                          |
    |             V                                                            |
    |      Dependency parsing                                                  |
    |             |                                                            |
    +--------------------------------------------------------------------------+
                  |
                  V
                  |
    +-------------------------------------- ABSA pipeline ---------------------+
    |             |                                                            |
    +-------------------------------------- Aspect Classification -------------+
    |             V                                                            |
    | Sentence Level Aspect Classification                                     |
    |             |                                                            |
    |             |     List[ParsedSentence]                                   |
    |             V                                                            |
    | Target Level Aspect Classification                                       |
    |             |                                                            |
    +--------------------------------------------------------------------------+
    |             |     List[ParsedSentence]                                   |
    |             V                                                            |
    |      Polarity Classification                                             |
    |             |                                                            |
    |             |                                                            |
    +--------------------------------------------------------------------------+
                  |
                  V
