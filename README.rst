*****
Setup
*****

**Requirements**

- `Python <https://www.python.org/downloads/>`_>=3.7
- `Poetry <https://python-poetry.org/docs/>`_>=0.12 # or another dependency manager
- `Torch <https://pytorch.org/get-started/locally/>`_>=1.5


**Install dependencies with poetry**

.. code-block:: bash

    poetry install --no-dev
    printf 'y\n\n' | python -c 'import stanfordnlp; stanfordnlp.download("ru")'


**Download embeddings**

.. code-block:: bash

    wget http://vectors.nlpl.eu/repository/20/185.zip -P ./RusVectores/
    unzip RusVectores/185.zip -d RusVectores/tayga_upos_skipgram_300_2_2019
    rm RusVectores/185.zip

Take a look at `RusVectōrēs <https://rusvectores.org/ru/models/>`_ for
another pretrained embeddings. In this case don't forget to replace filename
in `src/__init__.py <https://gitlab.com/davydovdmitry/diploma-research/-/blob/master/src/__init__.py>`_
