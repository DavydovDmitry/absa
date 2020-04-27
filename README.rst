*****
Setup
*****

**Requirements**

- `Python <https://www.python.org/downloads/>`_>=3.7
- `Torch <https://pytorch.org/get-started/locally/>`_>=1.5


**Install dependencies with poetry**

.. code-block:: bash

    poetry install


**Download embeddings**

.. code-block:: bash

    wget http://vectors.nlpl.eu/repository/20/185.zip -P ./RusVectores/
    unzip RusVectores/185.zip -d RusVectores/tayga_upos_skipgram_300_2_2019
    rm RusVectores/185.zip
