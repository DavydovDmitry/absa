import os

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import torch as th
import numpy as np

from absa import embeddings_path, embed_matrix_path, vocabulary_dump,\
    UNKNOWN_WORD, PAD_WORD
from .dump import make_dump, load_dump


def _get_embeddings() -> KeyedVectors:
    """Load embeddings

    Returns
    -------
    word2vec : KeyedVectors
    """
    word2vec = KeyedVectors.load_word2vec_format(datapath(embeddings_path), binary=True)

    for word in [UNKNOWN_WORD]:
        embed_dim = word2vec.vector_size
        embed = th.empty(embed_dim, 1)
        embed = th.nn.init.xavier_uniform_(embed).numpy().reshape(embed_dim, )
        word2vec.add(entities=[
            word,
        ], weights=[
            embed,
        ])

    for word in [PAD_WORD]:
        embed_dim = word2vec.vector_size
        embed = np.zeros((embed_dim, ))
        word2vec.add(entities=[
            word,
        ], weights=[
            embed,
        ])

    # word2vec.init_sims(replace=True)
    return word2vec


class MetaEmbeddings:
    def __init__(self, *args, **kwargs):
        if os.path.isfile(vocabulary_dump) and (os.path.isfile(embed_matrix_path)):
            self._vocabulary = load_dump(pathway=vocabulary_dump)
            self._embeddings_matrix = load_dump(pathway=embed_matrix_path)
        else:
            embeddings = _get_embeddings()
            self._vocabulary = {w: i for i, w in enumerate(embeddings.index2word)}
            self._embeddings_matrix = th.FloatTensor(embeddings.vectors)

            make_dump(obj=self._vocabulary, pathway=vocabulary_dump)
            make_dump(obj=self._embeddings_matrix, pathway=embed_matrix_path)

    @property
    def vocabulary(self):
        return self._vocabulary

    @property
    def embeddings_matrix(self):
        return self._embeddings_matrix


class Embeddings(metaclass=MetaEmbeddings):
    pass
