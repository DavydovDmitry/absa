from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import torch as th
import numpy as np

from src import word2vec_model_path
from src import UNKNOWN_WORD, PAD_WORD


def get_embeddings() -> KeyedVectors:
    word2vec = KeyedVectors.load_word2vec_format(datapath(word2vec_model_path), binary=True)

    for word in [UNKNOWN_WORD]:
        embed_dim = word2vec.vector_size
        embed = th.empty(embed_dim, 1)
        embed = th.nn.init.xavier_uniform_(embed).numpy().reshape(embed_dim, )
        word2vec.add(entities=[word], weights=[embed])

    for word in [PAD_WORD]:
        embed_dim = word2vec.vector_size
        embed = np.zeros((embed_dim, ))
        word2vec.add(entities=[word], weights=[embed])

    # word2vec.init_sims(replace=True)
    return word2vec
