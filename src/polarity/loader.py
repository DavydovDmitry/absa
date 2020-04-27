import copy
from typing import List, Tuple
from collections import namedtuple

import torch as th
import numpy as np
import networkx as nx
from gensim.models import KeyedVectors

from src import UNKNOWN_WORD, PAD_WORD
from src.review.parsed_sentence import ParsedSentence

Batch = namedtuple(
    'Batch',
    ['sentence_index', 'target_index', 'sentence_len', 'embed_ids', 'adj', 'mask', 'polarity'])


class DataLoader:
    def __init__(self, word2vec: KeyedVectors, sentences: List[ParsedSentence],
                 batch_size: int, device: th.device):
        self.word2vec = word2vec
        self.batch_size = batch_size
        self.device = device

        data = self.process(sentences)
        self.num_examples = len(data)

        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def process(self, sentences: List[ParsedSentence]) -> List[Batch]:
        """Process all sentences"""
        processed = []
        for sentence_index, sentence in enumerate(sentences):
            if sentence.graph.nodes and sentence.targets:
                processed.extend(
                    self.process_sentence(sentence=sentence, sentence_index=sentence_index))
        return processed

    def process_sentence(self, sentence: ParsedSentence, sentence_index: int) -> List[Batch]:
        """Process one sentence"""
        processed = []

        embed_ids = []
        for word in sentence.get_sentence_order():
            if word in sentence.id2lemma:
                if sentence.id2lemma[word] in self.word2vec.vocab:
                    embed_ids.append(self.word2vec.vocab[sentence.id2lemma[word]].index)
                else:
                    embed_ids.append(self.word2vec.vocab[UNKNOWN_WORD].index)

        sentence_len = len(sentence.graph.nodes)

        for target_index, target in enumerate(sentence.targets):
            if not target.nodes:
                mask = [1 for _ in embed_ids]
            else:
                mask = [0 for _ in embed_ids]
                for word_index, word in enumerate(sentence.get_sentence_order()):
                    if word in target.nodes:
                        mask[word_index] = 1
            polarity = target.polarity.value

            processed.append(
                Batch(
                    sentence_index=sentence_index,
                    target_index=target_index,
                    sentence_len=sentence_len,
                    embed_ids=embed_ids,
                    adj=sentence.graph,  # todo: pass graph instead of matrix
                    mask=mask,
                    polarity=polarity))
        return processed

    def __getitem__(self, key) -> Batch:
        """
        Return batch as named tuple where each element is th.Tensor where
        first dimension is element in batch
        """

        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError

        batch = self.data[key]
        batch_size = len(batch)
        # batch = list(zip(*batch))

        # sort all fields by lens for easy RNN operations
        max_len = max([x.sentence_len for x in batch])
        batch = sort_all(batch=batch)  # todo: pass original order

        sentence_index = th.LongTensor([x.sentence_index for x in batch])
        target_index = th.LongTensor([x.target_index for x in batch])
        sentence_lens = th.LongTensor([x.sentence_len for x in batch])
        polarity = th.LongTensor([x.polarity for x in batch])

        embed_ids = th.LongTensor(batch_size,
                                  max_len).fill_(self.word2vec.vocab[PAD_WORD].index)
        mask = th.FloatTensor(batch_size, max_len).fill_(0)
        adj = []
        for i, b in enumerate(batch):
            embed_ids[i, :b.sentence_len] = th.LongTensor(b.embed_ids)
            mask[i, :b.sentence_len] = th.FloatTensor(b.mask)

            # adjacency matrix
            graph = copy.deepcopy(b.adj)
            for node in graph:
                graph.add_edge(node, node)
            a = np.zeros((1, max_len, max_len), dtype=np.float32)
            a[:, :b.sentence_len, :b.sentence_len] = nx.to_numpy_matrix(graph)
            adj.append(a)
        adj = np.concatenate(adj, axis=0)
        adj = th.from_numpy(adj)

        batch = Batch(sentence_index=sentence_index,
                      target_index=target_index,
                      sentence_len=sentence_lens.to(self.device),
                      embed_ids=embed_ids.to(self.device),
                      adj=adj.to(self.device),
                      mask=mask.to(self.device),
                      polarity=polarity.to(self.device))
        return batch

    def __iter__(self) -> Batch:
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __len__(self) -> int:
        return len(self.data)


def sort_all(batch: List[Batch]) -> List[Batch]:
    """
    Sort all fields by descending order of lens, and return the original indices.
    """
    _, sorted_batch = list(
        zip(*sorted(enumerate(batch), key=lambda x: x[1].sentence_len, reverse=True)))
    return sorted_batch
