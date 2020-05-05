import copy
from typing import List, Dict
from collections import namedtuple

import torch as th
import numpy as np
import networkx as nx

from src import UNKNOWN_WORD, PAD_WORD
from src.review.parsed_sentence import ParsedSentence

Batch = namedtuple(
    'Batch',
    ['sentence_index', 'target_index', 'sentence_len', 'embed_ids', 'adj', 'mask', 'polarity'])


class DataLoader:
    """DataLoader for polarity classifier.

    Create batch element for every target.

    Attributes
    ----------
    vocabulary : dict
        dictionary where key is wordlemma_POS value - index in embedding matrix
    """
    def __init__(self,
                 vocabulary: Dict[str, int],
                 sentences: List[ParsedSentence],
                 batch_size: int,
                 device: th.device,
                 unknown_word=UNKNOWN_WORD,
                 pad_word=PAD_WORD):
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.device = device
        self.unknown_word = unknown_word
        self.pad_word = pad_word

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
                if sentence.id2lemma[word] in self.vocabulary:
                    embed_ids.append(self.vocabulary[sentence.id2lemma[word]])
                else:
                    embed_ids.append(self.vocabulary[self.unknown_word])

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
                    adj=sentence.graph,  # pass graph instead of matrix
                    mask=mask,
                    polarity=polarity))
        return processed

    def __getitem__(self, key: int) -> Batch:
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
        max_len = max([x.sentence_len for x in batch])
        batch = sort_by_sentence_len(batch=batch)

        sentence_index = th.LongTensor([x.sentence_index for x in batch])
        target_index = th.LongTensor([x.target_index for x in batch])
        sentence_lens = th.LongTensor([x.sentence_len for x in batch])
        polarity = th.LongTensor([x.polarity for x in batch])

        embed_ids = th.LongTensor(batch_size, max_len).fill_(self.vocabulary[self.pad_word])
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
        """Iterate over batches"""
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __len__(self) -> int:
        """Return number of batches"""
        return len(self.data)


def sort_by_sentence_len(batch: List[Batch]) -> List[Batch]:
    """
    Sort sentences by descending order of it's len.

    Not return original index because there is already sentence and target
    indexes in batch.
    """
    _, sorted_batch = list(
        zip(*sorted(enumerate(batch), key=lambda x: x[1].sentence_len, reverse=True)))
    return sorted_batch
