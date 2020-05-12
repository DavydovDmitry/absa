from typing import List, Dict
from collections import namedtuple

import torch as th
import dgl
import numpy as np

from src import UNKNOWN_WORD, PAD_WORD
from src.review.parsed_sentence import ParsedSentence

Batch = namedtuple(
    'Batch',
    [
        # GPU part. This fields will be passed to GPU.
        'sentence_len',
        'embed_ids',
        'graph',
        # CPU part,
        'labels'
    ])


class DataLoader:
    """DataLoader for polarity classifier.

    Create batches from sentences. Create batch element for every target.

    Attributes
    ----------
    vocabulary : dict
        dictionary where key is 'wordlemma_POS' value - index in embedding matrix
    data : List[List[Batch]]
        Split to chunks batch elements.
    """
    def __init__(self,
                 vocabulary: Dict[str, int],
                 sentences: List[ParsedSentence],
                 batch_size: int,
                 device: th.device,
                 aspect_labels: np.array,
                 unknown_word=UNKNOWN_WORD,
                 pad_word=PAD_WORD):
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.device = device
        self.unknown_word = unknown_word
        self.pad_word = pad_word
        self.aspect_labels = aspect_labels

        data = self.process(sentences)
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def process(self, sentences: List[ParsedSentence]) -> List[Batch]:
        """Process all sentences"""
        processed = []
        for sentence_index, sentence in enumerate(sentences):
            if sentence.graph.nodes and sentence.targets:
                processed.append(
                    self.process_sentence(sentence=sentence, sentence_index=sentence_index))
        return processed

    def process_sentence(self, sentence: ParsedSentence, sentence_index: int) -> Batch:
        """Process one sentence
        """

        embed_ids = []
        for word in sentence.get_sentence_order():
            if word in sentence.id2lemma:
                if sentence.id2lemma[word] in self.vocabulary:
                    embed_ids.append(self.vocabulary[sentence.id2lemma[word]])
                else:
                    embed_ids.append(self.vocabulary[self.unknown_word])

        sentence_len = len(sentence.graph.nodes)

        graph = dgl.DGLGraph()
        graph.from_networkx(sentence.graph)

        labels = th.zeros(size=(sentence_len, self.aspect_labels.shape[0]), dtype=th.float)
        for target_index, target in enumerate(sentence.targets):
            if not target.nodes:
                labels[:, np.where(self.aspect_labels == target.category)] = 1
            else:
                for word_index, word in enumerate(sentence.get_sentence_order()):
                    if word in target.nodes:
                        labels[word_index, np.where(self.aspect_labels == target.category)] = 1

        return Batch(
            sentence_len=sentence_len,
            embed_ids=embed_ids,
            graph=graph,
            labels=labels,
        )

    def __getitem__(self, batch_index: int) -> Batch:
        """
        Return batch as named tuple where each element is th.Tensor or batched dgl.DGLGraph.
        First dimension of tensors correspond to batch elements.

        Parameters
        ----------
        batch_index : int
            Index of batch in dataset.

        Returns
        -------
        batch : Batch
            namedtuple of th.Tensors and batched dgl.DGLGraph.
        """

        if not isinstance(batch_index, int):
            raise TypeError
        if batch_index < 0 or batch_index >= len(self.data):
            raise IndexError

        batch = self.data[batch_index]
        batch_size = len(batch)
        max_len = max([x.sentence_len for x in batch])
        batch = sort_by_sentence_len(batch=batch)

        sentence_lens = th.LongTensor([x.sentence_len for x in batch])

        embed_ids = th.LongTensor(batch_size, max_len).fill_(self.vocabulary[self.pad_word])
        for i, b in enumerate(batch):
            embed_ids[i, :b.sentence_len] = th.LongTensor(b.embed_ids)

        return Batch(
            sentence_len=sentence_lens.to(self.device),
            embed_ids=embed_ids.to(self.device),
            graph=dgl.batch([item.graph for item in batch]),
            labels=th.cat([item.labels for item in batch]).to(self.device),
        )

    def __iter__(self) -> Batch:
        """Iterate over batches - chunks of dataset"""
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __len__(self) -> int:
        """Return number of batches"""
        return len(self.data)


def sort_by_sentence_len(batch: List[Batch]) -> List[Batch]:
    """Sort sentences by descending order of it's len.

    Not return original index because there is already sentence and target
    indexes in batch.
    """
    _, sorted_batch = list(
        zip(*sorted(enumerate(batch), key=lambda x: x[1].sentence_len, reverse=True)))
    return sorted_batch
