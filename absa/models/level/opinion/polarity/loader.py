from typing import List, Dict
from collections import namedtuple

import torch as th
import dgl

from absa import UNKNOWN_WORD, PAD_WORD
from absa.text.parsed.text import ParsedText
from absa.text.parsed.sentence import ParsedSentence

Batch = namedtuple(
    'Batch',
    [
        # Indexes to set classification results to sentence
        'text_index',
        'sentence_index',
        'opinion_index',
        # GPU part. This fields will be passed to GPU.
        'sentence_len',
        'embed_ids',
        'graph',
        # CPU part
        'term_mask',
        'polarity'
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
                 texts: List[ParsedText],
                 batch_size: int,
                 device: th.device,
                 unknown_word=UNKNOWN_WORD,
                 pad_word=PAD_WORD):
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.device = device
        self.unknown_word = unknown_word
        self.pad_word = pad_word

        data = self.process(texts)
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def process(self, texts: List[ParsedText]) -> List[Batch]:
        """Process all texts

        Parameters
        ----------
        todo:

        Return
        ------
        """
        processed = []
        for text_index, text in enumerate(texts):
            for sentence_index, sentence in enumerate(text):
                if sentence.graph.nodes and sentence.opinions:
                    processed.extend(
                        self.process_sentence(sentence=sentence,
                                              text_index=text_index,
                                              sentence_index=sentence_index))
        return processed

    def process_sentence(self, sentence: ParsedSentence, text_index: int,
                         sentence_index: int) -> List[Batch]:
        """Process one sentence

        Returns
        -------
        processed : List[Batch]
            Batch elements for every target.
        """
        processed = []

        embed_ids = []
        for word in sentence.nodes_sentence_order():
            if word in sentence.id2lemma:
                if sentence.id2lemma[word] in self.vocabulary:
                    embed_ids.append(self.vocabulary[sentence.id2lemma[word]])
                else:
                    embed_ids.append(self.vocabulary[self.unknown_word])

        sentence_len = len(sentence.graph.nodes)

        graph = dgl.DGLGraph()
        graph.from_networkx(sentence.graph)

        for opinion_index, opinion in enumerate(sentence.opinions):
            if not opinion.nodes:
                term_mask = [1 for _ in embed_ids]
            else:
                term_mask = [0 for _ in embed_ids]
                for word_index, word in enumerate(sentence.nodes_sentence_order()):
                    if word in opinion.nodes:
                        term_mask[word_index] = 1
            polarity = opinion.polarity.value

            processed.append(
                Batch(text_index=text_index,
                      sentence_index=sentence_index,
                      opinion_index=opinion_index,
                      sentence_len=sentence_len,
                      embed_ids=embed_ids,
                      graph=graph,
                      term_mask=th.FloatTensor(term_mask),
                      polarity=polarity))
        return processed

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

        embed_ids = th.LongTensor(batch_size, max_len).fill_(self.vocabulary[self.pad_word])
        for i, b in enumerate(batch):
            embed_ids[i, :b.sentence_len] = th.LongTensor(b.embed_ids)

        return Batch(text_index=th.LongTensor([x.text_index for x in batch]),
                     sentence_index=th.LongTensor([x.sentence_index for x in batch]),
                     opinion_index=th.LongTensor([x.opinion_index for x in batch]),
                     sentence_len=th.LongTensor([x.sentence_len
                                                 for x in batch]).to(self.device),
                     embed_ids=embed_ids.to(self.device),
                     graph=dgl.batch([item.graph for item in batch]),
                     term_mask=th.cat([item.term_mask for item in batch]),
                     polarity=th.LongTensor([x.polarity for x in batch]))

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
