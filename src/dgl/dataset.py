from typing import List
from collections import namedtuple

from gensim.models import KeyedVectors
from torch.utils.data import Dataset
import networkx as nx
import dgl
import torch as th
from torch.nn.utils.rnn import pad_sequence

from src import UNKNOWN_WORD, PAD_WORD
from src.review import ParsedSentence

DatasetItem = namedtuple(
    'DatasetItem',
    [
        'sentence_index',
        'target_index',
        # graph attr
        'graph',
        'target_mask',
        'polarity',
        # tensors
        'embed_ids',
        'sentence_len'
    ])

BatchItem = namedtuple(
    'BatchItem',
    [
        'sentence_index',
        'target_index',

        # graph attr
        'graph',
        'target_mask',
        'node_index',

        # tensors
        'embed_ids',
        'target_matrix',
        'sentence_len',
        'polarity',
    ])


class ABSADataset(Dataset):
    def __init__(self, sentences: List[ParsedSentence], word2vec: KeyedVectors, batch_size=32):
        self.dataset = []
        self.word2vec = word2vec
        self.batch_size = batch_size

        for sentence_index, sentence in enumerate(sentences):
            self.dataset.extend(
                self._process_sentence(sentence=sentence, sentence_index=sentence_index))

    def _process_sentence(self, sentence: ParsedSentence,
                          sentence_index: int) -> List[DatasetItem]:
        batch = []

        if sentence.graph.nodes and sentence.targets:

            # get word embedding indexes
            embed_ids = []
            for node_id, _ in sorted(sentence.id2prev_id.items(), key=lambda item: item[1]):
                if node_id in sentence.id2lemma:
                    if sentence.id2lemma[node_id] in self.word2vec.vocab:
                        emd_id = self.word2vec.vocab[sentence.id2lemma[node_id]].index
                    else:
                        emd_id = self.word2vec.vocab[UNKNOWN_WORD].index
                    embed_ids.append(emd_id)

            node_indexes = {node: node for node in sentence.graph}

            # get target mask
            for target_index, target in enumerate(sentence.targets):
                target_nodes = target.nodes

                if not target_nodes:
                    target_nodes = [
                        node
                        for node in sentence.graph  # if sentence.graph.out_degree(node) == 0
                    ]
                target_mask = dict.fromkeys(sentence.graph.nodes, 0)
                polarity = dict.fromkeys(sentence.graph.nodes, 0)
                for node_index in target_nodes:
                    target_mask[node_index] = 1
                #     polarity[node_index] = POLARITIES[target.polarity].value

                graph = dgl.DGLGraph()
                nx.set_node_attributes(G=sentence.graph,
                                       name='target_mask',
                                       values=target_mask)
                nx.set_node_attributes(G=sentence.graph, name='polarity', values=polarity)
                nx.set_node_attributes(G=sentence.graph,
                                       name='node_index',
                                       values=node_indexes)
                graph.from_networkx(sentence.graph, node_attrs=['target_mask', 'node_index'])
                sentence_len = len(sentence.id2lemma)
                polarity = target.polarity.value
                batch.append(
                    DatasetItem(
                        sentence_index=sentence_index,
                        target_index=target_index,
                        graph=graph,
                        embed_ids=embed_ids,
                        sentence_len=sentence_len,
                        polarity=polarity,
                        target_mask=target_mask,
                    ))
        return batch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> DatasetItem:
        return self.dataset[index]

    def collate_batch(self, device):
        def internal_collate(batch):
            target_lens = [sum(item.target_mask.values()) for item in batch]
            target_matrix = th.zeros(len(target_lens), sum(target_lens))
            col_ix = 0
            for r_ix, c_num in enumerate(target_lens):
                target_matrix[r_ix, col_ix:col_ix + c_num] = 1.0 / c_num
                col_ix += c_num

            embed_ids = [th.tensor(x.embed_ids) for x in batch]
            embed_ids = pad_sequence(embed_ids,
                                     batch_first=False,
                                     padding_value=self.word2vec.vocab[PAD_WORD].index)

            sentence_len = th.tensor([item.sentence_len for item in batch])
            # polarity = []
            # for p, t_len in zip([item.polarity for item in batch], target_lens):
            #     polarity.extend([p] * t_len)
            # polarity = th.tensor(polarity)
            polarity = th.tensor(data=[item.polarity for item in batch], dtype=th.int64)
            batch_trees = dgl.batch([item.graph for item in batch])
            return BatchItem(
                sentence_index=[item.sentence_index for item in batch],
                target_index=[item.target_index for item in batch],
                # graph
                graph=batch_trees,
                target_mask=batch_trees.ndata['target_mask'].to(device),
                node_index=batch_trees.ndata['node_index'].to(device),
                # tensor
                target_matrix=target_matrix.to(device),
                polarity=polarity.to(device),
                embed_ids=embed_ids.to(device),
                sentence_len=sentence_len.to(device),
            )

        return internal_collate
