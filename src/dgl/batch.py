from collections import namedtuple

import torch as th
from torch.nn.utils.rnn import pad_sequence
import dgl

from src import PAD_WORD

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


def collate_batch(device):
    def internal_collate(batch):
        target_lens = [sum(item.target_mask.values()) for item in batch]
        target_matrix = th.zeros(len(target_lens), sum(target_lens))
        col_ix = 0
        for r_ix, c_num in enumerate(target_lens):
            target_matrix[r_ix, col_ix:col_ix + c_num] = 1.0 / c_num
            col_ix += c_num

        embed_ids = [th.tensor(x.embed_ids) for x in batch]
        embed_ids = pad_sequence(embed_ids, batch_first=False, padding_value=)

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
