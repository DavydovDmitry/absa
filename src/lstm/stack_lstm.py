from typing import Tuple

import torch as th
import torch.nn as nn
from torch.autograd import Variable
import dgl

from .tree_lstm import ChildSumTreeLSTMCell


class StackLSTM(nn.Module):
    """Stack of Bi-LSTM and Tree-LSTM neural networks.

    First one run through sequence elements order. The last one run
    through dependency tree of sentence.
    """
    def __init__(self,
                 emb_matrix: th.Tensor,
                 device: th.device,
                 rnn_dim: int,
                 out_dim: int,
                 rnn_layers=1,
                 input_dropout=0.7,
                 rnn_dropout=0.1):
        super(StackLSTM, self).__init__()
        self.device = device

        self.embed = nn.Embedding(*emb_matrix.shape)
        self.embed.weight = nn.Parameter(emb_matrix.to(device), requires_grad=False)

        self.rnn_hidden = rnn_dim  # rnn out dim
        self.rnn_layers = rnn_layers  # number of lstm layers
        self.mem_dim = out_dim  # tree lstm out dim
        self.bidirectional = True  # use bi lstm

        # rnn layer
        self.rnn = nn.LSTM(input_size=self.embed.embedding_dim,
                           hidden_size=rnn_dim,
                           num_layers=self.rnn_layers,
                           batch_first=True,
                           bidirectional=self.bidirectional)

        # tree lstm layer
        input_dim = rnn_dim * 2 if self.bidirectional else rnn_dim
        self.tree_lstm = ChildSumTreeLSTMCell(input_dim, out_dim)
        self.tree_lstm_hidden = nn.Linear(input_dim, out_dim)

        self.in_drop = nn.Dropout(input_dropout)
        self.rnn_drop = nn.Dropout(rnn_dropout)

    def forward(self, graph: dgl.DGLGraph, embed_ids: th.Tensor,
                sentence_len: th.Tensor) -> th.Tensor:
        embeds = self.embed(embed_ids)
        embeds = self.in_drop(embeds)

        # rnn layer
        h = self.rnn_drop(self.encode_with_rnn(rnn_inputs=embeds, sentence_len=sentence_len))

        # tree lstm layer
        g = graph
        g.register_message_func(self.tree_lstm.message_func)
        g.register_reduce_func(self.tree_lstm.reduce_func)
        g.register_apply_node_func(self.tree_lstm.apply_node_func)
        g.ndata['iou'] = self.tree_lstm.W_iou(h)
        g.ndata['h'] = self.tree_lstm_hidden(h)
        g.ndata['c'] = Variable(th.zeros((graph.number_of_nodes(), self.mem_dim),
                                         dtype=th.float32),
                                requires_grad=False).to(self.device)
        dgl.prop_nodes_topo(g)

        return g.ndata.pop('h')

    def encode_with_rnn(self, rnn_inputs: th.Tensor, sentence_len: th.Tensor) -> th.Tensor:
        """Encode batch with LSTM
        Can highlight such stages:
        - pack batch sequence
        - encode packed sequences with LSTM
        - unpack sequence
        - concatenate to one dimension all sequences

        Parameters
        ----------
        rnn_inputs : th.Tensor
            Tensor with sequence elements embeddings. Tensor size = [B x T x E],
            where B - batch size, T - length of longest sequence, E - embedding
            dimension.
        sentence_len : th.Tensor
            Lengths of sequences. Size = [B].

        Returns
        -------
        rnn_outputs : th.Tensor
            Hidden states of sequence elements. Sequences stacked together.
            Size = [* x E], where * - sum of lengths of sequences (true lengths
            without padding)
        """
        def rnn_zero_state() -> Tuple[th.Tensor, th.Tensor]:
            total_layers = self.rnn_layers * 2 if self.bidirectional else self.num_layers
            state_shape = (total_layers, rnn_inputs.size(0), self.rnn_hidden)
            h0 = Variable(th.zeros(*state_shape, dtype=th.float32), requires_grad=False)
            c0 = Variable(th.zeros(*state_shape, dtype=th.float32), requires_grad=False)
            return h0.to(self.device), c0.to(self.device)

        h0, c0 = rnn_zero_state()
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(input=rnn_inputs,
                                                       lengths=sentence_len,
                                                       batch_first=True)
        rnn_outputs, (_, _) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        rnn_outputs = th.cat([rnn_outputs[i, :x] for i, x in enumerate(sentence_len)], dim=0)
        return rnn_outputs
