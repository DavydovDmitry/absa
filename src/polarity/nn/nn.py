import torch as th
import torch.nn as nn
from torch.autograd import Variable
import dgl

from .tree_lstm import ChildSumTreeLSTMCell


class LSTMNN(nn.Module):
    def __init__(self,
                 emb_matrix: th.Tensor,
                 device: th.device,
                 rnn_dim=50,
                 rnn_layers=1,
                 out_dim=50,
                 input_dropout=0.8,
                 rnn_dropout=0.1,
                 bidirectional=True):
        super(LSTMNN, self).__init__()
        self.device = device

        self.embed = nn.Embedding(*emb_matrix.shape)
        self.embed.weight = nn.Parameter(emb_matrix.to(device), requires_grad=False)

        self.rnn_layers = rnn_layers
        self.rnn_hidden = rnn_dim
        self.mem_dim = out_dim
        self.bidirectional = True

        # rnn layer
        self.rnn = nn.LSTM(self.embed.embedding_dim,
                           rnn_dim,
                           rnn_layers,
                           batch_first=True,
                           dropout=rnn_dropout,
                           bidirectional=bidirectional)

        # tree lstm layer
        self.tree_lstm = ChildSumTreeLSTMCell(rnn_dim * 2 if bidirectional else rnn_dim,
                                              out_dim)
        self.tree_lstm_hidden = nn.Linear(rnn_dim * 2 if bidirectional else rnn_dim, out_dim)

        self.in_drop = nn.Dropout(input_dropout)
        self.rnn_drop = nn.Dropout(rnn_dropout)

    def forward(self, graph: dgl.DGLGraph, embed_ids: th.Tensor, sentence_len: th.Tensor):
        embeds = self.embed(embed_ids)
        embeds = self.in_drop(embeds)

        # rnn layer
        x = self.rnn_drop(
            self.encode_with_rnn(rnn_inputs=embeds,
                                 sentence_len=sentence_len,
                                 batch_size=sentence_len.shape[0]))

        # tree lstm layer
        g = graph
        g.register_message_func(self.tree_lstm.message_func)
        g.register_reduce_func(self.tree_lstm.reduce_func)
        g.register_apply_node_func(self.tree_lstm.apply_node_func)
        g.ndata['iou'] = self.tree_lstm.W_iou(x)
        g.ndata['h'] = self.tree_lstm_hidden(x)
        g.ndata['c'] = Variable(th.zeros((graph.number_of_nodes(), self.mem_dim),
                                         dtype=th.float32),
                                requires_grad=False).to(self.device)
        dgl.prop_nodes_topo(g)

        return g.ndata.pop('h')

    def encode_with_rnn(self, rnn_inputs: th.Tensor, sentence_len: th.Tensor, batch_size: int):
        h0, c0 = self.rnn_zero_state(batch_size=batch_size,
                                     hidden_dim=self.rnn_hidden,
                                     num_layers=self.rnn_layers,
                                     bidirectional=self.bidirectional)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(input=rnn_inputs,
                                                       lengths=sentence_len,
                                                       batch_first=True)
        rnn_outputs, (_, _) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        rnn_outputs = th.cat([rnn_outputs[i, :x] for i, x in enumerate(sentence_len)], dim=0)
        return rnn_outputs

    def rnn_zero_state(self, batch_size: int, hidden_dim: int, num_layers: int,
                       bidirectional: bool):
        total_layers = num_layers * 2 if bidirectional else num_layers
        state_shape = (total_layers, batch_size, hidden_dim)
        h0 = c0 = Variable(th.zeros(*state_shape, dtype=th.float32), requires_grad=False)
        return h0.to(self.device), c0.to(self.device)
