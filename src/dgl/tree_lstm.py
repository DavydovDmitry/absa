import torch as th
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import dgl

from .dataset import BatchItem


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size: int, h_size: int):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size), requires_grad=True)
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = th.sum(nodes.mailbox['h'], 1)
        f = th.sigmoid(self.U_f(nodes.mailbox['h']))
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild) + nodes.data['iou'], 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}


class TreeLSTM(nn.Module):
    def __init__(
        self,
        embeddings: th.Tensor,
        num_classes: int,
        bi_lstm_hidden_size: int,
        tree_lstm_hidden_size: int,
        num_layers: int,
        dropout: float,
        device,
        bidirectional=True,
    ):
        super(TreeLSTM, self).__init__()
        self.x_size = embeddings.shape[1]
        self.bi_lstm_hidden_size = bi_lstm_hidden_size
        self.tree_lstm_hidden_size = tree_lstm_hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = device

        # set pre-trained word embeddings and lock it.
        self.embedding = nn.Embedding(*embeddings.shape)
        self.embedding.from_pretrained(embeddings)
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(input_size=self.x_size,
                            hidden_size=self.bi_lstm_hidden_size,
                            bidirectional=self.bidirectional,
                            num_layers=self.num_layers,
                            dropout=dropout)
        out_size = bi_lstm_hidden_size * 2 if bidirectional else bi_lstm_hidden_size
        self.bi_lstm_dropout = th.nn.Dropout(dropout)
        self.tree_lstm = ChildSumTreeLSTMCell(out_size, tree_lstm_hidden_size)
        self.tree_lstm_dropout = th.nn.Dropout(dropout)
        self.linear_1 = nn.Linear(tree_lstm_hidden_size, tree_lstm_hidden_size)
        self.linear_2 = nn.Linear(tree_lstm_hidden_size, num_classes)

    def forward(self, batch: BatchItem):
        batch_first = False
        batch_size = batch.sentence_len.size(0)

        # bi lstm
        embeds = self.embedding(batch.embed_ids)
        packed = pack_padded_sequence(embeds,
                                      batch.sentence_len,
                                      batch_first=batch_first,
                                      enforce_sorted=False)
        x, (h_t, c_t) = self.lstm(packed, (self._init_bi_lstm(batch_size=batch_size)))
        x = x.data

        # tree lstm
        x = self.bi_lstm_dropout(x)
        g = batch.graph
        g.register_message_func(self.tree_lstm.message_func)
        g.register_reduce_func(self.tree_lstm.reduce_func)
        g.register_apply_node_func(self.tree_lstm.apply_node_func)
        g.ndata['iou'] = self.tree_lstm.W_iou(x)
        g.ndata['h'], g.ndata['c'] = self._init_tree_lstm(
            number_of_nodes=batch.graph.number_of_nodes(),
            tree_lstm_hidden_size=self.bi_lstm_hidden_size)
        dgl.prop_nodes_topo(g)
        x = self.tree_lstm_dropout(g.ndata.pop('h'))

        # select aspect words, avg and return predictions
        x = x[batch.target_mask.nonzero().squeeze(1)]
        x = th.relu(th.mm(batch.target_matrix, x))
        x = th.relu(self.linear_1(x))
        logits = th.relu(self.linear_2(x))
        return logits

    def _init_bi_lstm(self, batch_size: int):
        h_0 = th.zeros(size=(self.num_layers * (2 if self.bidirectional else 1), batch_size,
                             self.bi_lstm_hidden_size)).to(self.device)
        c_0 = th.zeros(size=(self.num_layers * (2 if self.bidirectional else 1), batch_size,
                             self.bi_lstm_hidden_size)).to(self.device)
        return h_0, c_0

    def _init_tree_lstm(self, number_of_nodes, tree_lstm_hidden_size):
        h_0 = th.zeros((number_of_nodes, tree_lstm_hidden_size)).to(self.device)
        c_0 = th.zeros((number_of_nodes, tree_lstm_hidden_size)).to(self.device)
        return h_0, c_0
