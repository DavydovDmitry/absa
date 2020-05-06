import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import dgl


class GCNClassifier(nn.Module):
    def __init__(self, emb_matrix: th.Tensor, device: th.device, num_class, mem_dim=50):
        super().__init__()
        self.device = device
        self.gcn = GCN(emb_matrix=emb_matrix, device=self.device)
        self.classifier = nn.Linear(mem_dim, num_class)

    def forward(self, embed_ids: th.Tensor, graph: dgl.DGLGraph, target_mask: th.Tensor,
                sentence_len: th.Tensor):
        h = self.gcn(embed_ids=embed_ids, graph=graph, sentence_len=sentence_len)

        # get aspect
        h = h[target_mask.nonzero().squeeze(1)]
        target_lens = [
            int(x.sum().item()) for x in target_mask.split([l for l in sentence_len], dim=0)
        ]
        outputs = th.stack([x.sum(dim=0) / x.size(0) for x in h.split(target_lens, dim=0)])
        # outputs = th.cat(
        #     [x.sum(dim=0) / x.size(0) for x in h.split([l for l in sentence_len], dim=0)])
        # outputs = th.relu(th.mm(target_matrix, h))

        # asp_wn = mask.sum(dim=1).unsqueeze(-1)  # aspect words num
        # mask = mask.unsqueeze(-1).repeat(1, 1, self.gcn.rnn_hidden)  # mask for h
        # outputs = (h * mask).sum(dim=1) / asp_wn  # mask h

        # make predictions for each class
        logits = self.classifier(outputs)
        return logits, outputs


class GCN(nn.Module):
    def __init__(self,
                 emb_matrix: th.Tensor,
                 device: th.device,
                 rnn_hidden=50,
                 rnn_layers=1,
                 mem_dim=50,
                 num_layers=2,
                 input_dropout=0.8,
                 rnn_dropout=0.1,
                 gcn_dropout=0.1,
                 bidirectional=True):
        super(GCN, self).__init__()
        self.device = device

        self.embed = nn.Embedding(*emb_matrix.shape)
        self.embed.weight = nn.Parameter(emb_matrix.to(device), requires_grad=False)

        self.rnn_layers = rnn_layers
        self.rnn_hidden = rnn_hidden
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.bidirectional = True

        # rnn layer
        self.rnn = nn.LSTM(self.embed.embedding_dim,
                           rnn_hidden,
                           rnn_layers,
                           batch_first=True,
                           dropout=rnn_dropout,
                           bidirectional=bidirectional)

        self.tree_lstm = ChildSumTreeLSTMCell(rnn_hidden * 2 if bidirectional else rnn_hidden,
                                              mem_dim)
        self.tree_lstm_hidden = nn.Linear(rnn_hidden * 2 if bidirectional else rnn_hidden,
                                          mem_dim)

        self.in_drop = nn.Dropout(input_dropout)
        self.rnn_drop = nn.Dropout(rnn_dropout)
        self.gcn_drop = nn.Dropout(gcn_dropout)

    def forward(self, graph: dgl.DGLGraph, embed_ids: th.Tensor, sentence_len: th.Tensor):
        embeds = self.embed(embed_ids)
        embeds = self.in_drop(embeds)

        # bi lstm layer
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
        x = self.gcn_drop(g.ndata.pop('h'))

        return x

    def encode_with_rnn(self, rnn_inputs: th.Tensor, sentence_len: th.Tensor, batch_size: int):
        h0, c0 = self.rnn_zero_state(batch_size=batch_size,
                                     hidden_dim=self.rnn_hidden,
                                     num_layers=self.rnn_layers,
                                     bidirectional=self.bidirectional)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(input=rnn_inputs,
                                                       lengths=sentence_len,
                                                       batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        rnn_outputs = th.cat([rnn_outputs[i, :x] for i, x in enumerate(sentence_len)], dim=0)
        return rnn_outputs

    def rnn_zero_state(self, batch_size: int, hidden_dim: int, num_layers: int,
                       bidirectional: bool):
        total_layers = num_layers * 2 if bidirectional else num_layers
        state_shape = (total_layers, batch_size, hidden_dim)
        h0 = c0 = Variable(th.zeros(*state_shape, dtype=th.float32), requires_grad=False)
        return h0.to(self.device), c0.to(self.device)


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
        h = nodes.data['h'] + o * th.tanh(c)
        return {'h': h, 'c': c}
