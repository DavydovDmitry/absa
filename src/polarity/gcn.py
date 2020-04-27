import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.review.target import Polarity


class GCNClassifier(nn.Module):
    def __init__(self,
                 emb_matrix: th.Tensor,
                 device: th.device,
                 emb_matrix_shape,
                 num_class=len(Polarity),
                 mem_dim=50):
        super().__init__()
        self.device = device
        self.gcn = GCN(emb_matrix=emb_matrix,
                       device=self.device,
                       emb_matrix_shape=emb_matrix_shape)
        self.classifier = nn.Linear(mem_dim, num_class)

    def forward(self, embed_ids: th.Tensor, adj: th.Tensor, mask: th.Tensor,
                sentence_len: th.Tensor):
        h = self.gcn(embed_ids=embed_ids, adj=adj, sentence_len=sentence_len)

        # get aspect
        asp_wn = mask.sum(dim=1).unsqueeze(-1)  # aspect words num
        mask = mask.unsqueeze(-1).repeat(1, 1, self.gcn.rnn_hidden)  # mask for h
        outputs = (h * mask).sum(dim=1) / asp_wn  # mask h

        # make predictions for each class
        logits = self.classifier(outputs)
        return logits, outputs


class GCN(nn.Module):
    def __init__(
        self,
        emb_matrix: th.Tensor,
        emb_matrix_shape,
        device: th.device,
        rnn_hidden=50,
        rnn_layers=1,
        mem_dim=50,
        num_layers=2,
        input_dropout=0.8,
        rnn_dropout=0.1,
        gcn_dropout=0.1,
    ):
        super(GCN, self).__init__()
        self.device = device

        if emb_matrix is not None:
            self.embed = nn.Embedding(*emb_matrix.shape)
            self.embed.weight = nn.Parameter(emb_matrix.cuda(), requires_grad=False)
        else:
            self.embed = nn.Embedding(*emb_matrix_shape)

        self.rnn_layers = rnn_layers
        self.rnn_hidden = rnn_hidden
        self.layers = num_layers
        self.mem_dim = mem_dim

        # rnn layer
        self.rnn = nn.LSTM(self.embed.embedding_dim,
                           rnn_hidden,
                           rnn_layers,
                           batch_first=True,
                           dropout=rnn_dropout,
                           bidirectional=True)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = rnn_hidden * 2 if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.in_drop = nn.Dropout(input_dropout)
        self.rnn_drop = nn.Dropout(rnn_dropout)
        self.gcn_drop = nn.Dropout(gcn_dropout)

    def forward(self, adj: th.Tensor, embed_ids: th.Tensor, sentence_len: th.Tensor):
        embeds = self.embed(embed_ids)
        embeds = self.in_drop(embeds)

        # rnn layer
        gcn_inputs = self.rnn_drop(
            self.encode_with_rnn(rnn_inputs=embeds,
                                 sentence_len=sentence_len,
                                 batch_size=sentence_len.shape[0]))

        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1  # norm
        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return gcn_inputs

    def encode_with_rnn(self, rnn_inputs: th.Tensor, sentence_len: th.Tensor, batch_size: int):
        h0, c0 = self.rnn_zero_state(batch_size=batch_size,
                                     hidden_dim=self.rnn_hidden,
                                     num_layers=self.rnn_layers)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(input=rnn_inputs,
                                                       lengths=sentence_len,
                                                       batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def rnn_zero_state(self,
                       batch_size: int,
                       hidden_dim: int,
                       num_layers: int,
                       bidirectional=True):
        total_layers = num_layers * 2 if bidirectional else num_layers
        state_shape = (total_layers, batch_size, hidden_dim)
        h0 = c0 = Variable(th.zeros(*state_shape, dtype=th.float32), requires_grad=False)
        return h0.to(self.device), c0.to(self.device)
