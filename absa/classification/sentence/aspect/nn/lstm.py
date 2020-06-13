from typing import Tuple

import torch as th
import torch.nn as nn
from torch.autograd import Variable


class LSTMClassifier(nn.Module):
    """Bi LSTM

    First one run through sequence elements order. The last one run
    through dependency tree of sentence.
    """
    def __init__(self,
                 emb_matrix: th.Tensor,
                 device: th.device,
                 rnn_dim: int,
                 bidirectional: bool,
                 rnn_layers=1,
                 input_dropout=0.7,
                 *args,
                 **kwargs):
        super(LSTMClassifier, self).__init__()
        self.device = device

        self.embed = nn.Embedding(*emb_matrix.shape)
        self.embed.weight = nn.Parameter(emb_matrix.to(device), requires_grad=False)

        self.rnn_hidden = rnn_dim  # rnn out dim
        self.rnn_layers = rnn_layers  # number of lstm layers
        self.bidirectional = bidirectional

        # rnn layer
        self.rnn = nn.LSTM(input_size=self.embed.embedding_dim,
                           hidden_size=rnn_dim,
                           num_layers=self.rnn_layers,
                           batch_first=True,
                           bidirectional=self.bidirectional)

        self.in_drop = nn.Dropout(input_dropout)

    def forward(self, embed_ids: th.Tensor, sentence_len: th.Tensor) -> th.Tensor:
        embeds = self.embed(embed_ids)
        embeds = self.in_drop(embeds)

        rnn_inputs = nn.utils.rnn.pack_padded_sequence(input=embeds,
                                                       lengths=sentence_len,
                                                       batch_first=True)

        total_layers = self.rnn_layers * 2 if self.bidirectional else self.num_layers
        state_shape = (total_layers, embeds.size(0), self.rnn_hidden)
        h0 = Variable(th.zeros(*state_shape, dtype=th.float32),
                      requires_grad=False).to(self.device)
        c0 = Variable(th.zeros(*state_shape, dtype=th.float32),
                      requires_grad=False).to(self.device)

        _, (out, _) = self.rnn(rnn_inputs, (h0, c0))
        out = th.cat([x for x in out], dim=1)
        return out
