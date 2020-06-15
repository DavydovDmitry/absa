from itertools import chain

import numpy as np
import torch as th
import torch.nn as nn
from torch.autograd import Variable


class LSTMClassifier(nn.Module):
    """Stack of LSTM networks"""
    def __init__(self,
                 embeddings,
                 device: th.device,
                 bidirectional: bool,
                 layers_dim: np.array,
                 input_dropout=0.7,
                 *args,
                 **kwargs):
        super(LSTMClassifier, self).__init__()
        self.device = device

        self.embed = nn.Embedding(*embeddings.shape)
        self.embed.weight = nn.Parameter(embeddings.to(self.device), requires_grad=False)

        self.bidirectional = bidirectional
        self.layers_dim = layers_dim
        self.rnn = nn.ModuleList()
        for in_dim, out_dim in zip(
                chain([self.embed.embedding_dim],
                      layers_dim[:-1] * 2 if self.bidirectional else layers_dim[:-1]),
                layers_dim):
            self.rnn.append(
                nn.LSTM(input_size=in_dim,
                        hidden_size=out_dim,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=self.bidirectional))

        self.in_drop = nn.Dropout(input_dropout)

    def forward(self, embed_ids: th.Tensor, sentence_len: th.Tensor) -> th.Tensor:
        embeds = self.embed(embed_ids)
        embeds = self.in_drop(embeds)

        rnn_inputs = nn.utils.rnn.pack_padded_sequence(input=embeds,
                                                       lengths=sentence_len,
                                                       batch_first=True)

        for layer_index, hidden_dim in enumerate(self.layers_dim[:-1]):
            h0, c0 = self._init_hidden(batch_size=embeds.size(0),
                                       hidden_dim=hidden_dim,
                                       bidirectional=self.bidirectional)
            output, (_, _) = self.rnn[layer_index](rnn_inputs,
                                                   (h0.to(self.device), c0.to(self.device)))
            rnn_inputs = output
        # last one layer
        h0, c0 = self._init_hidden(batch_size=embeds.size(0),
                                   hidden_dim=self.layers_dim[-1],
                                   bidirectional=self.bidirectional)
        _, (out, _) = self.rnn[-1](rnn_inputs, (h0.to(self.device), c0.to(self.device)))

        out = th.cat([x for x in out], dim=1)
        return out

    @staticmethod
    def _init_hidden(batch_size: int, hidden_dim: int, bidirectional: bool):
        state_shape = (2 if bidirectional else 1, batch_size, hidden_dim)
        h0 = Variable(th.zeros(*state_shape, dtype=th.float32), requires_grad=False)
        c0 = Variable(th.zeros(*state_shape, dtype=th.float32), requires_grad=False)
        return h0, c0
