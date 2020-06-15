from itertools import chain
from typing import Tuple

import numpy as np
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence


class StackLSTM(nn.Module):
    """Stack of LSTM networks"""
    def __init__(self, layers_dim: np.array, bidirectional: bool, device: th.device):
        """
        Parameters
        ----------
        layers_dim : np.array
            array of dimensions
            np.array([in_1, out_1, out_2, ..., out_n])
        bidirectional : bool
            are LSTMs bi directionals
        device: th.device
            CPU or GPU
        """

        super(StackLSTM, self).__init__()
        self.device = device

        self.bidirectional = bidirectional
        self.layers_dim = layers_dim
        self.rnn = nn.ModuleList()
        for in_dim, out_dim in zip(
                chain([layers_dim[0]],
                      layers_dim[1:-1] * 2 if self.bidirectional else layers_dim[:-1]),
                layers_dim[1:]):
            self.rnn.append(
                nn.LSTM(input_size=in_dim,
                        hidden_size=out_dim,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=self.bidirectional))

    def forward(self,
                rnn_inputs: PackedSequence) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """Forward step

        Parameters
        ----------
        rnn_inputs : PackedSequence

        Returns
        -------
        output
        h_n
        c_n
        """

        batch_size = max(rnn_inputs.batch_sizes).item()
        for rnn_layer in self.rnn[:-1]:
            h0, c0 = self._init_hidden(batch_size=batch_size,
                                       hidden_dim=rnn_layer.hidden_size.item(),
                                       bidirectional=self.bidirectional)
            output, (_, _) = rnn_layer(rnn_inputs, (h0.to(self.device), c0.to(self.device)))
            rnn_inputs = output

        # pass through the last one layer
        h0, c0 = self._init_hidden(batch_size=batch_size,
                                   hidden_dim=self.rnn[-1].hidden_size.item(),
                                   bidirectional=self.bidirectional)
        output, (h_n, c_n) = self.rnn[-1](rnn_inputs, (h0.to(self.device), c0.to(self.device)))
        return output, (h_n, c_n)

    @staticmethod
    def _init_hidden(batch_size: int, hidden_dim: int,
                     bidirectional: bool) -> Tuple[th.Tensor, th.Tensor]:
        """Return zero state hidden and memory

        Parameters
        ----------
        batch_size : int
        hidden_dim : int
        bidirectional : bool

        Returns
        -------
        h_0 : th.Tensor
        c_0 : th.Tensor
        """

        state_shape = (2 if bidirectional else 1, batch_size, hidden_dim)
        h_0 = Variable(th.zeros(*state_shape, dtype=th.float32), requires_grad=False)
        c_0 = Variable(th.zeros(*state_shape, dtype=th.float32), requires_grad=False)
        return h_0, c_0
