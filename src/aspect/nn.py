from typing import Tuple

import torch as th
import torch.nn as nn
import dgl

from ..lstm.stack_lstm import StackLSTM


class NeuralNetwork(nn.Module):
    """Targets polarity classifier

    Essential stages:
    - Encode sequence elements with neural networks.
    - Get encodings for targets.
    - Map encodings to num_classes
    """
    def __init__(self,
                 emb_matrix: th.Tensor,
                 device: th.device,
                 num_class: int,
                 rnn_dim=50,
                 tree_lstm_dim=30):
        super().__init__()
        self.device = device
        self.nn = StackLSTM(emb_matrix=emb_matrix,
                            device=self.device,
                            rnn_dim=rnn_dim,
                            out_dim=tree_lstm_dim)
        self.linear = nn.Linear(tree_lstm_dim, num_class)

    def forward(self, embed_ids: th.Tensor, graph: dgl.DGLGraph,
                sentence_len: th.Tensor) -> th.Tensor:
        """Forward step.

        Return
        -------
        logits : th.Tensor(device=self.device)
            For every class
        """
        h = self.nn(embed_ids=embed_ids, graph=graph, sentence_len=sentence_len)
        logits = self.linear(h)
        return logits
