from typing import Tuple

import torch as th
import torch.nn as nn
import dgl

from .nn import LSTMNN


class NNWrap(nn.Module):
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
        self.nn = LSTMNN(emb_matrix=emb_matrix,
                         device=self.device,
                         rnn_dim=rnn_dim,
                         out_dim=tree_lstm_dim).to(self.device)
        self.classifier = nn.Linear(tree_lstm_dim, num_class)

    def forward(self, embed_ids: th.Tensor, graph: dgl.DGLGraph,
                sentence_len: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Forward step.

        Return
        -------
        logits : th.Tensor
            For every class.
        outputs : th.Tensor
            Hidden state (encoding) for every target.
        """
        h = self.nn(embed_ids=embed_ids, graph=graph, sentence_len=sentence_len)
        logits = self.classifier(h)
        return logits
