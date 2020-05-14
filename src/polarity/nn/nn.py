from typing import Tuple

import torch as th
import torch.nn as nn
import dgl

from src.polarity.nn.stack_lstm import StackLSTM


class TargetClassifier(nn.Module):
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
                            out_dim=tree_lstm_dim).to(self.device)
        self.classifier = nn.Linear(tree_lstm_dim, num_class)

    def forward(self, embed_ids: th.Tensor, graph: dgl.DGLGraph, target_mask: th.Tensor,
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

        # get hidden state for every target
        h = h[target_mask.nonzero().squeeze(1)].cpu()
        target_lens = [
            int(x.sum().item()) for x in target_mask.split([l for l in sentence_len], dim=0)
        ]
        outputs = th.stack([x.sum(dim=0) / x.size(0) for x in h.split(target_lens, dim=0)])

        # make predictions for every class
        logits = self.classifier(outputs)
        return logits, outputs
