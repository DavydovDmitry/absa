import torch as th
import torch.nn as nn

from .lstm import LSTMClassifier


class NeuralNetwork(nn.Module):
    """Targets polarity classifier

    Essential stages:
    - Encode sequence elements with neural networks.
    - Get encodings for targets.
    - Map encodings to num_classes
    """
    def __init__(
        self,
        emb_matrix: th.Tensor,
        device: th.device,
        num_class: int,
        rnn_dim=40,
        bidirectional=True,
    ):
        super().__init__()
        self.device = device
        self.nn = LSTMClassifier(emb_matrix=emb_matrix,
                                 device=self.device,
                                 rnn_dim=rnn_dim,
                                 bidirectional=bidirectional)
        rnn_dim = rnn_dim * 2 if bidirectional else rnn_dim
        self.linear = nn.Linear(rnn_dim, num_class)

    def forward(self, embed_ids: th.Tensor, sentence_len: th.Tensor) -> th.Tensor:
        """Forward step.

        Return
        -------
        logits : th.Tensor(device=self.device)
            For every class
        """
        h = self.nn(embed_ids=embed_ids, sentence_len=sentence_len)
        logits = self.linear(h)
        return logits
