import numpy as np
import torch as th
import torch.nn as nn

from .lstm import LSTMClassifier


class NeuralNetwork(nn.Module):
    def __init__(self,
                 device: th.device,
                 embeddings: th.Tensor,
                 num_classes: int,
                 layers_dim: np.array,
                 bidirectional=True,
                 *args,
                 **kwargs):
        super().__init__()
        self.device = device
        self.nn = LSTMClassifier(embeddings=embeddings,
                                 device=self.device,
                                 bidirectional=bidirectional,
                                 layers_dim=layers_dim,
                                 *args,
                                 **kwargs)
        self.linear = nn.Linear(layers_dim[-1] * 2 if bidirectional else layers_dim[-1],
                                num_classes)

    def forward(self, embed_ids: th.Tensor, sentence_len: th.Tensor) -> th.Tensor:
        """Forward step

        Parameters
        ----------
        embed_ids : th.Tensor
        sentence_len: th.Tensor

        Returns
        -------
        logits : th.Tensor
        """

        h = self.nn(embed_ids=embed_ids, sentence_len=sentence_len)
        logits = self.linear(h)
        return logits
