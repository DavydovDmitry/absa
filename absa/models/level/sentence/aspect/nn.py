import numpy as np
import torch as th
import torch.nn as nn

from absa.models.nn.lstm import StackLSTM


class NeuralNetwork(nn.Module):
    def __init__(self,
                 device: th.device,
                 embeddings: th.Tensor,
                 num_classes: int,
                 layers_dim: np.array,
                 bidirectional=True,
                 input_dropout=0.7,
                 *args,
                 **kwargs):
        super().__init__()
        self.device = device

        self.embeddings = nn.Embedding(*embeddings.shape)
        self.embeddings.weight = nn.Parameter(embeddings.to(self.device), requires_grad=False)
        self.in_drop = nn.Dropout(input_dropout)

        self.lstm = StackLSTM(layers_dim=np.concatenate(
            (np.array([self.embeddings.embedding_dim]), layers_dim)),
                              bidirectional=bidirectional,
                              device=self.device)
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

        embeds = self.embeddings(embed_ids)
        embeds = self.in_drop(embeds)

        rnn_inputs = nn.utils.rnn.pack_padded_sequence(input=embeds,
                                                       lengths=sentence_len,
                                                       batch_first=True)
        _, (h_n, _) = self.lstm(rnn_inputs=rnn_inputs)

        h = th.cat([x for x in h_n], dim=1)
        logits = self.linear(h)
        return logits
