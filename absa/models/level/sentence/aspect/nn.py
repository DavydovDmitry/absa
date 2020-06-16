import numpy as np
import torch as th
import torch.nn as nn

from absa.models.nn.lstm import StackLSTM


class NeuralNetwork(nn.Module):
    def __init__(self,
                 embeddings: th.Tensor,
                 num_classes: int,
                 layers_dim: np.array,
                 emb_dropout: float,
                 device: th.device,
                 bidirectional: bool = True,
                 *args,
                 **kwargs):
        """

        embeddings : th.Tensor
            pretrained embeddings
        num_classes : int
            number of prediction classes
        layers_dim: np.array
            dimensions of LSTM layers
        device : th.device
            CPU or GPU
        bidirectional : bool
            is all layers bidirectional or not
        emb_dropout : float
            probability of dropout layer after embeddings
        """

        super().__init__()
        self.device = device

        self.embeddings = nn.Embedding(*embeddings.shape)
        self.embeddings.weight = nn.Parameter(embeddings.to(self.device), requires_grad=False)
        self.emb_dropout = nn.Dropout(emb_dropout)

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
        embeds = self.emb_dropout(embeds)

        rnn_inputs = nn.utils.rnn.pack_padded_sequence(input=embeds,
                                                       lengths=sentence_len,
                                                       batch_first=True)
        _, (h_n, _) = self.lstm(rnn_inputs=rnn_inputs)

        h = th.cat([x for x in h_n], dim=1)
        logits = self.linear(h)
        return logits
