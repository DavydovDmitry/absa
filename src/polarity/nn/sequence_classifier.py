import torch as th
import torch.nn as nn
import dgl

from .nn import LSTMNN


class SequenceClassifier(nn.Module):
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

    def forward(self, embed_ids: th.Tensor, graph: dgl.DGLGraph, target_mask: th.Tensor,
                sentence_len: th.Tensor):
        h = self.nn(embed_ids=embed_ids, graph=graph, sentence_len=sentence_len)

        # get hidden state for each aspect
        h = h[target_mask.nonzero().squeeze(1)].cpu()
        target_lens = [
            int(x.sum().item()) for x in target_mask.split([l for l in sentence_len], dim=0)
        ]
        outputs = th.stack([x.sum(dim=0) / x.size(0) for x in h.split(target_lens, dim=0)])

        # make predictions for each class
        logits = self.classifier(outputs)
        return logits, outputs
