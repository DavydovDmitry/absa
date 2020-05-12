from typing import List, Dict
import logging
import time
import copy
from functools import reduce

import torch as th
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from gensim.models import KeyedVectors
from frozendict import frozendict

from src import SCORE_DECIMAL_LEN, polarity_classifier_dump_path
from src.review.parsed_sentence import ParsedSentence
from src.review.target import Polarity
from .loader import DataLoader, Batch
from .nn.target_classifier import TargetClassifier
from .score.display import display_score


class PolarityClassifier:
    """Polarity classifier

    Attributes
    ----------
    word2vec : KeyedVectors
        Vocabulary and embed_matrix are extracting from word2vec.
        Otherwise you can pass vocabulary and emb_matrix.
    vocabulary : dict
        dictionary where key is wordlemma_POS value - index in embedding matrix
    emb_matrix : th.Tensor
        matrix of pretrained embeddings
    """
    def __init__(self,
                 word2vec: KeyedVectors = ...,
                 vocabulary: Dict[str, int] = ...,
                 emb_matrix: th.Tensor = ...,
                 batch_size=100):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        # prepare vocabulary and embeddings
        if isinstance(word2vec, KeyedVectors):
            self.vocabulary = {w: i for i, w in enumerate(word2vec.index2word)}
            emb_matrix = th.FloatTensor(word2vec.vectors)
        else:
            self.vocabulary = vocabulary

        while True:
            try:
                self.model = TargetClassifier(emb_matrix=emb_matrix,
                                              device=self.device,
                                              num_class=len(Polarity))
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    # todo: handle
                    # for p in self.model.parameters():
                    #     if p.grad is not None:
                    #         del p.grad
                    th.cuda.empty_cache()
            else:
                break

    def batch_metrics(self, batch: Batch):
        """Make a forward step and return metrics"""
        logits, gcn_outputs = self.model(embed_ids=batch.embed_ids,
                                         graph=batch.graph,
                                         target_mask=batch.target_mask,
                                         sentence_len=batch.sentence_len)
        loss = F.cross_entropy(logits, batch.polarity, reduction='mean')
        corrects = (th.max(logits, 1)[1].view(
            batch.polarity.size()).data == batch.polarity.data).sum()
        acc = np.float(corrects) / batch.polarity.size()[0]
        return logits, loss, acc

    def fit(self,
            train_sentences: List[ParsedSentence],
            val_sentences=None,
            optimizer_class=th.optim.Adamax,
            optimizer_params=frozendict({
                'lr': 0.01,
                'weight_decay': 0,
            }),
            num_epoch=30,
            verbose=True):
        """Fit on train sentences and save model state."""
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if verbose:
            logging.info(
                f'Number of parameters: {sum((reduce(lambda x, y: x * y, p.shape)) for p in parameters)}'
            )
        optimizer = optimizer_class(parameters, **optimizer_params)

        train_batches = DataLoader(sentences=train_sentences,
                                   batch_size=self.batch_size,
                                   vocabulary=self.vocabulary,
                                   device=self.device)
        if val_sentences:
            val_batches = DataLoader(sentences=val_sentences,
                                     batch_size=self.batch_size,
                                     vocabulary=self.vocabulary,
                                     device=self.device)
            val_acc_history, val_loss_history, f1_history = [], [], []

        train_acc_history, train_loss_history = [], []

        for epoch in range(num_epoch):
            # Train
            start_time = time.process_time()
            train_len = len(train_batches)
            self.model.train()
            train_loss, train_acc = 0., 0.
            for i, batch in enumerate(train_batches):
                optimizer.zero_grad()
                _, loss, acc = self.batch_metrics(batch=batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.data
                train_acc += acc

            # Validation
            if val_sentences:
                val_len = len(val_batches)
                self.model.eval()
                predictions, labels = [], []
                val_loss, val_acc = 0., 0.
                for i, batch in enumerate(val_batches):
                    logits, loss, acc = self.batch_metrics(batch=batch)
                    val_loss += loss.data
                    val_acc += acc
                    predictions += np.argmax(logits.data.numpy(), axis=1).tolist()
                    labels += batch.polarity.data.numpy().tolist()
                f1_score = metrics.f1_score(labels, predictions, average='macro')

                train_loss = train_loss / train_len
                train_acc = train_acc / train_len
                val_loss = val_loss / val_len
                val_acc = val_acc / val_len
                val_loss_history.append(val_loss)
                val_acc_history.append(val_acc)
                f1_history.append(f1_score)

                if verbose:
                    logging.info('-' * 40 + f' Epoch {epoch:03d} ' + '-' * 40)
                    logging.info(
                        f'Elapsed time: {(time.process_time() - start_time):.{3}f} sec')
                    logging.info(f'Train      ' +
                                 f'loss: {train_loss:.{SCORE_DECIMAL_LEN}f}| ' +
                                 f'acc: {train_acc:.{SCORE_DECIMAL_LEN}f}')
                    logging.info(f'Validation ' +
                                 f'loss: {(val_loss/val_len):.{SCORE_DECIMAL_LEN}f}| ' +
                                 f'acc: {val_acc:.{SCORE_DECIMAL_LEN}f}| ' +
                                 f'f1_score: {f1_score:.{SCORE_DECIMAL_LEN}f}')

            train_acc_history.append(train_acc)
            train_loss_history.append(train_loss)

        # display_score(parameter_values=[x for x in range(num_epoch)],
        #               train_values=train_acc_history,
        #               val_values=val_acc_history)
        self.save_model()
        if val_sentences:
            return train_acc_history, val_acc_history
        return train_acc_history

    def predict(self, sentences: List[ParsedSentence]) -> List[ParsedSentence]:
        """Modify passed sentences. Define every target polarity.

        Parameters
        ----------
        sentences : List[ParsedSentence]
            Sentences with extracted targets.

        Return
        -------
        sentences : List[ParsedSentence]
            Sentences with defined polarity of every target.
        """
        self.model.eval()
        sentences = copy.deepcopy(sentences)
        batches = DataLoader(sentences=sentences,
                             batch_size=self.batch_size,
                             vocabulary=self.vocabulary,
                             device=self.device)
        for batch_index, batch in enumerate(batches):
            logits, gcn_outputs = self.model(embed_ids=batch.embed_ids,
                                             graph=batch.graph,
                                             target_mask=batch.target_mask,
                                             sentence_len=batch.sentence_len)
            pred_labels = np.argmax(logits.data.numpy(), axis=1).tolist()
            for item_index, item in enumerate(pred_labels):
                sentence_index = batch.sentence_index[item_index]
                target_index = batch.target_index[item_index]
                sentences[sentence_index].targets[target_index].set_polarity(item)
        return sentences

    def save_model(self):
        """Save model state."""
        th.save({
            'vocabulary': self.vocabulary,
            'model': self.model.state_dict()
        }, polarity_classifier_dump_path)

    @staticmethod
    def load_model() -> 'PolarityClassifier':
        """Load pretrained model."""
        checkpoint = th.load(polarity_classifier_dump_path)
        model = checkpoint['model']
        classifier = PolarityClassifier(vocabulary=checkpoint['vocabulary'],
                                        emb_matrix=model['nn.embed.weight'])
        classifier.model.load_state_dict(model)
        return classifier
